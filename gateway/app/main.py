"""Application factory and lifespan.

Responsibilities (Phase 0/1):
  * configure logging
  * open shared singletons once (httpx client + Redis pool) and close them
    cleanly on shutdown — never per-request
  * wire the middleware stack in the correct order
  * mount routers

Later phases register the compliance engine, providers, audit worker, and
Prometheus metrics here.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.compliance.vault import InMemoryVault, VaultUnavailable

from app import __version__
from app.config import Settings, get_settings
from app.middleware.auth import AuthMiddleware
from app.middleware.logging import AccessLogMiddleware
from app.middleware.metrics import ActiveRequestsMiddleware
from app.middleware.request_context import RequestContextMiddleware
from app.audit.logger import AuditLogger
from app.compliance.engine import build_compliance_engine
from app.compliance.vault import RedisVault
from app.db.pool import build_repositories
from app.history.sink import HistorySink
from app.compliance.policy import load_all_policies
from app.providers.router import build_provider_registry, get_provider
from app.routers import admin, analyze, auth, chat, dashboard, health, history
from app.utils.http import build_async_client


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    log = logging.getLogger("gateway")

    # Shared, connection-pooled HTTP client (reused across all upstream calls).
    app.state.http_client = build_async_client(settings.upstream_timeout_seconds)
    # Redis client/pool. Probed with a quick PING at startup so we can fall back
    # to InMemoryVault immediately rather than discovering the problem at request time.
    redis_client = aioredis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=settings.redis_timeout_seconds,
        socket_timeout=settings.redis_timeout_seconds,
    )
    try:
        await redis_client.ping()
        log.info("gateway: Redis vault connected (%s)", settings.redis_url.split("@")[-1])
        app.state.redis = redis_client
        app.state.vault = RedisVault(
            redis_client,
            ttl_seconds=settings.vault_ttl_seconds,
            token_style=settings.vault_token_style,
            encryption_key=settings.vault_encryption_key,
        )
    except Exception as _redis_err:  # noqa: BLE001
        log.warning(
            "gateway: Redis unavailable (%s) — falling back to InMemoryVault. "
            "PHI tokens will NOT persist across requests or workers. "
            "Set REDIS_URL in your environment to enable the full vault.",
            _redis_err,
        )
        await redis_client.aclose()
        app.state.redis = None
        app.state.vault = InMemoryVault(token_style=settings.vault_token_style)
    app.state.engine = build_compliance_engine(settings, app.state.vault)
    app.state.provider = get_provider(settings, app.state.http_client)
    # Registries for per-request provider/policy selection (frontend selectors).
    app.state.providers = build_provider_registry(settings, app.state.http_client)
    app.state.policies = load_all_policies()
    app.state.audit = AuditLogger(settings.audit_log_path, settings.audit_queue_maxsize)
    await app.state.audit.start()

    # Persistence: Neon (or in-memory fallback) + off-hot-path history sink.
    users_repo, history_repo, db_pool = await build_repositories(settings)
    app.state.users = users_repo
    app.state.history_repo = history_repo
    app.state.db_pool = db_pool
    app.state.history_sink = HistorySink(history_repo)
    await app.state.history_sink.start()

    log.info(
        "gateway %s started: engine=%s provider=%s auth=%s",
        __version__,
        settings.detection_engine.value,
        settings.llm_provider.value,
        settings.auth_mode.value,
    )
    try:
        yield
    finally:
        await app.state.history_sink.stop()
        await app.state.audit.stop()
        if app.state.db_pool is not None:
            await app.state.db_pool.close()
        await app.state.http_client.aclose()
        if app.state.redis is not None:
            await app.state.redis.aclose()
        log.info("gateway shut down cleanly")


def create_app() -> FastAPI:
    settings = get_settings()
    _configure_logging(settings)
    settings.validate_for_runtime()  # fail fast on insecure prod config

    app = FastAPI(
        title="AI Privacy & Compliance Gateway",
        version=__version__,
        description="Privacy/compliance middleware between any client and any LLM provider.",
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(VaultUnavailable)
    async def _vault_unavailable(_request: Request, exc: VaultUnavailable) -> JSONResponse:
        # Redis (the privacy vault) is unreachable — fail clearly rather than
        # leak raw tokens or PHI to the client.
        return JSONResponse(
            status_code=503,
            content={"error": "privacy_vault_unavailable", "detail": "masking vault is unavailable"},
        )

    # Middleware: add_middleware wraps inside-out, so the LAST added is the
    # OUTERMOST. We want request-context outermost (ids available everywhere),
    # then access logging, then active-requests gauge, then auth innermost.
    app.add_middleware(AuthMiddleware, settings=settings)
    app.add_middleware(ActiveRequestsMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestContextMiddleware)
    # CORS must be the OUTERMOST layer so it answers browser preflight (OPTIONS)
    # and stamps headers on every response — including errors. Added last.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    # Prometheus: HTTP latency histogram + request counts, exposed at /metrics
    # (public path, so auth-exempt). Domain metrics live on the same registry.
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(analyze.router)
    app.include_router(chat.router)
    app.include_router(history.router)
    app.include_router(admin.router)
    app.include_router(dashboard.router)
    return app


app = create_app()
