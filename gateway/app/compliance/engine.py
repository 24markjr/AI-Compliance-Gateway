"""Compliance Engine — the heart of the gateway.

Orchestration only; it owns no detection logic. For each request it:

    1. checks for prompt injection (flag, or block per policy)
    2. runs all detectors, resolves overlapping spans
    3. asks the Policy Engine what to do with each entity
       - any BLOCK  -> abort, nothing reaches the provider
       - MASK       -> tokenise via the Redis vault
       - ALLOW      -> leave in place
    4. returns a ComplianceResult the router acts on

Rehydration of the provider's reply is done by the router via masker.rehydrate.
"""

from __future__ import annotations

import asyncio

from app.compliance.detectors.base import Detector
from app.compliance.detectors.injection import InjectionDetector
from app.compliance.detectors.patterns_in import build_pii_detectors
from app.compliance.detectors.secrets import build_secret_detectors
from app.compliance.masker import apply_masking
from app.compliance.overlap import resolve_overlaps
from app.compliance.policy import Policy, load_policy
from app.compliance.types import Action, ComplianceResult, PIIEntity
from app.compliance.vault import InMemoryVault, RedisVault
from app.config import DetectionEngine, Settings


class ComplianceEngine:
    def __init__(
        self,
        detectors: list[Detector],
        injection: InjectionDetector,
        policy: Policy,
        vault: RedisVault | InMemoryVault,
        *,
        offload_detection: bool = False,
    ) -> None:
        self._detectors = detectors
        self._injection = injection
        self._policy = policy
        self._vault = vault
        # Presidio (Phase 5) is CPU-heavy and must run off the event loop.
        self._offload = offload_detection

    @property
    def policy(self) -> Policy:
        return self._policy

    def _detect_all(self, text: str) -> list[PIIEntity]:
        found: list[PIIEntity] = []
        for d in self._detectors:
            found.extend(d.detect(text))
        return found

    async def process(
        self, text: str, session_id: str, policy: Policy | None = None
    ) -> ComplianceResult:
        pol = policy or self._policy
        violations: list[str] = []

        # 1. Prompt injection.
        injection_hits = self._injection.detect(text)
        injection_flag = bool(injection_hits)
        if injection_flag:
            violations.extend(f"injection:{h}" for h in injection_hits)
            if pol.injection_action is Action.block:
                return ComplianceResult(
                    masked_text=text,
                    session_id=session_id,
                    violations=violations,
                    injection_flag=True,
                    blocked=True,
                    reason="prompt injection detected",
                )

        # 2. Detect + resolve overlaps (offloaded when using a heavy engine).
        if self._offload:
            entities = await asyncio.to_thread(self._detect_all, text)
        else:
            entities = self._detect_all(text)
        entities = resolve_overlaps(entities)

        # 3. Policy decisions.
        to_mask: list[PIIEntity] = []
        blocked_labels: list[str] = []
        for e in entities:
            decision = pol.decide(e.label)
            if decision.action is Action.block:
                blocked_labels.append(e.label)
                violations.append(f"policy:{decision.rule}")
            elif decision.action is Action.mask:
                to_mask.append(e)

        if blocked_labels:
            return ComplianceResult(
                masked_text=text,
                entities=entities,
                session_id=session_id,
                violations=violations,
                injection_flag=injection_flag,
                blocked=True,
                reason=f"blocked entities: {', '.join(sorted(set(blocked_labels)))}",
            )

        # 4. Mask the permitted entities.
        masked_text = await apply_masking(text, to_mask, self._vault, session_id)
        return ComplianceResult(
            masked_text=masked_text,
            entities=to_mask,
            session_id=session_id,
            violations=violations,
            injection_flag=injection_flag,
            blocked=False,
        )


def build_compliance_engine(settings: Settings, vault: RedisVault | InMemoryVault) -> ComplianceEngine:
    """Assemble the engine from configuration.

    Regex engine is always available (default + fallback). Presidio is wired in
    Phase 5; selecting it before then falls back to regex with the same patterns.
    """
    detectors: list[Detector] = []
    detectors.extend(build_secret_detectors())
    detectors.extend(build_pii_detectors())

    offload = False
    if settings.detection_engine is DetectionEngine.presidio:
        try:
            from app.compliance.detectors.presidio_engine import build_presidio_detectors

            detectors = [
                *build_secret_detectors(),
                *build_presidio_detectors(
                    settings.presidio_spacy_model, settings.presidio_score_threshold
                ),
            ]
            offload = True
        except Exception:  # noqa: BLE001 — missing package OR missing spaCy model
            import logging

            logging.getLogger("gateway").warning(
                "Presidio engine unavailable; falling back to regex engine", exc_info=True
            )

    policy = load_policy(settings.policy_file)
    return ComplianceEngine(
        detectors=detectors,
        injection=InjectionDetector(),
        policy=policy,
        vault=vault,
        offload_detection=offload,
    )
