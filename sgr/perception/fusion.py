"""Fast detector + Gemma fusion (spec §7.3). Shared by mock and local/remote modes.

- Gemma agrees with the detector label: combined = 0.4 * score_fast + 0.6 * gemma.confidence
- Gemma disagrees:                      combined = 0.3 * score_fast
- combined >= 0.7 -> confirmed
- otherwise -> low_conf: the swarm takes a closer look (request_relook, at most 2)
- still low after the re-looks -> rejected (Gemma disagrees) or unconfirmed (Gemma agrees weakly)
- a clear disagreement (combined < 0.25) on a re-look -> rejected at once (the decoy beat, §5)
- no Gemma answer (timeout / down) -> unconfirmed, combined = score_fast (a second view must agree)
"""

from __future__ import annotations

from sgr.contracts import DetectionStatus, GemmaResult, Label

CONFIRM_AT = 0.7
REJECT_EARLY_BELOW = 0.25
MAX_RELOOKS = 2


def agrees(label: Label, g: GemmaResult) -> bool:
    if label == "person":
        return g.person
    return g.fire  # "fire" (and a "decoy" label, which the detector reports as fire)


def fuse(label: Label, score_fast: float, gemma: GemmaResult | None, relooks: int) -> tuple[float, DetectionStatus]:
    """-> (combined, status)."""
    if gemma is None:
        return round(score_fast, 3), "unconfirmed"
    ok = agrees(label, gemma)
    combined = 0.4 * score_fast + 0.6 * gemma.confidence if ok else 0.3 * score_fast
    combined = round(combined, 3)
    if combined >= CONFIRM_AT:
        return combined, "confirmed"
    if not ok and relooks >= 1 and combined < REJECT_EARLY_BELOW:
        return combined, "rejected"
    if relooks >= MAX_RELOOKS:
        return combined, ("unconfirmed" if ok else "rejected")
    return combined, "low_conf"
