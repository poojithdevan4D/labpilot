# labpilot/gate.py — what earns GPU time.
#
# GPU hours are the scarcest thing here. Everything below is checkable without
# spending any, so anything that fails is caught for free.
from __future__ import annotations

from dataclasses import dataclass, field

from . import store
from .provenance import check as prov_check


@dataclass
class Decision:
    allowed: bool
    reasons: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    fingerprint: str = ""

    def report(self) -> str:
        head = "GPU RUN APPROVED" if self.allowed else "GPU RUN BLOCKED"
        L = [head]
        L += [f"  BLOCKED: {r}" for r in self.reasons]
        L += [f"  note   : {w}" for w in self.warnings]
        return "\n".join(L)


def evaluate(code: str, paper_text: str | None = None, novel: bool = False,
             hypothesis: str = "", expected_minutes: float | None = None,
             force: bool = False) -> Decision:
    """Four gates, cheapest first. Any failure means no GPU time is spent."""
    d = Decision(allowed=True)

    # --- 1. does it even work? (CPU, milliseconds) ---
    try:
        from .verify import QuantizerRejected, validate_quantizer
        clean, stats = validate_quantizer(code)
        d.warnings.append(f"verified on CPU · non-zero {stats['nonzero_fraction']:.0%}")
        code = clean
    except QuantizerRejected as exc:
        d.allowed = False
        d.reasons.append(f"fails CPU verification — {exc}")
        return d
    except Exception as exc:  # noqa: BLE001
        d.allowed = False
        d.reasons.append(f"could not verify: {type(exc).__name__}: {exc}")
        return d

    # --- 2. have we already run exactly this? ---
    fp = store.method_fingerprint(code)
    d.fingerprint = fp
    prior = store.already_run(fp)
    if prior and not force:
        d.allowed = False
        d.reasons.append(
            f"already run on {prior.get('date')} — result was "
            f"'{prior.get('result')}' (metric {prior.get('metric')}). "
            f"Re-running spends GPU hours to learn nothing.")
        return d
    seen = store.seen_method(code)
    if seen and not prior:
        d.warnings.append(f"same method as '{seen.get('name')}' "
                          f"({seen.get('date')}) but never run")

    # --- 3. does the code match the paper it claims to come from? ---
    p = prov_check(code, paper_text, novel=novel)
    if not p.gpu_worthy and not force:
        d.allowed = False
        d.reasons.append(
            f"provenance {p.status} — the implementation does not match its "
            f"source. Fix it or declare it novel.")
        d.warnings.append(p.report().replace("\n", "\n  "))
        return d
    d.warnings.append(f"provenance {p.status}")

    # --- 4. is there a question this answers? ---
    if not hypothesis and not force:
        d.allowed = False
        d.reasons.append(
            "no hypothesis given. A run whose outcome would not change any "
            "decision is not worth GPU time. State what you expect and what "
            "would refute it.")
        return d
    if hypothesis:
        d.warnings.append(f"hypothesis: {hypothesis[:90]}")
    if expected_minutes:
        d.warnings.append(f"expected cost: ~{expected_minutes:.0f} min")
    return d
