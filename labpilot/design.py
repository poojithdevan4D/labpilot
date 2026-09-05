# labpilot/design.py — generate many, verify all, promote one.
#
# Two things were wrong with generating a single method:
#
#   1. CPU verification costs milliseconds. Generating one candidate through a
#      free gate wastes the gate. Generate N in parallel, verify all N, keep the
#      best — the extra cost is LLM time on a machine that is otherwise idle.
#
#   2. Nothing learned from previous runs. A pipeline that ignores its own
#      results is automation, not research. `context_from_history` feeds what
#      actually worked back into the next prompt, which is the difference.
from __future__ import annotations

from dataclasses import dataclass, field

from . import journal, store
from .provenance import check as prov_check


@dataclass
class Candidate:
    code: str
    fingerprint: str
    stats: dict = field(default_factory=dict)
    provenance: str = ""
    novel_here: bool = True          # not already in our memory
    score: float = 0.0
    note: str = ""


def context_from_history(limit: int = 6) -> str:
    """What we already know, in a form the model can use.

    This is the feedback loop. Without it every design starts from zero and the
    system relearns the same dead ends.
    """
    runs = [r for r in store.runs.all() if r.get("metric") is not None]
    if not runs:
        return ""
    by_method = {m["fingerprint"]: m for m in store.methods.all()}
    scored = sorted(runs, key=lambda r: r["metric"])[:limit]

    lines = ["What we have already measured (lower is better):"]
    for r in scored:
        m = by_method.get(r["fingerprint"], {})
        name = r.get("metric_name", "metric")
        val = r["metric"]
        shown = (f"{1-val:.3f} retention" if name == "1-retention"
                 else f"{val:.1f} {name}")
        lines.append(f"  - {m.get('name','(unknown)')[:55]} -> {shown}")
    tried = [m.get("name", "")[:50] for m in store.methods.all()][-8:]
    if tried:
        lines.append("\nAlready implemented (do NOT repeat these):")
        lines += [f"  - {t}" for t in tried]
    return "\n".join(lines)


def score(cand: Candidate, target_nonzero: tuple = (0.40, 0.80)) -> float:
    """Rank verified candidates without spending GPU time.

    Everything here is a proxy. It decides which candidate is worth a run, not
    which is better — only the GPU can say that.
    """
    s = 0.0
    nz = cand.stats.get("nonzero_fraction", 0)
    lo, hi = target_nonzero
    s += 3.0 if lo <= nz <= hi else -2.0 * min(abs(nz - lo), abs(nz - hi)) * 5
    if cand.novel_here:
        s += 4.0                       # something we have never run is worth more
    if cand.provenance == "MATCHED":
        s += 3.0
    elif cand.provenance == "NOVEL":
        s += 1.5
    elif cand.provenance in ("PARTIAL", "MISMATCH"):
        s -= 4.0
    # Prefer simple, vectorised implementations: a python loop over rows is slow
    # and is where most generated code goes wrong.
    if "for " in cand.code:
        s -= 1.5
    s -= 0.002 * len(cand.code)
    return s


def evaluate_batch(codes: list[str], paper_text=None, novel=False,
                   log=print) -> list[Candidate]:
    """Verify every candidate on CPU. Free, so do it for all of them."""
    from .verify import QuantizerRejected, validate_quantizer

    out = []
    for i, raw in enumerate(codes, 1):
        if not raw:
            continue
        try:
            clean, stats = validate_quantizer(raw)
        except QuantizerRejected as exc:
            log(f"   [{i}] rejected: {str(exc)[:76]}")
            journal.write("design.attempt", attempt=i, verified=False,
                          reason=str(exc)[:120])
            continue
        except Exception as exc:  # noqa: BLE001
            log(f"   [{i}] rejected: {type(exc).__name__}")
            continue

        fp = store.method_fingerprint(clean)
        if any(c.fingerprint == fp for c in out):
            log(f"   [{i}] duplicate of an earlier candidate in this batch")
            continue
        seen = store.seen_method(clean)
        p = prov_check(clean, paper_text, novel=novel)
        c = Candidate(clean, fp, stats, p.status, seen is None)
        c.score = score(c)
        c.note = (f"non-zero {stats['nonzero_fraction']:.0%} · {p.status}"
                  + ("" if c.novel_here else f" · SEEN as '{seen.get('name','?')[:30]}'"))
        out.append(c)
        log(f"   [{i}] verified · {c.note} · score {c.score:.1f}")
        journal.write("design.attempt", attempt=i, verified=True,
                      nonzero=stats["nonzero_fraction"], provenance=p.status)

    return sorted(out, key=lambda c: -c.score)
