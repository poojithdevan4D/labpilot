# labpilot/store.py — what we have already done. Survives sessions.
#
# Requirement: never repeat research. That needs three separate memories, because
# "already seen" means different things:
#   papers.jsonl   — read and judged; never read again
#   methods.jsonl  — implemented; matched by BEHAVIOUR, not by source text
#   runs.jsonl     — actually executed on the GPU, with the result
#   queries.jsonl  — searches already made, so sweeps do not re-cover ground
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

MEM = Path(__file__).resolve().parent / "memory"


class _Store:
    def __init__(self, name: str):
        self.path = MEM / f"{name}.jsonl"

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def add(self, rec: dict) -> None:
        """Append under an exclusive lock.

        Two people running the labpilot at once would otherwise interleave partial
        lines and corrupt the memory. JSONL append is atomic under flock on
        Linux; the lock makes it safe on any filesystem that honours it.
        """
        MEM.mkdir(parents=True, exist_ok=True)
        rec.setdefault("date", date.today().isoformat())
        line = json.dumps(rec) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass          # non-POSIX: append is still near-atomic for one line
            f.write(line)
            f.flush()

    def keys(self, field_name: str) -> set:
        return {r.get(field_name) for r in self.all() if r.get(field_name)}


papers = _Store("papers")
methods = _Store("methods")
runs = _Store("runs")
queries = _Store("queries")


# ---------------------------------------------------------------- papers
def seen_paper(ident: str) -> dict | None:
    for r in papers.all():
        if r.get("id") == ident:
            return r
    return None


def record_paper(ident: str, title: str, verdict: str, note: str = "",
                 answers: str = "") -> None:
    papers.add({"id": ident, "title": title, "verdict": verdict,
                "note": note[:400], "answers_question": answers})


# ---------------------------------------------------------------- methods
def method_fingerprint(code: str) -> str:
    """Behavioural identity: what the function COMPUTES, not how it is written.

    Source-text comparison fails immediately — a model rewrites the same rule
    with a renamed variable and it looks new. Two functions that agree on a
    fixed probe are the same method.
    """
    try:
        import torch
        ns: dict = {}
        exec(compile(code, "<fp>", "exec"), ns)  # noqa: S102
        fn = ns["quantize"]
        g = torch.Generator().manual_seed(20240501)
        out = fn(torch.randn(24, 48, generator=g))
        return hashlib.sha256(
            torch.round(out.detach() * 1e5).to(torch.int64).numpy().tobytes()
        ).hexdigest()[:16]
    except Exception:
        return "src:" + hashlib.sha256(code.encode()).hexdigest()[:16]


def seen_method(code: str) -> dict | None:
    fp = method_fingerprint(code)
    for r in methods.all():
        if r.get("fingerprint") == fp:
            return r
    return None


def record_method(name: str, code: str, source: str, provenance: str,
                  note: str = "") -> str:
    fp = method_fingerprint(code)
    methods.add({"fingerprint": fp, "name": name, "source": source,
                 "provenance": provenance, "note": note[:300], "code": code})
    return fp


# ---------------------------------------------------------------- runs
def record_run(fingerprint: str, script: str, result: str, metric=None,
               cost_minutes=None, metric_name: str = "metric") -> None:
    """`metric` is always lower-is-better so history can be sorted uniformly.
    `metric_name` says what it actually was, so a mixed history stays readable."""
    runs.add({"fingerprint": fingerprint, "script": script, "result": result,
              "metric": metric, "metric_name": metric_name,
              "cost_minutes": cost_minutes})


def already_run(fingerprint: str) -> dict | None:
    for r in runs.all():
        if r.get("fingerprint") == fingerprint:
            return r
    return None


# ---------------------------------------------------------------- queries
def record_query(q: str, found: int, new: int) -> None:
    queries.add({"query": q, "found": found, "new": new})


def query_yield(q: str) -> float | None:
    """How much NEW material this query produced last time. A query that has
    stopped producing anything should not be run again."""
    hits = [r for r in queries.all() if r.get("query") == q]
    if not hits:
        return None
    last = hits[-1]
    return last["new"] / max(last["found"], 1)


def summary() -> dict:
    return {"papers": len(papers.all()), "methods": len(methods.all()),
            "runs": len(runs.all()), "queries": len(queries.all())}
