# labpilot/provenance.py — does this code actually match the paper it came from?
#
# The expensive failure is not code that crashes; the verifier catches that. It
# is code that runs, looks plausible, and implements something the paper never
# said. That burns a GPU run and produces a number attributed to the wrong
# method.
#
# So: extract the constants and operations the PAPER states, check the CODE
# contains them, and refuse to spend GPU time on an unexplained mismatch.
#
# A genuinely novel method has no paper to match. That is fine — it is labelled
# NOVEL and the human decides. What is never fine is code SILENTLY diverging
# from the source it claims.
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Numeric constants that matter in a quantization rule: thresholds, scale
# factors, exponents. Integers like "2" or years are too common to be evidence.
_CONST = re.compile(r"(?<![\w.])(\d\.\d{1,3})(?![\w])")

# Operations a quantization rule is built from, and how a paper says them.
_OPS = {
    "mean_abs":   [r"mean\s*(of the )?absolute", r"absmean", r"E\s*\[\s*\|", r"mean\(\|"],
    "std":        [r"standard deviation", r"\bstd\b", r"\bsigma\b", r"σ"],
    "max_abs":    [r"absmax", r"maximum absolute", r"max\s*\|"],
    "median":     [r"median"],
    "quantile":   [r"quantile", r"percentile", r"top-?k"],
    "rms":        [r"root mean square", r"\bRMS\b", r"sqrt.*mean.*squar"],
    "threshold":  [r"threshold", r"\bdelta\b", r"Δ", r"clip"],
    "per_row":    [r"per-?row", r"per-?channel", r"per output channel", r"row-?wise"],
    "per_tensor": [r"per-?tensor", r"layer-?wise", r"global scale"],
    "per_group":  [r"per-?group", r"group-?wise", r"block-?wise", r"group size"],
    "learnable":  [r"learnable", r"learned scal", r"trainable scal"],
    "sign":       [r"\bsign\b", r"sgn"],
}

_CODE_OPS = {
    "mean_abs":   [r"\.abs\(\)[^\n]*\.mean\(", r"abs\(\)\.mean"],
    "std":        [r"\.std\("],
    "max_abs":    [r"\.abs\(\)[^\n]*\.(a?max)\(", r"\.amax\("],
    "median":     [r"\.median\("],
    "quantile":   [r"\.quantile\(", r"\.topk\(", r"\.kthvalue\("],
    "rms":        [r"\.pow\(2\)[^\n]*\.sqrt\(", r"\.sqrt\(\)", r"\*\*\s*2[^\n]*sqrt"],
    "threshold":  [r"thresh", r"\.clamp\(", r">\s*t", r"mask"],
    "per_row":    [r"dim\s*=\s*1", r"dim\s*=\s*-1", r"keepdim\s*=\s*True"],
    "per_tensor": [r"\.mean\(\)\s*$", r"\.abs\(\)\.mean\(\)", r"\.max\(\)"],
    "per_group":  [r"reshape\(-1,", r"view\(-1,", r"group"],
    "learnable":  [r"nn\.Parameter", r"requires_grad"],
    "sign":       [r"torch\.sign\("],
}


@dataclass
class Provenance:
    status: str                      # MATCHED | PARTIAL | MISMATCH | NOVEL
    matched_ops: list = field(default_factory=list)
    missing_ops: list = field(default_factory=list)
    extra_ops: list = field(default_factory=list)
    matched_consts: list = field(default_factory=list)
    missing_consts: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def gpu_worthy(self) -> bool:
        """Only MATCHED or an explicitly declared NOVEL method earns GPU time."""
        return self.status in ("MATCHED", "NOVEL")

    def report(self) -> str:
        L = [f"provenance: {self.status}"]
        if self.matched_ops:
            L.append(f"  operations confirmed : {', '.join(self.matched_ops)}")
        if self.missing_ops:
            L.append(f"  paper says, code lacks: {', '.join(self.missing_ops)}")
        if self.extra_ops:
            L.append(f"  code does, paper never mentions: {', '.join(self.extra_ops)}")
        if self.matched_consts:
            L.append(f"  constants confirmed  : {', '.join(self.matched_consts)}")
        if self.missing_consts:
            L.append(f"  paper constants absent from code: "
                     f"{', '.join(self.missing_consts[:6])}")
        L += [f"  {n}" for n in self.notes]
        return "\n".join(L)


def _find(patterns, text) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def check(code: str, paper_text: str | None, novel: bool = False) -> Provenance:
    """Compare a generated implementation against its source paper."""
    if novel or not paper_text:
        return Provenance(
            status="NOVEL",
            notes=["no source paper — this is an original method.",
                   "Nothing to verify against; the result stands on its own."])

    paper_ops = {k for k, pats in _OPS.items() if _find(pats, paper_text)}
    code_ops = {k for k, pats in _CODE_OPS.items() if _find(pats, code)}

    # Granularity is exclusive: a rule is per-row OR per-tensor OR per-group.
    gran = {"per_row", "per_tensor", "per_group"}
    matched = sorted(paper_ops & code_ops)
    missing = sorted((paper_ops - code_ops) - gran)
    extra = sorted((code_ops - paper_ops) - gran - {"threshold", "sign"})

    paper_consts = set(_CONST.findall(paper_text))
    code_consts = set(_CONST.findall(code))
    # Only constants the paper actually emphasises near a threshold/scale word.
    key_consts = {c for c in paper_consts
                  if re.search(rf"(threshold|scale|factor|delta|Δ|alpha)[^.]{{0,60}}{re.escape(c)}"
                               rf"|{re.escape(c)}[^.]{{0,60}}(threshold|scale|factor)",
                               paper_text, re.IGNORECASE)}
    m_const = sorted(key_consts & code_consts)
    miss_const = sorted(key_consts - code_consts)

    notes = []
    gran_paper = paper_ops & gran
    gran_code = code_ops & gran
    if gran_paper and gran_code and not (gran_paper & gran_code):
        notes.append(f"GRANULARITY MISMATCH: paper says {sorted(gran_paper)}, "
                     f"code does {sorted(gran_code)}")

    if not paper_ops:
        status = "NOVEL"
        notes.append("no recognisable method description found in the source text; "
                     "treating as unverifiable rather than as a match.")
    elif notes or (missing and not matched):
        status = "MISMATCH"
    elif missing or miss_const or extra:
        status = "PARTIAL"
    else:
        status = "MATCHED"

    return Provenance(status, matched, missing, extra, m_const, miss_const, notes)
