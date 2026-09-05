# utils/codegen.py
# ============================================================
# TURNS LLM PROSE INTO A VERIFIED, RUNNABLE quantize() FUNCTION
#
# The original pipeline wrote the model's raw reply straight to train.py, so
# the deployed file began with "Here is the modified code:" followed by a
# ```python fence. Every remote run died with a SyntaxError on line 1.
#
# Here we (1) strip prose/fences, (2) keep only the function, (3) compile it,
# (4) execute it against toy tensors locally, and only then let it near the GPU.
# ============================================================
from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

from pathlib import Path as _Path

PROJECT_ROOT = _Path(__file__).resolve().parent.parent

TEMPLATE_PATH = PROJECT_ROOT / "labpilot" / "runners"

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

ALLOWED_IMPORTS = {"torch", "math", "torch.nn", "torch.nn.functional"}


class QuantizerRejected(ValueError):
    """The model's code is not usable; the message is fed back to it."""


def extract_code(reply: str) -> str:
    """Pull python out of a chat reply: fenced blocks first, else raw text."""
    blocks = _FENCE.findall(reply)
    if blocks:
        # Prefer the block that actually defines quantize().
        for b in blocks:
            if "def quantize" in b:
                return textwrap.dedent(b).strip()
        return textwrap.dedent(blocks[0]).strip()
    # No fences: drop leading prose lines until something looks like code.
    lines = reply.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import |from |def |@|#)", line):
            return "\n".join(lines[i:]).strip()
    return reply.strip()


def _strip_to_quantizer(code: str) -> str:
    """Keep imports + helper defs + quantize(); drop stray top-level statements
    (prints, training loops, `model = ...`) that the model likes to append."""
    tree = ast.parse(code)
    kept, saw_quantize = [], False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            kept.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            kept.append(node)
            if isinstance(node, ast.FunctionDef) and node.name == "quantize":
                saw_quantize = True
        elif isinstance(node, ast.Assign) and all(
            isinstance(t, ast.Name) and t.id.isupper() for t in node.targets
        ):
            kept.append(node)  # module-level constants are fine
        # everything else (calls, prints, loops) is discarded
    if not saw_quantize:
        raise QuantizerRejected(
            "Your code did not define a function named exactly `quantize`."
        )
    new = ast.Module(body=kept, type_ignores=[])
    return ast.unparse(ast.fix_missing_locations(new))


def _check_imports(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        for m in mods:
            root = m.split(".")[0]
            if m not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                raise QuantizerRejected(
                    f"You imported `{m}`, which does not exist on the worker. "
                    f"Only these are available: {sorted(ALLOWED_IMPORTS)}."
                )


def _run_selftest(code: str) -> dict:
    """Execute the quantizer on CPU toy tensors. Catches the failure modes the
    old pipeline only discovered after a remote crash."""
    import torch

    ns: dict = {}
    try:
        exec(compile(code, "<quantizer>", "exec"), ns)  # noqa: S102 - sandboxed by _check_imports
    except Exception as exc:
        raise QuantizerRejected(f"Your code failed to import/run: {type(exc).__name__}: {exc}")

    fn = ns.get("quantize")
    if not callable(fn):
        raise QuantizerRejected("`quantize` is not callable.")

    stats = {}
    for name, w in {
        "gaussian": torch.randn(16, 32),
        "small": torch.randn(8, 8) * 1e-4,
        "sparse": torch.randn(8, 8) * (torch.rand(8, 8) > 0.7),
    }.items():
        w = w.requires_grad_(False)
        try:
            out = fn(w)
        except Exception as exc:
            raise QuantizerRejected(
                f"quantize() raised on {name} input: {type(exc).__name__}: {exc}"
            )
        if not torch.is_tensor(out):
            raise QuantizerRejected(f"quantize() returned {type(out).__name__}, not a Tensor.")
        if out.shape != w.shape:
            raise QuantizerRejected(
                f"quantize() changed the shape on {name} input: "
                f"{tuple(w.shape)} -> {tuple(out.shape)}. It must be shape-preserving."
            )
        if not torch.isfinite(out).all():
            raise QuantizerRejected(
                f"quantize() produced NaN/Inf on {name} input. This is usually a "
                f"division by a scale that can be zero — add an epsilon."
            )
        for r in range(out.shape[0]):
            uniq = torch.unique(out[r])
            if uniq.numel() > 3:
                raise QuantizerRejected(
                    f"Row {r} of the {name} output has {uniq.numel()} distinct values. "
                    f"A ternary quantizer must emit at most 3 per row: {{-s, 0, +s}}."
                )
        if name == "gaussian":
            if float(out.abs().sum()) == 0.0:
                raise QuantizerRejected(
                    "quantize() zeroed every weight on normal input — the threshold "
                    "is far too aggressive."
                )
            nz = float((out != 0).float().mean())
            if nz < 0.05:
                raise QuantizerRejected(
                    f"Only {nz:.1%} of weights survived quantization. A usable ternary "
                    f"scheme keeps roughly 40-80% non-zero. Lower your threshold."
                )
            stats["nonzero_fraction"] = nz
            stats["mean_abs"] = float(out.abs().mean())
    return stats


def validate_quantizer(reply: str) -> tuple[str, dict]:
    """Full pipeline: reply -> clean, verified quantizer source. Raises
    QuantizerRejected with a message meant to be handed back to the model."""
    code = extract_code(reply)
    if not code.strip():
        raise QuantizerRejected("You returned no code at all.")
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise QuantizerRejected(f"Your code is not valid Python: {exc}")
    code = _strip_to_quantizer(code)
    _check_imports(code)
    if "import torch" not in code:
        code = "import torch\nimport torch.nn.functional as F\nimport math\n" + code
    stats = _run_selftest(code)
    return code, stats


def render_train_script(quantizer_code: str, profile: dict) -> str:
    """Splice the verified quantizer + run config into the fixed harness."""
    template = TEMPLATE_PATH.read_text()
    script = template.replace("__QUANTIZER__", quantizer_code)
    script = script.replace("__CONFIG_JSON__", json.dumps(profile))
    # Fail loudly rather than shipping a half-substituted file.
    for marker in ("__QUANTIZER__", "__CONFIG_JSON__"):
        if marker in script:
            raise RuntimeError(f"Template substitution failed for {marker}")
    compile(script, "train.py", "exec")  # syntax gate before deployment
    return script


def parse_result(log_text: str) -> dict | None:
    """Read the RESULT_JSON sentinel the harness emits. Last one wins."""
    found = None
    for line in log_text.splitlines():
        idx = line.find("RESULT_JSON:")
        if idx >= 0:
            try:
                found = json.loads(line[idx + len("RESULT_JSON:"):].strip())
            except json.JSONDecodeError:
                continue
    return found


def normalize_for_novelty(code: str) -> str:
    """A BEHAVIOURAL fingerprint of a quantizer.

    The repair loop hands the model a skeleton to fill in, and a weak model
    will return it essentially unchanged — sometimes with a cosmetic edit
    (splitting a line, renaming a local) that defeats any source-text
    comparison. So identity is decided by what the function *computes*: its
    output on a fixed probe tensor, rounded and hashed. Two rules that agree on
    the probe are the same rule, however differently they are written.
    """
    import hashlib

    import torch

    try:
        ns: dict = {}
        exec(compile(code, "<novelty>", "exec"), ns)  # noqa: S102
        fn = ns["quantize"]
        g = torch.Generator().manual_seed(20240501)
        probe = torch.randn(24, 48, generator=g)
        out = fn(probe)
        digest = hashlib.sha256(
            torch.round(out.detach() * 1e5).to(torch.int64).numpy().tobytes()
        ).hexdigest()
        return f"behaviour:{digest}"
    except Exception:
        # Unrunnable code cannot be fingerprinted; fall back to source identity.
        try:
            return "source:" + ast.unparse(ast.parse(code)).strip()
        except SyntaxError:
            return "source:" + code.strip()


SKELETON_QUANTIZER = normalize_for_novelty(
    "import torch\n"
    "def quantize(w):\n"
    "    thresh = 0.7 * w.abs().mean(dim=1, keepdim=True)\n"
    "    mask = (w.abs() > thresh).to(w.dtype)\n"
    "    scale = (w.abs() * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)\n"
    "    return torch.sign(w) * mask * scale.clamp_min(1e-8)\n"
)
