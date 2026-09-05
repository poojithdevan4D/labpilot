# labpilot/llm.py — talk to the box's model. Parallel where it helps, never on the
# laptop, and every request goes over the network to the GPU box.
from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import MODEL, WORKDIR  # noqa: F401


class LLMUnavailable(RuntimeError):
    """The box is not serving a model. Almost always: no session is open."""


def server_up(box) -> bool:
    r = box.run_ps('(Test-NetConnection localhost -Port 11434 '
                   '-WarningAction SilentlyContinue).TcpTestSucceeded', timeout=60)
    return "True" in r.combined


def ask(box, prompt: str, temperature: float = 0.2, timeout: int = 600,
        model: str = MODEL) -> str:
    """One request. The payload is uploaded as a FILE, never embedded in the
    PowerShell command — braces and quotes in a prompt break the parser."""
    payload = {"model": model, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        local = f.name
    remote = f"{WORKDIR}/fab_{Path(local).stem}.json"
    try:
        box.upload(local, remote)
        out = box.run_ps(f'''
try {{
  $r = Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post `
    -InFile "{remote}" -ContentType "application/json" -TimeoutSec {timeout}
  "__OK__"
  $r.response
}} catch {{
  "__ERR__ " + $_.Exception.Message
}}
Remove-Item "{remote}" -Force -EA SilentlyContinue
''', timeout=timeout + 60)
        text = out.combined.strip()
        # Without this, a dead server's error text gets returned as if it were a
        # model reply, and downstream code tries to parse it as Python.
        if "__ERR__" in text or "__OK__" not in text:
            raise LLMUnavailable(
                "no reply from the model server on the box. Is a session open? "
                "Run `labpilot/run.py status`, then `start`. "
                f"({text.split('__ERR__')[-1].strip()[:120]})")
        return text.split("__OK__", 1)[1].strip()
    finally:
        Path(local).unlink(missing_ok=True)


def ask_many(box_factory, prompts: list[str], workers: int = 4,
             temperature: float = 0.2, log=print) -> list[str]:
    """Stage-1 parallelism: many prompts against one loaded model.

    Each thread needs its own SSH connection — paramiko channels are not safe to
    share. `workers` should not exceed OLLAMA_NUM_PARALLEL on the box, or the
    requests just queue.
    """
    def one(i_p):
        i, p = i_p
        try:
            with box_factory() as b:
                r = ask(b, p, temperature)
            log(f"   [{i+1}/{len(prompts)}] done ({len(r)} chars)")
            return r
        except Exception as exc:  # noqa: BLE001
            log(f"   [{i+1}/{len(prompts)}] FAILED {type(exc).__name__}")
            return ""

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, enumerate(prompts)))
