# labpilot/config.py — everything site-specific, in one place.
#
# The labpilot itself contains no lab's details. Point it at your own box with a
# labpilot.toml or environment variables and nothing else changes.
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_CFG = ROOT / "labpilot.toml"


def _load() -> dict:
    if not _CFG.exists():
        return {}
    try:
        import tomllib
        return tomllib.loads(_CFG.read_text(encoding="utf-8"))
    except Exception:
        return {}


_c = _load()


def _get(section: str, key: str, env: str, default):
    if os.environ.get(env):
        v = os.environ[env]
        return type(default)(v) if not isinstance(default, str) else v
    return _c.get(section, {}).get(key, default)


# ---------------------------------------------------------------- the box
REMOTE_USER = _get("box", "user", "LABPILOT_USER", "CHANGE-ME")
REMOTE_HOST = _get("box", "host", "LABPILOT_HOST", "127.0.0.1")
REMOTE_PORT = int(_get("box", "port", "LABPILOT_PORT", 22))
SSH_KEYS = [os.path.expanduser(p) for p in
            (_c.get("box", {}).get("ssh_keys")
             or [os.environ.get("LABPILOT_SSH_KEY", "~/.ssh/id_ed25519"),
                 "~/.ssh/id_ed25519", "~/.ssh/id_rsa"])]

# Where jobs run on the box. Must exist or be creatable.
WORKDIR = _get("box", "workdir", "LABPILOT_WORKDIR",
               "C:/Users/CHANGE-ME/labpilot-work")
OLLAMA_EXE = _get("box", "ollama_exe", "LABPILOT_OLLAMA",
                  r"$env:USERPROFILE\ollama\ollama.exe")

# ---------------------------------------------------------------- model
MODEL = _get("model", "name", "LABPILOT_MODEL", "qwen2.5-coder:14b")
MODEL_VRAM_MB = int(_get("model", "vram_mb", "LABPILOT_MODEL_VRAM", 9900))
PARALLEL = int(_get("model", "parallel", "LABPILOT_PARALLEL", 4))

# ---------------------------------------------------------------- policy
IDLE_CEILING_MB = int(_get("policy", "idle_ceiling_mb", "LABPILOT_IDLE_CEILING", 1500))
DEFAULT_PROFILE = _get("policy", "profile", "LABPILOT_PROFILE",
                       str(ROOT / "profiles" / "example-llm-quantization.yaml"))


def configured() -> bool:
    """False until someone has actually pointed this at their own machine."""
    return "CHANGE-ME" not in f"{REMOTE_USER}{WORKDIR}" and REMOTE_HOST != "127.0.0.1"


def describe() -> str:
    return (f"box      {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_PORT}\n"
            f"workdir  {WORKDIR}\n"
            f"model    {MODEL} (~{MODEL_VRAM_MB}MB, {PARALLEL} parallel)\n"
            f"profile  {DEFAULT_PROFILE}\n"
            f"config   {_CFG if _CFG.exists() else '(none)'}"
            + ("" if configured() else
               "\n\nNOT CONFIGURED — copy labpilot.toml.example to labpilot.toml "
               "and edit it,\nor set LABPILOT_HOST / LABPILOT_USER / LABPILOT_WORKDIR."))
