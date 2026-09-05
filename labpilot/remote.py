# utils/remote.py
# ============================================================
# ROBUST REMOTE EXECUTION LAYER (Windows / PowerShell 5.1)
#
# Why this module exists:
#   The remote login shell is *Windows PowerShell 5.1*, not bash. The original
#   code sent `mkdir -p "path"` and `cd "path" && python train.py`. PowerShell
#   5.1 rejects `&&` outright ("The token '&&' is not a valid statement
#   separator") and `mkdir -p` is invalid too, so NOTHING ever ran remotely and
#   every experiment folder came back empty.
#
#   Quoting a Windows path containing spaces through ssh -> paramiko ->
#   PowerShell is also a losing game. So every command is marshalled as a
#   UTF-16LE base64 `-EncodedCommand`, which is completely quoting-proof.
# ============================================================
from __future__ import annotations

import base64
import os
import posixpath
import re
import socket
import time
from dataclasses import dataclass

import paramiko
from scp import SCPClient

from .config import (
    REMOTE_USER, REMOTE_HOST as REMOTE_IP, REMOTE_PORT,
    SSH_KEYS as SSH_KEY_CANDIDATES, MODEL_VRAM_MB as MIN_FREE_VRAM_MB,
)

SSH_CONNECT_TIMEOUT = 30
REMOTE_CMD_TIMEOUT = 120
GPU_WAIT_RETRIES = 10
GPU_WAIT_SECONDS = 60

# PowerShell serialises remote stderr as CLIXML. The progress records are pure
# noise, but the <S S="Error"> records carry the actual error message — an
# earlier version deleted the whole blob and so silently discarded every
# remote error, which made failures look like empty output.
_CLIXML_HEADER = re.compile(r"^#< CLIXML\s*", re.MULTILINE)
_CLIXML_ERROR = re.compile(r'<S S="(?:Error|Warning)">(.*?)</S>', re.DOTALL)
_CLIXML_OBJS = re.compile(r"<Objs\b.*?</Objs>", re.DOTALL)
_SSH_NOISE = re.compile(
    r"^\s*\*\*.*$|^\s*Warning: You are sending unauthenticated.*$", re.MULTILINE
)


def _decode_clixml_escapes(text: str) -> str:
    def sub(m):
        return chr(int(m.group(1), 16))
    return re.sub(r"_x([0-9A-Fa-f]{4})_", sub, text)


def _clean(text: str) -> str:
    """Strip transport noise while PRESERVING any PowerShell error text."""
    if "<Objs" in text:
        messages = []
        for blob in _CLIXML_OBJS.findall(text):
            for msg in _CLIXML_ERROR.findall(blob):
                messages.append(_decode_clixml_escapes(msg))
        text = _CLIXML_OBJS.sub("", text)
        if messages:
            text += "\n" + "".join(messages)
    text = _CLIXML_HEADER.sub("", text)
    text = _SSH_NOISE.sub("", text)
    return text.strip()


class RemoteError(RuntimeError):
    pass


@dataclass
class CmdResult:
    exit_status: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined(self) -> str:
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)

    @property
    def ok(self) -> bool:
        return self.exit_status == 0 and not self.timed_out


class RemoteBox:
    """A live SSH session to the Windows GPU box. Use as a context manager."""

    def __init__(self):
        self.ssh: paramiko.SSHClient | None = None

    # ---------- lifecycle ----------
    def connect(self, attempts: int = 3, backoff: float = 4.0) -> "RemoteBox":
        """Connect, retrying transient failures.

        The network link to the box can drop intermittently — a single timeout is
        usually not "the machine is off", it is a link that reconnects within
        seconds. Failing the whole research session on the first timeout wasted
        real work.
        """
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                return self._connect_once()
            except RemoteError as exc:
                last_exc = exc
                if attempt < attempts:
                    wait = backoff * attempt
                    print(f"   ⟳ SSH attempt {attempt}/{attempts} failed; "
                          f"retrying in {wait:.0f}s")
                    time.sleep(wait)
        raise last_exc

    def _connect_once(self) -> "RemoteBox":
        ssh = paramiko.SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        last_err = None
        # Try each candidate key explicitly, then fall back to agent/default keys.
        for key_path in [p for p in SSH_KEY_CANDIDATES if os.path.exists(p)] + [None]:
            try:
                ssh.connect(
                    REMOTE_IP,
                    port=REMOTE_PORT,
                    username=REMOTE_USER,
                    key_filename=key_path,
                    look_for_keys=key_path is None,
                    allow_agent=True,
                    timeout=SSH_CONNECT_TIMEOUT,
                    banner_timeout=SSH_CONNECT_TIMEOUT,
                    auth_timeout=SSH_CONNECT_TIMEOUT,
                )
                self.ssh = ssh
                return self
            except (paramiko.SSHException, socket.error, OSError) as exc:
                last_err = exc
                continue
        raise RemoteError(
            f"Could not SSH to {REMOTE_USER}@{REMOTE_IP}:{REMOTE_PORT}. "
            f"Tried keys {SSH_KEY_CANDIDATES}. Last error: {last_err}"
        )

    def close(self):
        if self.ssh:
            self.ssh.close()
            self.ssh = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()
        return False

    # ---------- command execution ----------
    def run_ps(self, script: str, timeout: int = REMOTE_CMD_TIMEOUT) -> CmdResult:
        """Run a PowerShell script block remotely, quoting-proof, with a timeout."""
        if not self.ssh:
            raise RemoteError("Not connected.")
        encoded = base64.b64encode(script.encode("utf-16-le")).decode()
        cmd = f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}"

        chan = self.ssh.get_transport().open_session()
        chan.settimeout(timeout)
        chan.exec_command(cmd)

        out, err = bytearray(), bytearray()
        deadline = time.monotonic() + timeout
        timed_out = False
        while True:
            if chan.recv_ready():
                out += chan.recv(65536)
            if chan.recv_stderr_ready():
                err += chan.recv_stderr(65536)
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
            if time.monotonic() > deadline:
                timed_out = True
                chan.close()
                break
            time.sleep(0.2)

        status = chan.recv_exit_status() if not timed_out else -1
        chan.close()
        return CmdResult(status, _clean(out.decode("utf-8", "replace")),
                         _clean(err.decode("utf-8", "replace")), timed_out)

    def mkdir(self, remote_path: str) -> None:
        """PowerShell has no `mkdir -p`; New-Item -Force is the equivalent."""
        # NOTE: New-Item takes -Path, not -LiteralPath (Test-Path does take
        # -LiteralPath). Getting this wrong silently created nothing.
        res = self.run_ps(
            f"New-Item -ItemType Directory -Force -Path '{remote_path}' | Out-Null\n"
            f"if (Test-Path -LiteralPath '{remote_path}') {{ 'DIR_OK' }} "
            f"else {{ Write-Error 'mkdir failed' }}"
        )
        if "DIR_OK" not in res.combined:
            raise RemoteError(f"Failed to create remote dir {remote_path}: {res.combined}")

    def run_python(self, remote_dir: str, script_name: str, timeout: int) -> CmdResult:
        """Run `python <script>` inside remote_dir. No `&&` — PowerShell can't."""
        ps = (
            f"Set-Location -LiteralPath '{remote_dir}'\n"
            f"$env:PYTHONUNBUFFERED = '1'\n"
            f"$env:PYTHONIOENCODING = 'utf-8'\n"
            f"& python '{script_name}' 2>&1 | ForEach-Object {{ \"$_\" }}\n"
            f"exit $LASTEXITCODE\n"
        )
        return self.run_ps(ps, timeout=timeout)

    # ---------- file transfer ----------
    def upload(self, local_path: str, remote_path: str) -> None:
        with SCPClient(self.ssh.get_transport(), socket_timeout=120) as scp:
            scp.put(local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> bool:
        try:
            with SCPClient(self.ssh.get_transport(), socket_timeout=120) as scp:
                scp.get(remote_path, local_path)
            return True
        except Exception:
            return False

    def remote_exists(self, remote_path: str) -> bool:
        res = self.run_ps(f"if (Test-Path -LiteralPath '{remote_path}') {{ 'YES' }} else {{ 'NO' }}")
        return "YES" in res.stdout

    # ---------- GPU gating ----------
    def gpu_free_mb(self) -> tuple[int, int]:
        """Returns (free_mb, total_mb). Raises if nvidia-smi is unusable."""
        res = self.run_ps(
            "nvidia-smi --query-gpu=memory.used,memory.total "
            "--format=csv,noheader,nounits"
        )
        line = next((l for l in res.stdout.splitlines() if "," in l), "")
        try:
            used, total = (int(x.strip()) for x in line.split(",")[:2])
        except ValueError:
            raise RemoteError(f"Could not read nvidia-smi output: {res.combined!r}")
        return total - used, total

    def wait_for_gpu(self, retries: int = GPU_WAIT_RETRIES,
                     wait_s: int = GPU_WAIT_SECONDS) -> bool:
        """Block until enough free VRAM. Returns False if it never frees up."""
        for attempt in range(1, retries + 1):
            free, total = self.gpu_free_mb()
            if free >= MIN_FREE_VRAM_MB:
                print(f"   ✅ GPU ready: {free}MB free of {total}MB.")
                return True
            print(f"   ⏳ GPU busy: only {free}MB free (need {MIN_FREE_VRAM_MB}MB). "
                  f"Attempt {attempt}/{retries}, waiting {wait_s}s...")
            if attempt < retries:
                time.sleep(wait_s)
        return False


def remote_join(*parts: str) -> str:
    return posixpath.join(*[p.replace("\\", "/").rstrip("/") for p in parts if p])
