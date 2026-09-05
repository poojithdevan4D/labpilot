# labpilot/session.py — the box is SHARED. This owns starting and stopping cleanly.
#
# Design rule: nothing of ours runs unless a session is explicitly open, and a
# session leaves the GPU exactly as it found it. No logon tasks, no daemons, no
# idle model squatting on VRAM while a teammate wants the card.
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import IDLE_CEILING_MB as _IDLE, MODEL_VRAM_MB as _VRAM, OLLAMA_EXE

PORT = 11434

# A teammate's job, or just the desktop. Below this the card is effectively idle.
IDLE_CEILING_MB = _IDLE
# What our own model needs resident.
MODEL_VRAM_MB = _VRAM


# Windows desktop processes (dwm, explorer, Edge WebView, SearchHost) can hold
# ~10GB of VRAM on this box when someone is logged in at the console. That is
# annoying but it is NOT a colleague running a job, and refusing to start
# because of it would make the labpilot unusable. Classify by process, not by
# total bytes.
DESKTOP_PROCS = (
    "dwm.exe", "explorer.exe", "ShellHost", "SearchHost", "StartMenu",
    "LockApp", "CrossDeviceResume", "msedgewebview2", "ApplicationFrameHost",
    "ShellExperienceHost", "TextInputHost", "SystemSettings", "WidgetService",
    "NVIDIA", "GameBar", "PhoneExperienceHost", "WhatsApp", "Lively",
    "EpicGames", "Docker", "Code.exe", "chrome.exe", "msedge.exe",
)
COMPUTE_HINTS = ("python", "ollama", "llama", "torch", "train")
# Ollama spawns llama-server from inside its own install directory. That is OUR
# process, not a colleague's, and treating it as theirs makes the labpilot refuse
# to start because of its own leftovers.
OURS_PATH_HINT = "\\ollama\\"


@dataclass
class GpuState:
    used_mb: int
    total_mb: int
    ollama_procs: int
    other_python: int
    compute_procs: tuple = ()

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb

    @property
    def looks_idle(self) -> bool:
        return self.used_mb <= IDLE_CEILING_MB

    @property
    def someone_else_working(self) -> bool:
        """A real compute job that is not ours.

        Decided by which PROCESSES hold the card, not by how many bytes are in
        use: a logged-in desktop can hold 10GB without anyone doing any work.
        """
        theirs = [p for p in self.compute_procs if not p.startswith("OURS:")]
        return bool(theirs)

    @property
    def our_orphans(self) -> list:
        """Our own processes still holding the card — usually a failed stop."""
        return [p[5:] for p in self.compute_procs if p.startswith("OURS:")]

    @property
    def desktop_heavy(self) -> bool:
        return self.used_mb > IDLE_CEILING_MB and not self.someone_else_working


def probe(box) -> GpuState:
    out = box.run_ps(r'''
$g = (nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -split ','
$o = (Get-Process ollama -EA SilentlyContinue | Measure-Object).Count
$py = (Get-Process python -EA SilentlyContinue | Measure-Object).Count
"STATE|$($g[0].Trim())|$($g[1].Trim())|$o|$py"
"PROCS"
nvidia-smi --query-compute-apps=process_name --format=csv,noheader
''', timeout=90).combined

    line = next((l for l in out.splitlines() if l.startswith("STATE|")), "STATE|0|12227|0|0")
    _, u, t, o, py = line.split("|")

    procs = []
    seen = False
    for l in out.splitlines():
        if l.strip() == "PROCS":
            seen = True
            continue
        if not seen or not l.strip():
            continue
        name = l.strip().split("\\")[-1]
        if any(d.lower() in l.lower() for d in DESKTOP_PROCS):
            continue
        if not any(h in name.lower() for h in COMPUTE_HINTS):
            continue
        ours = OURS_PATH_HINT.lower() in l.lower()
        procs.append(("OURS:" if ours else "") + name)
    return GpuState(int(u), int(t), int(o), int(py), tuple(procs))


def start(box, log=print, force: bool = False) -> bool:
    """Open a session. Refuses if someone else is clearly using the card."""
    st = probe(box)
    log(f"GPU: {st.used_mb}MB used of {st.total_mb}MB · "
        f"ollama={st.ollama_procs} · compute jobs: "
        f"{', '.join(st.compute_procs) if st.compute_procs else 'none'}")

    if st.someone_else_working and not force:
        log(f"\nREFUSING TO START — a compute job is running that is not ours:")
        for p in st.compute_procs:
            log(f"    {p}")
        log("Wait for it to finish, or --force if you are sure it is stale.")
        return False

    if st.desktop_heavy:
        log(f"NOTE: {st.used_mb}MB is held by the logged-in desktop, not by a job.")
        log("      Closing browser/Edge windows on the box frees most of it.")

    if st.free_mb < MODEL_VRAM_MB and not force:
        log(f"\nREFUSING — only {st.free_mb}MB free, the model needs ~{MODEL_VRAM_MB}MB.")
        return False

    if st.our_orphans and st.ollama_procs == 0:
        log(f"cleaning up our own leftovers: {', '.join(st.our_orphans)}")
        stop(box, log=lambda *_: None)
        st = probe(box)
        log(f"  reclaimed -> {st.free_mb}MB free")

    if st.ollama_procs == 0:
        log("starting ollama (detached via WMI — SSH kills nohup on this box)")
        box.run_ps(rf'''
$exe = "$env:USERPROFILE\ollama\ollama.exe"
([WMICLASS]"\\.\ROOT\CIMV2:Win32_Process").Create("`"$exe`" serve") | Out-Null
''', timeout=90)
        for _ in range(15):
            time.sleep(3)
            r = box.run_ps(f'(Test-NetConnection localhost -Port {PORT} '
                           f'-WarningAction SilentlyContinue).TcpTestSucceeded', timeout=60)
            if "True" in r.combined:
                break
        else:
            log("server did not come up")
            return False
    log("session OPEN — ollama serving on 11434")
    return True


def evict(box, model: str, log=print) -> int:
    """Unload the model. Must be called before any GPU run."""
    box.run_ps(rf'''
$b = @{{model="{model}"; prompt="x"; keep_alive=0; stream=$false}} | ConvertTo-Json
try {{ Invoke-RestMethod -Uri "http://localhost:{PORT}/api/generate" -Method Post `
  -Body $b -ContentType "application/json" -TimeoutSec 120 | Out-Null }} catch {{}}
''', timeout=180)
    time.sleep(5)
    st = probe(box)
    log(f"model evicted · {st.free_mb}MB free for the run")
    return st.free_mb


def stop(box, log=print) -> None:
    """Close the session and leave the card as we found it.

    Killing `ollama.exe` alone is NOT enough: it spawns `llama-server.exe` as a
    child, and that child is what actually holds the VRAM. Orphaning it leaves
    ~10GB reserved on a shared card with no obvious owner — and it then looks
    like somebody else's job to the next courtesy check.
    """
    box.run_ps(r'''
Get-Process ollama -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue
# the backend child that holds the model
Get-Process llama-server -EA SilentlyContinue |
  Where-Object { $_.Path -like "*\ollama\*" } |
  Stop-Process -Force -EA SilentlyContinue
Start-Sleep -Seconds 4
''', timeout=120)
    st = probe(box)
    log(f"session CLOSED — GPU back to {st.used_mb}MB, ollama procs {st.ollama_procs}")
    if st.ollama_procs:
        log("WARNING: an ollama process survived; check the box.")
