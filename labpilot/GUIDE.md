# How to use this — the short version

## The rule

**Nothing runs unless you start it. It stops when you're done.**

No daemon. No auto-start. No background process. If you haven't typed a command,
the box has nothing of yours on it.

## Everyday use

```bash
cd /path/to/this/repo

# 1. is the box free?
./venv/bin/python fabric/run.py status

# 2. do a full pass (finds papers, reads them, writes a method)
./venv/bin/python fabric/run.py cycle "ternary speech recognition"

# that's it — the GPU is released automatically when it finishes
```

`cycle` opens a session, works, and always closes. Ctrl+C also releases the card.

## If you want to stay open longer

```bash
./venv/bin/python fabric/run.py start          # open
./venv/bin/python fabric/run.py discover ...   # as many steps as you like
./venv/bin/python fabric/run.py design "..."
./venv/bin/python fabric/run.py gpu train.py
./venv/bin/python fabric/run.py stop           # YOU must close this one
```

Run as long as you need. Just remember to `stop`.

## What each step does

| step | what happens | GPU |
|---|---|---|
| `status` | who's using the box | none |
| `discover` | searches OpenAlex/arXiv/EuropePMC, then reads the best abstracts **4 at a time** | 9.6 GB |
| `design` | writes a method, then **checks it on CPU** — bad code never reaches the GPU | 9.6 GB |
| `gpu <script>` | unloads the model first, then runs your script with the full card | 11.6 GB |
| `stop` | releases everything | → 0.6 GB |

The model and a training run **cannot both fit**. `gpu` unloads the model for you.

## It won't disturb your teammates

`start` refuses if someone else is running a real job:

```
REFUSING TO START — a compute job is running that is not ours:
    llama-server.exe
```

It looks at *which processes* hold the card, not how much memory is used — the
Windows desktop alone can hold 10 GB without anyone working.

If you're sure it's stale: `start --force`.

## Where things end up

```
fabric/work/discover.md    paper notes from the last discover
fabric/work/method.py      the last verified method
fabric/work/gpu_run.log    the last GPU run's output
```

## If something goes wrong

```bash
./venv/bin/python fabric/run.py stop    # always safe, always releases the card
```

If the box seems stuck, `status` tells you what's holding it.

## What's on the box

- Ollama in your user directory on the box — started only by `fabric start`
- `qwen2.5-coder:14b` (9 GB) — the model that writes code
- Nothing auto-starts. Nothing survives a `stop`.

## Two things to know

**Your laptop is the boss, the box does the work.** Claude plans and debugs;
the GPU box runs the model and the training. That keeps Claude usage small.

**Verification is free, GPU time isn't.** `design` checks code on CPU first —
it caught invented functions, wrong shapes and silent collapse before any run.
Trust that gate; it's why a bad idea costs seconds instead of hours.

---

# The five guarantees

## 1. It never repeats research

Three memories in `fabric/memory/`, kept across sessions:

- **papers** — read and judged. `discover` skips them silently.
- **methods** — matched by **what the code computes**, not by its text. Rename a
  variable and it is still recognised as the same method.
- **runs** — what actually reached the GPU, and what came back.
- **queries** — a search that stopped finding anything new is not run again.

## 2. It targets your actual questions

Searches are built from `open_questions` in your profile, not from generic terms.
"ternary" alone returns materials science; *"Has anyone quantized a speech model
to 1.58-bit?"* returns the thing you need.

Each paper is then asked which of your questions it answers. If the answer is
NONE, it is recorded as off-target and never read again.

```bash
fabric/run.py discover --question prior_art     # target one question
```

## 3. Nothing low-value reaches the GPU

Four gates, all free, all before any GPU time:

| gate | blocks |
|---|---|
| CPU verification | code that does not run, or is not actually ternary |
| already-run | a method whose result you already have |
| **provenance** | code that does not match the paper it came from |
| hypothesis | a run whose outcome would not change any decision |

```
GPU RUN BLOCKED
  BLOCKED: already run on 2026-09-03 — result was 'completed' (metric 58.3).
           Re-running spends GPU hours to learn nothing.
```

## 4. Generated code must match its source

```bash
fabric/run.py design "..." --paper paper.txt --hypothesis "..."
```

It compares the operations and constants the paper states against what the code
does:

| case | verdict |
|---|---|
| faithful implementation | **MATCHED** → allowed |
| paper says 0.7, code uses 0.35 | PARTIAL → blocked |
| paper says per-row, code does per-tensor | MISMATCH → blocked |
| paper says mean-abs, code uses std | PARTIAL → blocked |
| your own original idea (`--novel`) | **NOVEL** → allowed |

A new technique with nothing to verify against is fine. Code silently diverging
from its source is not.

## 5. It uses the machine, and gives it back

- 4 concurrent LLM requests during discovery (4 abstracts in 10s)
- Verification runs on CPU — free, unlimited, never queues behind the GPU
- The model is evicted before any run, so training gets the whole card
- `stop` kills **both** `ollama.exe` and its `llama-server.exe` child

That last one matters: killing only the parent orphans the child, which keeps
holding ~10GB with no obvious owner. `status` now labels your own leftovers as
YOURS and `start` cleans them up automatically.

---

# Closing the loop

The system now **learns from its own runs**. This is the difference between
automation and research.

```
design → verify (free) → gate → GPU → record → feeds the next design
```

After a run, the result is written to memory. The next `design` opens with:

```
What we have already measured (lower is better):
  - adaptive per-row threshold -> 3966.2

Already implemented (do NOT repeat these):
  - adaptive per-row threshold
```

And the gate refuses to spend GPU time re-running it:

```
GPU RUN BLOCKED
  BLOCKED: already run on 2026-09-03 — result was 'ok' (metric 3966.2).
           Re-running spends GPU hours to learn nothing.
```

## Generate many, verify all, run one

CPU verification is free, so generating a single candidate wastes it.

```bash
fabric/run.py design "adaptive per-row threshold" --candidates 4
```

```
[1] verified · non-zero 100% · score 2.8      <- degenerate, no zeros
[2] verified · non-zero  65% · score 7.8      <- promoted
[3] verified · non-zero 100% · score 2.7
[4] verified · non-zero   9% · score 1.2      <- too sparse
```

**Three of four were bad.** Single-shot generation would have promoted whichever
arrived first. Runners-up are kept as `method_alt*.py`.

## Running on the GPU

```bash
fabric/run.py gpu fabric/runners/quant_eval.py \
  --hypothesis "std threshold recovers within 2x of fp32"
```

`quant_eval.py` swaps your verified method into every projection layer of GPT-2
and reports perplexity against the fp32 baseline. ~3 seconds of training; it
exists to decide whether a method deserves a long run.

The result is recorded automatically and feeds the next design.

## Resuming and sharing

- `cycle --resume` continues an interrupted cycle instead of redoing discovery
- memory files are appended under a lock, so two people can run at once
- `fabric/run.py journal` shows where time goes and how much GPU was saved

```
CODE GENERATION
  attempts               8
  overall success rate   88%      (it was ~50% on llama3.1:8b)
```
