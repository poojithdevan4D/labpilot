# labpilot

**Automates the boring half of ML research, and refuses to waste your GPU.**

Find the papers that matter. Turn one into working code. Check the code against
the paper. Run it. Remember the result so you never do it twice.

You have one GPU, maybe two, shared with colleagues. Every hour spent on a run
that was never going to teach you anything is an hour you don't get back — so
four **free** gates sit in front of every run:

```
GPU RUN BLOCKED
  BLOCKED: already run on 2026-09-03 — result was 'ok' (retention 0.21).
           Re-running spends GPU hours to learn nothing.
```

| gate | blocks | costs |
|---|---|---|
| **verification** | code that doesn't run, or doesn't do what it claims | milliseconds, CPU |
| **memory** | a method whose result you already have | nothing |
| **provenance** | code that doesn't match the paper it came from | nothing |
| **hypothesis** | a run whose outcome would change no decision | nothing |

All four run before a single GPU cycle is spent.

## The loop

```
discover  →  design  →  verify  →  gate  →  GPU  →  record
   ↑                                                   │
   └─────────────── feeds the next design ─────────────┘
```

Results are written to memory, and the next design opens with them:

```
What we have already measured (lower is better):
  - adaptive per-row threshold -> 0.213 retention

Already implemented (do NOT repeat these):
  - adaptive per-row threshold
```

That feedback edge is the difference between automation and research. Without it
the system re-proposes dead ends forever.

## Four ideas worth stealing even if you don't use this

**1. Verification is free, so generate many.**

```
[1] verified · non-zero 100% · score 2.8      <- degenerate, no zeros
[2] verified · non-zero  65% · score 7.8      <- promoted
[3] verified · non-zero 100% · score 2.7
[4] verified · non-zero   9% · score 1.2      <- too sparse
```

Three of four were bad. Single-shot generation promotes whichever arrives first.

**2. Generated code must match its source paper.**

| case | verdict |
|---|---|
| faithful implementation | MATCHED → allowed |
| paper says 0.7, code uses 0.35 | PARTIAL → blocked |
| paper says per-row, code does per-tensor | MISMATCH → blocked |
| your own original idea (`--novel`) | NOVEL → allowed |

Code that silently diverges from its source produces a number attributed to the
wrong method. That's worse than code that crashes — a crash you notice.

**3. Capability decides; loss explains.**

```
retention: -0.213      <- below chance. The model is destroyed.
  arc_easy: +0.07
  piqa:     -0.50
perplexity: 2493 (44x fp32)    <- explanatory only
```

Perplexity said "44x worse". Retention said *destroyed on physical reasoning
specifically*. Ranking on an unvalidated proxy is how months get spent improving
a number nobody cares about.

**4. Bulk-parallel within a stage; sequential between stages.**

Most agentic setups run agents concurrently and call that the innovation. But
code generation depends on the literature findings, and training depends on the
code. Running them in parallel means agents working on stale inputs.

| stage | VRAM | parallel? |
|---|---:|---|
| discover — search, read abstracts | model | **yes** |
| design — generate N candidates | model | **yes** |
| verify — compile, run, check invariants | **0** | free, unlimited |
| run — training / eval | **all of it** | one job |

Stages never overlap, so VRAM contention is structurally impossible — not an
estimate.

## Sharing a GPU

Nothing runs unless you start it. No daemon, no logon task.

```bash
python labpilot/run.py status     # is the box free?
python labpilot/run.py start      # refuses if a teammate has a job running
python labpilot/run.py stop       # release the card
```

`start` classifies by **process**, not by bytes: a logged-in Windows desktop can
hold 10GB without anyone working, and refusing on that makes the tool unusable.

`stop` kills both `ollama.exe` and its `llama-server.exe` child. Killing only the
parent orphans the child, which keeps ~10GB reserved with no obvious owner —
and then looks like somebody else's job to the next check.

Ctrl+C releases the GPU too.

## What a session looks like

```bash
labpilot status                    # is the box free?
labpilot cycle "ternary quantization calibration"
```

That one command searches OpenAlex, arXiv and Europe PMC, filters ~400 papers
down to the handful that answer *your* open questions, reads their abstracts
four at a time on the box, writes a method, verifies it on CPU, and releases
the GPU. About three minutes, no GPU time spent.

Then when you want a number:

```bash
labpilot design "<your idea>" --candidates 4 --hypothesis "<what you expect>"
labpilot gpu labpilot/runners/capability_eval.py
```

## Setup

You need a laptop, a GPU box reachable over SSH, and
[Ollama](https://ollama.com) on the box.

```bash
pip install -e .
cp labpilot.toml.example labpilot.toml    # edit: host, user, workdir
python labpilot/run.py config            # check it points where you think
```

On the box:
```bash
ollama pull qwen2.5-coder:14b
```

A coder model beats a general one substantially. Measured on the same task:
`llama3.1:8b` failed ~50% of generations and invented functions that don't
exist; `qwen2.5-coder:14b` passes ~88%.

## Use

```bash
python labpilot/run.py cycle "your topic"          # discover + design, auto-releases
python labpilot/run.py design "<idea>" --candidates 4 --hypothesis "..."
python labpilot/run.py gpu labpilot/runners/capability_eval.py
python labpilot/run.py journal                     # where did the time go?
```

## Adapting it

- **`labpilot/runners/*.py`** — what an experiment measures. Two ship:
  `quant_eval` (perplexity, fast) and `capability_eval` (downstream retention).
  Write your own; it only needs to print `RESULT_JSON: {...}`.
- **`profiles/*.yaml`** — hardware, budget, licence policy, and the open
  questions searches are built from. Uses the same format as
  [paperfit](https://github.com/poojithdevan4D/paperfit).
- **`labpilot.toml`** — where your box is.

## Honest limits

- **The verifier assumes a `quantize(w)` function.** It works for any
  quantization work today; another ML subfield needs to swap that one function.
  That's the main change required, and it's in `labpilot/verify.py`.
- No multi-GPU scheduling. One box, one job.
- The provenance checker is a heuristic over operations and constants. It
  catches real divergence; it is not a proof of equivalence.
- **Nobody outside its authors has used it yet.** Treat the numbers above as
  measured-by-us, not independently validated.

Apache-2.0.
