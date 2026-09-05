#!/usr/bin/env python
"""labpilot — on-demand research labpilot. Nothing runs unless you start it.

  labpilot status          is the box free? is anything of mine running?
  labpilot start           open a session (refuses if a teammate is using the GPU)
  labpilot discover "..."  stage 1: parallel literature triage
  labpilot design "..."    stage 2: write a method, verified on CPU before any GPU
  labpilot gpu <script>    stage 4: evict the model, then run
  labpilot stop            close the session, release the card
  labpilot cycle "..."     all stages end to end

The shape is a funnel, not a swarm: wide and cheap at the top, one job at the
bottom. Stages never overlap, so VRAM is never contended.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labpilot import checkpoint, design, gate, journal, llm, session, store  # noqa: E402
from labpilot import config as fabcfg                 # noqa: E402
from labpilot.remote import RemoteBox, RemoteError     # noqa: E402

WORK = ROOT / "labpilot" / "work"


def box():
    return RemoteBox()


class Session:
    """Opens a session and GUARANTEES it closes.

    Without this, a Ctrl+C or a crash leaves the model holding ~9.6GB of a
    shared card until someone notices. Cleanup runs on normal exit, on
    exception, and on Ctrl+C / SIGTERM.
    """

    def __init__(self, force=False, keep_open=False):
        self.force, self.keep_open, self.opened = force, keep_open, False

    def __enter__(self):
        import atexit, signal
        with box() as b:
            if not session.start(b, force=self.force):
                sys.exit(1)
        self.opened = True
        atexit.register(self.close)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass
        return self

    def _on_signal(self, *_):
        print("\ninterrupted — releasing the GPU before exit")
        self.close()
        sys.exit(130)

    def close(self):
        if not self.opened or self.keep_open:
            return
        self.opened = False
        try:
            with box() as b:
                session.stop(b)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not stop cleanly ({exc}).")
            print("Run `python labpilot/run.py stop` to release the card.")

    def __exit__(self, *exc):
        self.close()
        return False


def _open() -> RemoteBox:
    b = RemoteBox().connect()
    return b


# ------------------------------------------------------------------ commands
def cmd_status(a):
    with box() as b:
        st = session.probe(b)
    print(f"\nGPU        {st.used_mb} MB used of {st.total_mb} MB "
          f"({st.free_mb} MB free)")
    print(f"ollama     {st.ollama_procs} process(es)")
    theirs = [p for p in st.compute_procs if not p.startswith("OURS:")]
    print(f"compute    {', '.join(theirs) if theirs else 'none (not yours)'}")
    if st.our_orphans:
        print(f"YOURS      {', '.join(st.our_orphans)} still holding the card")
    ck = checkpoint.describe()
    if "no cycle" not in ck:
        print(f"cycle      {ck}")
    if st.ollama_procs:
        print("\nA session of yours is OPEN. `labpilot stop` to release the card.")
    elif st.someone_else_working:
        print(f"\nA compute job is running that is not yours. Do not start.")
    elif st.our_orphans:
        print(f"\nYour own leftovers are holding {st.used_mb}MB. "
              f"Run `labpilot/run.py stop` to release.")
    elif st.desktop_heavy:
        print(f"\nNo job running, but the logged-in desktop holds {st.used_mb}MB.")
        print(f"Only {st.free_mb}MB free — close browser windows on the box, "
              f"or log the console session out.")
    else:
        print("\nBox is idle. Safe to `labpilot start`.")


def cmd_start(a):
    with box() as b:
        ok = session.start(b, force=a.force)
    sys.exit(0 if ok else 1)


def cmd_stop(a):
    with box() as b:
        session.stop(b)


def cmd_cycle(a):
    """One full pass: discover, then design. Always releases the GPU at the end.

    Resumable: discovery costs minutes of LLM time, so a dropped connection
    should not mean starting over.
    """
    if a.resume and checkpoint.load():
        print(checkpoint.describe())
    elif not a.resume:
        checkpoint.clear()

    with Session(force=a.force, keep_open=a.keep_open):
        if a.resume and checkpoint.done("discover"):
            print("discover already done in this cycle — skipping (--no-resume to redo)")
        else:
            cmd_discover(a)
            checkpoint.save("discover", query=a.query)
        if a.idea:
            a.paper = getattr(a, "paper", None); a.novel = getattr(a, "novel", False)
            a.hypothesis = getattr(a, "hypothesis", "")
            print("\n" + "=" * 60)
            cmd_design(a)
    checkpoint.save("complete")
    print("\ncycle complete — GPU released.")


def cmd_discover(a):
    """Stage 1 — targeted at the profile's OPEN QUESTIONS, and never repeated."""
    from paperfit.constraints import Profile
    from paperfit.hunt import hunt

    prof = Profile.load(a.profile)
    print(f"\nprofile: {prof.name}")

    # Queries come from the open questions, not from generic field terms.
    # "ternary" alone returns materials science; a question is specific.
    questions = prof.g("open_questions", default=[]) or []
    q_for = {}
    queries = list(a.query or [])
    if questions and not a.query:
        for q in questions:
            if a.question and q.get("id") != a.question:
                continue
            for term in q.get("queries", []):
                y = store.query_yield(term)
                if y is not None and y < 0.05 and not a.repeat:
                    print(f"   skipping '{term}' — last sweep found nothing new")
                    continue
                queries.append(term)
                q_for[term] = q["id"]
        print(f"targeting {len({q_for[t] for t in q_for})} open question(s), "
              f"{len(queries)} queries")

    cands, hits, rejected, survivors = hunt(
        prof, queries or None, a.limit, a.since,
        ("openalex", "arxiv", "europepmc"), None, a.min_topical)
    for term in queries:
        store.record_query(term, len(cands), len(survivors))

    # Never re-read what we already judged.
    fresh = [h for h in survivors if not store.seen_paper(h.candidate.ident)]
    skipped = len(survivors) - len(fresh)
    print(f"\n{len(rejected)} blocked by constraints · {len(survivors)} relevant"
          + (f" · {skipped} already read (skipped)" if skipped else ""))

    # Rank by the QUESTION we are asking, not just by global interest match.
    # `interests.strong` describes the project as a whole, so on a targeted
    # sweep it promotes whatever is most on-brand rather than whatever actually
    # answers the question -- ask a narrow question outside the project's usual
    # vocabulary and every paper read is on-topic for the project and useless
    # for the question. Papers retrieved for THIS question's queries, and papers
    # whose text echoes the question, come first.
    if a.question:
        qobj = next((q for q in questions if q.get("id") == a.question), None)
        if qobj:
            import re as _re
            STOP = {"the","a","an","of","for","and","or","to","in","on","at","is",
                    "are","do","does","did","what","which","how","why","we","our",
                    "that","this","it","its","be","can","with","from","by","as"}
            def _words(t):
                return {w for w in _re.findall(r"[a-z0-9-]+", (t or "").lower())
                        if len(w) > 2 and w not in STOP}
            qterms = _words(qobj.get("question", ""))
            for term in qobj.get("queries", []):
                qterms |= _words(term)

            def _affinity(h):
                c = h.candidate
                txt = _words(f"{c.title} {c.abstract}")
                if not qterms:
                    return 0.0
                return len(txt & qterms) / len(qterms)

            fresh = sorted(fresh, key=lambda h: (_affinity(h), h.score), reverse=True)
            if fresh:
                print(f"   re-ranked {len(fresh)} by affinity to '{a.question}' "
                      f"(top {_affinity(fresh[0]):.0%} term overlap)")

    top = fresh[:a.top]
    if not top:
        print("\nnothing NEW survived. Either the ground is covered, or "
              "loosen --min-topical / add --query terms.")
        return

    # Now the parallel part: read all the survivors' abstracts at once.
    with box() as _b:
        if not llm.server_up(_b):
            sys.exit("No model server on the box. Run `labpilot/run.py start` first.")
    print(f"reading {len(top)} abstracts in parallel on the box...")
    qlist = "\n".join(f"- {q['id']}: {q['question']}" for q in questions) or "(none stated)"
    prompts = [
        f"Our open research questions:\n{qlist}\n\n"
        f"Read the abstract below. Answer in this exact form:\n"
        f"ANSWERS: <question id, or NONE>\n"
        f"METHOD: <what THIS paper's own method does, one sentence>\n"
        f"COST: <hardware / training budget it states, or NOT STATED>\n"
        f"Do not speculate. If it answers none of our questions, say NONE.\n\n"
        f"TITLE: {h.candidate.title}\n\nABSTRACT: {h.candidate.abstract[:2500]}"
        for h in top]
    t0 = time.time()
    notes = llm.ask_many(box, prompts, workers=a.workers)
    print(f"   {len(top)} read in {time.time()-t0:.0f}s\n")

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "discover.md"
    kept = 0
    with out.open("a", encoding="utf-8") as f:
        f.write(f"\n# Sweep {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        for h, note in zip(top, notes):
            c = h.candidate
            ans = ""
            m = [l for l in note.splitlines() if l.upper().startswith("ANSWERS:")]
            if m:
                ans = m[0].split(":", 1)[1].strip()
            on_target = ans and ans.upper() not in ("NONE", "NONE.")
            store.record_paper(c.ident, c.title,
                               "on-target" if on_target else "off-target",
                               note, ans)
            flag = f"-> {ans}" if on_target else "(answers none of our questions)"
            print(f"[{h.score:5.1f}] {c.ident:20} {c.title[:48]}  {flag}")
            if on_target:
                kept += 1
                f.write(f"## {c.title}\n`{c.ident}` · {c.year} · answers **{ans}**\n\n"
                        f"{note}\n\n---\n\n")
    print(f"\n{kept} of {len(top)} actually answer one of our questions")
    print(f"notes -> {out.relative_to(ROOT)}   (memory: {store.summary()})")


def cmd_design(a):
    """Generate N candidates in parallel, verify all on CPU, promote the best.

    CPU verification is free, so generating one candidate through it wastes it.
    History from past runs is fed back in, so the model does not re-propose
    something we already measured.
    """
    CONTRACT = (
        "Write ONE Python function, exactly:\n"
        "    def quantize(w: torch.Tensor) -> torch.Tensor\n"
        "w is a 2D float tensor (out_features, in_features). Return the SAME shape.\n"
        "Every row must contain at most THREE distinct values: -s, 0, +s.\n"
        "Import ONLY torch, math, torch.nn.functional as F.\n"
        "Process the whole tensor at once — never loop over rows.\n"
        "Guard divisions with .clamp_min(1e-8). Keep 40-80 percent non-zero.\n"
        "Output ONLY a python code block.\n")

    history = design.context_from_history()
    if history:
        print("using what we already measured:")
        for l in history.splitlines()[:5]:
            print("  " + l)
        print()

    paper_text = None
    if a.paper:
        pt = Path(a.paper)
        paper_text = pt.read_text(encoding="utf-8", errors="replace") if pt.exists() else a.paper

    base = (f"{history}\n\n" if history else "") + \
           f"Implement this quantization idea.\n\nIDEA: {a.idea}\n\n{CONTRACT}"
    variants = [
        base,
        base + "\nUse a threshold derived from the standard deviation of each row.",
        base + "\nUse a threshold derived from a quantile of the absolute values.",
        base + "\nUse the root-mean-square of the surviving weights as the scale.",
    ][:a.candidates]

    print(f"generating {len(variants)} candidates in parallel on the box...")
    with journal.stage("design", candidates=len(variants)):
        with box() as b:
            if not llm.server_up(b):
                sys.exit("No model server. Run `labpilot/run.py start` first.")
        replies = llm.ask_many(box, variants, workers=min(a.workers, len(variants)),
                               temperature=0.3, log=lambda *_: None)
        print("verifying all of them on CPU (free):")
        ranked = design.evaluate_batch(replies, paper_text, a.novel)

    if not ranked:
        print("\nnone verified. The idea may be unimplementable as stated.")
        sys.exit(1)

    best = ranked[0]
    dec = gate.evaluate(best.code, paper_text, novel=a.novel, hypothesis=a.hypothesis)
    journal.write("gate.approved" if dec.allowed else "gate.blocked",
                  fingerprint=best.fingerprint)
    print("\n" + dec.report())

    store.record_method(a.idea[:60], best.code, a.paper or "novel",
                        best.provenance)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "method.py").write_text(best.code, encoding="utf-8")
    for i, c in enumerate(ranked[1:4], 2):
        (WORK / f"method_alt{i}.py").write_text(c.code, encoding="utf-8")

    print(f"\n{best.code}")
    print(f"-> labpilot/work/method.py   fingerprint {best.fingerprint}")
    if len(ranked) > 1:
        print(f"   {len(ranked)-1} runner(s)-up saved as method_alt*.py")
    if not dec.allowed:
        print("\nThis will NOT reach the GPU until the blockers above are fixed.")


def cmd_gpu(a):
    """Stage 4 — evict the model, then run. The card is entirely the job's."""
    script = Path(a.script)
    if not script.exists():
        sys.exit(f"no such script: {script}")

    method = WORK / "method.py"
    if method.exists() and not a.skip_gate:
        paper_text = None
        if a.paper and Path(a.paper).exists():
            paper_text = Path(a.paper).read_text(encoding="utf-8", errors="replace")
        dec = gate.evaluate(method.read_text(encoding="utf-8"), paper_text,
                            novel=a.novel, hypothesis=a.hypothesis, force=a.force)
        print(dec.report())
        if not dec.allowed:
            sys.exit("\nnot spending GPU hours on this. Fix the above, or --force.")
        print()
    with box() as b:
        free = session.evict(b, llm.MODEL)
        if free < a.need_mb:
            sys.exit(f"only {free}MB free, run needs {a.need_mb}MB. Aborting.")
        remote_dir = fabcfg.WORKDIR
        b.mkdir(remote_dir)
        b.upload(str(script), f"{remote_dir}/{script.name}")
        # the runner reads the verified method and the shared corpus
        if method.exists():
            b.upload(str(method), f"{remote_dir}/method.py")
        from labpilot.corpus import ensure_corpus
        b.upload(ensure_corpus(), f"{remote_dir}/corpus.txt")
        print(f"running {script.name} on the box (timeout {a.timeout}s)...")
        with journal.stage("gpu", script=script.name):
            res = b.run_python(remote_dir, script.name, timeout=a.timeout)
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "gpu_run.log").write_text(res.combined, encoding="utf-8")
        print(res.combined[-1200:])

        # record the result so the next design cycle can learn from it
        import json as _j, re as _re
        m = None
        for line in res.combined.splitlines():
            i = line.find("RESULT_JSON:")
            if i >= 0:
                try: m = _j.loads(line[i+12:].strip())
                except _j.JSONDecodeError: pass
        if m and method.exists():
            fp = store.method_fingerprint(method.read_text(encoding="utf-8"))
            # Retention decides when it is available; perplexity only explains.
            # Ranking on perplexity is the mistake this project already paid for.
            ret = m.get("retention")
            metric = (1 - ret) if ret is not None else m.get("perplexity")
            store.record_run(fp, script.name, m.get("status", "?"),
                             metric=metric,
                             cost_minutes=round(m.get("seconds", 0)/60, 1),
                             metric_name="1-retention" if ret is not None
                             else "perplexity")
            if ret is not None:
                print(f"\nrecorded: retention {ret:.3f} "
                      f"(perplexity {m.get('perplexity',0):.1f}, "
                      f"{m.get('ppl_ratio',0):.2f}x — explanatory only)")
                for t, v in (m.get("retention_per_task") or {}).items():
                    print(f"          {t}: {v}")
            else:
                print(f"\nrecorded: {m.get('status')} · ppl {m.get('perplexity')}")
            print("this now feeds into the next `design` automatically.")
        print(f"full log -> labpilot/work/gpu_run.log")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="is the box free?").set_defaults(fn=cmd_status)
    sub.add_parser("config", help="show where this labpilot points").set_defaults(
        fn=lambda _a: print("\n" + fabcfg.describe()))

    s = sub.add_parser("start", help="open a session")
    s.add_argument("--force", action="store_true",
                   help="start even if the GPU looks busy")
    s.set_defaults(fn=cmd_start)

    sub.add_parser("stop", help="close the session, release the GPU").set_defaults(fn=cmd_stop)

    d = sub.add_parser("discover", help="stage 1: parallel literature triage")
    d.add_argument("query", nargs="*", default=None)
    d.add_argument("--profile", default=fabcfg.DEFAULT_PROFILE)
    d.add_argument("--limit", type=int, default=12)
    d.add_argument("--since", type=int, default=2023)
    d.add_argument("--top", type=int, default=6)
    d.add_argument("--min-topical", type=float, default=2.0)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--question", help="target ONE open question by id")
    d.add_argument("--repeat", action="store_true",
                   help="re-run queries that previously found nothing new")
    d.set_defaults(fn=cmd_discover)

    g = sub.add_parser("design", help="stage 2+3: write a method, verify on CPU")
    g.add_argument("idea")
    g.add_argument("--candidates", type=int, default=4,
                   help="how many to generate; all are verified for free")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--paper", help="path to the source paper's text, or the text itself")
    g.add_argument("--novel", action="store_true",
                   help="declare this an original method with no source to match")
    g.add_argument("--hypothesis", default="",
                   help="what you expect, and what would refute it")
    g.set_defaults(fn=cmd_design)

    r = sub.add_parser("gpu", help="stage 4: evict the model, then run a script")
    r.add_argument("script")
    r.add_argument("--timeout", type=int, default=3600)
    r.add_argument("--need-mb", type=int, default=6000)
    r.add_argument("--paper"); r.add_argument("--novel", action="store_true")
    r.add_argument("--hypothesis", default="")
    r.add_argument("--force", action="store_true")
    r.add_argument("--skip-gate", action="store_true")
    r.set_defaults(fn=cmd_gpu)

    c = sub.add_parser("cycle", help="discover + design in one go, auto-releases the GPU")
    c.add_argument("query", nargs="*", default=None)
    c.add_argument("--idea", default=None, help="also design a method from this idea")
    c.add_argument("--profile", default=fabcfg.DEFAULT_PROFILE)
    c.add_argument("--limit", type=int, default=12)
    c.add_argument("--since", type=int, default=2023)
    c.add_argument("--top", type=int, default=6)
    c.add_argument("--min-topical", type=float, default=2.0)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--candidates", type=int, default=4)
    c.add_argument("--question", default=None)
    c.add_argument("--repeat", action="store_true")
    c.add_argument("--force", action="store_true")
    c.add_argument("--resume", action="store_true",
                   help="continue an interrupted cycle instead of restarting")
    c.add_argument("--keep-open", action="store_true",
                   help="leave the session open afterwards (you must stop it)")
    c.set_defaults(fn=cmd_cycle)

    j = sub.add_parser("journal", help="what has this cost, and where does time go")
    j.set_defaults(fn=lambda _a: print(journal.report()))

    a = p.parse_args()
    try:
        a.fn(a)
    except RemoteError as exc:
        sys.exit(f"box unreachable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
