#!/usr/bin/env python
"""Measure what a quantization method does to CAPABILITY, not just to perplexity.

This project has already paid for the lesson once: a 15.7% perplexity improvement
moved downstream accuracy by 0.0012, and a separate run's perplexity fell 99.96%
while MMLU stayed at chance. Ranking methods on perplexity optimises a number
nobody cares about.

So this reports **chance-corrected retention** on real tasks:

    retention = (quantized_acc - chance) / (fp_acc - chance)

Perplexity is still reported, but only as an explanatory number. Retention
decides.
"""
import json, math, sys, time
import torch, torch.nn as nn

MODEL = "gpt2"
SEQ, BATCH, STEPS, EVAL_B = 128, 4, 60, 12
TASKS = ["arc_easy", "piqa"]     # cheap, and both have a known chance floor
TASK_LIMIT = 150                 # per task; enough to rank, not to publish
CHANCE = {"arc_easy": 0.25, "piqa": 0.50, "hellaswag": 0.25,
          "winogrande": 0.50, "arc_challenge": 0.25}

def log(m):
    print(m, flush=True)
    open("logs.txt", "a", encoding="utf-8").write(str(m) + "\n")

ns = {}
exec(compile(open("method.py").read(), "method.py", "exec"), ns)
quantize = ns["quantize"]


class TernaryLinear(nn.Module):
    def __init__(self, base, conv1d):
        super().__init__()
        w = base.weight.data
        self.conv1d = conv1d
        self.weight = nn.Parameter((w.t() if conv1d else w).clone())
        self.bias = nn.Parameter(base.bias.data.clone()) if base.bias is not None else None
        self.nf = getattr(base, "nf", self.weight.shape[0])

    def forward(self, x):
        w = self.weight
        wq = quantize(w).to(w.dtype)
        w = w + (wq - w).detach()                 # straight-through estimator
        out = x.view(-1, x.size(-1)) @ w.t()
        if self.bias is not None:
            out = out + self.bias
        return out.view(*x.shape[:-1], w.shape[0])


@torch.no_grad()
def ppl(model, batches, dev):
    model.eval(); nll = tok = 0.0
    for b in batches:
        b = b.to(dev); o = model(b, labels=b)
        nll += o.loss.float().item() * b.numel(); tok += b.numel()
    model.train()
    return math.exp(min(nll / max(tok, 1), 20))


def bench(model, tokenizer, tag):
    """Downstream accuracy through lm-eval-harness — the numbers papers report."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except Exception as e:
        log(f"[eval] lm_eval unavailable: {e}")
        return {}
    was = model.training; model.eval()
    try:
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=16)
        res = simple_evaluate(model=lm, tasks=TASKS, limit=TASK_LIMIT)
    except Exception as e:
        log(f"[eval] {tag} failed: {type(e).__name__}: {e}")
        if was: model.train()
        return {}
    out = {}
    for t, m in (res.get("results") or {}).items():
        a = m.get("acc_norm,none", m.get("acc,none"))
        if a is not None:
            out[t] = float(a)
    log(f"[eval:{tag}] " + " ".join(f"{k}={v:.3f}" for k, v in out.items()))
    if was: model.train()
    return out


def retention(fp: dict, q: dict) -> tuple:
    """Chance-corrected, per task and averaged. A model at chance scores 0."""
    per, vals = {}, []
    for t in fp:
        if t not in q:
            continue
        c = CHANCE.get(t, 0.25)
        denom = fp[t] - c
        r = (q[t] - c) / denom if denom > 0.02 else float("nan")
        per[t] = round(r, 4)
        if not math.isnan(r):
            vals.append(r)
    return (sum(vals) / len(vals) if vals else float("nan")), per


def main():
    if not torch.cuda.is_available():
        print("RESULT_JSON: " + json.dumps({"status": "error", "reason": "no CUDA"})); return 3
    dev = torch.device("cuda"); torch.manual_seed(0)
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformers.pytorch_utils import Conv1D

    tk = AutoTokenizer.from_pretrained(MODEL)
    ids = tk(open("corpus.txt", encoding="utf-8").read()[:900_000])["input_ids"]
    per = SEQ * BATCH; n = (len(ids) // per) * per
    batches = list(torch.tensor(ids[:n]).view(-1, BATCH, SEQ))
    ev, tr = batches[:EVAL_B], batches[EVAL_B:] or batches

    m = GPT2LMHeadModel.from_pretrained(MODEL).to(dev); m.config.use_cache = False
    fp_ppl = ppl(m, ev, dev); log(f"[baseline] fp32 perplexity {fp_ppl:.3f}")
    fp_acc = bench(m, tk, "fp32")

    n_rep = 0
    for blk in m.transformer.h:
        for parent in (blk.attn, blk.mlp):
            for nm, ch in list(parent.named_children()):
                if isinstance(ch, (nn.Linear, Conv1D)):
                    setattr(parent, nm, TernaryLinear(ch, isinstance(ch, Conv1D))); n_rep += 1
    m.to(dev); log(f"[quantize] {n_rep} layers")
    ptq_ppl = ppl(m, ev, dev)

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4); t0 = time.time()
    for i in range(STEPS):
        b = tr[i % len(tr)].to(dev); opt.zero_grad(set_to_none=True)
        loss = m(b, labels=b).loss
        if not torch.isfinite(loss):
            print("RESULT_JSON: " + json.dumps({"status": "diverged", "step": i})); return 1
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if i % 20 == 0: log(f"[step {i}/{STEPS}] loss {loss.item():.4f}")
    train_s = time.time() - t0

    q_ppl = ppl(m, ev, dev)
    q_acc = bench(m, tk, "ternary")
    ret, per_task = retention(fp_acc, q_acc)

    log(f"[result] retention {ret:.3f} · perplexity {q_ppl:.1f} "
        f"({q_ppl/fp_ppl:.2f}x fp32)")
    print("RESULT_JSON: " + json.dumps({
        "status": "ok",
        "retention": None if math.isnan(ret) else round(ret, 4),   # DECIDES
        "retention_per_task": per_task,
        "fp_acc": fp_acc, "quantized_acc": q_acc,
        "perplexity": q_ppl, "baseline_ppl": fp_ppl,               # explains only
        "ppl_ratio": q_ppl / fp_ppl, "ptq_ppl": ptq_ppl,
        "layers": n_rep, "steps": STEPS, "seconds": round(train_s, 1),
        "metric_note": "retention decides; perplexity is explanatory only"}))
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e:
        import traceback; traceback.print_exc()
        print("RESULT_JSON: " + json.dumps({"status": "error", "reason": f"{type(e).__name__}: {e}"}))
        sys.exit(2)
