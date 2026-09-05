#!/usr/bin/env python
"""Measure a quantization method on a real model. Shipped to the box by `labpilot gpu`.

Reads method.py (the verified quantizer), swaps it into every projection layer of
a small causal LM, and reports perplexity before and after against the fp16
baseline. Deliberately small: this is the run that decides whether a method is
worth a long one.
"""
import json, math, sys, time
import torch, torch.nn as nn

MODEL = "gpt2"
SEQ, BATCH, STEPS, EVAL = 128, 4, 40, 12

def log(m):
    print(m, flush=True)
    open("logs.txt", "a", encoding="utf-8").write(str(m) + "\n")

ns = {}
exec(compile(open("method.py").read(), "method.py", "exec"), ns)
quantize = ns["quantize"]

class TernaryLinear(nn.Module):
    """STE wrapper. Presents the weight as (out, in) — the convention the
    verified method was written against."""
    def __init__(self, base, conv1d):
        super().__init__()
        w = base.weight.data
        self.conv1d = conv1d
        self.weight = nn.Parameter((w.t() if conv1d else w).clone())
        self.bias = nn.Parameter(base.bias.data.clone()) if base.bias is not None else None
        self.nf = getattr(base, "nf", self.weight.shape[0])
    def qw(self):
        w = self.weight
        wq = quantize(w).to(w.dtype)
        return w + (wq - w).detach()
    def forward(self, x):
        w = self.qw()
        out = x.view(-1, x.size(-1)) @ (w.t() if self.conv1d else w.t())
        if self.bias is not None: out = out + self.bias
        return out.view(*x.shape[:-1], self.nf if self.conv1d else w.shape[0])

@torch.no_grad()
def ppl(model, batches, dev):
    model.eval(); nll = tok = 0.0
    for b in batches:
        b = b.to(dev); o = model(b, labels=b)
        nll += o.loss.float().item() * b.numel(); tok += b.numel()
    model.train()
    return math.exp(min(nll / max(tok, 1), 20))

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
    ev, tr = batches[:EVAL], batches[EVAL:] or batches
    log(f"[data] {len(batches)} batches of {BATCH}x{SEQ}")

    m = GPT2LMHeadModel.from_pretrained(MODEL).to(dev); m.config.use_cache = False
    fp = ppl(m, ev, dev); log(f"[baseline] fp32 perplexity {fp:.3f}")

    n_rep = 0
    for blk in m.transformer.h:
        for parent in (blk.attn, blk.mlp):
            for nm, ch in list(parent.named_children()):
                if isinstance(ch, (nn.Linear, Conv1D)):
                    setattr(parent, nm, TernaryLinear(ch, isinstance(ch, Conv1D))); n_rep += 1
    m.to(dev); log(f"[quantize] {n_rep} layers")
    ptq = ppl(m, ev, dev); log(f"[ptq] before training {ptq:.3f}")

    opt = torch.optim.AdamW(m.parameters(), lr=1e-4); t0 = time.time()
    for i in range(STEPS):
        b = tr[i % len(tr)].to(dev); opt.zero_grad(set_to_none=True)
        loss = m(b, labels=b).loss
        if not torch.isfinite(loss):
            print("RESULT_JSON: " + json.dumps({"status": "diverged", "step": i})); return 1
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if i % 10 == 0: log(f"[step {i}/{STEPS}] loss {loss.item():.4f}")
    final = ppl(m, ev, dev); el = time.time() - t0
    log(f"[done] {final:.3f} vs fp32 {fp:.3f} = {final/fp:.3f}x in {el:.0f}s")
    print("RESULT_JSON: " + json.dumps({
        "status": "ok", "perplexity": final, "baseline_ppl": fp, "ptq_ppl": ptq,
        "ratio": final / fp, "recovered": bool(final < ptq),
        "layers": n_rep, "steps": STEPS, "seconds": el}))
    return 0

if __name__ == "__main__":
    try: sys.exit(main())
    except Exception as e:
        import traceback; traceback.print_exc()
        print("RESULT_JSON: " + json.dumps({"status": "error", "reason": f"{type(e).__name__}: {e}"}))
        sys.exit(2)
