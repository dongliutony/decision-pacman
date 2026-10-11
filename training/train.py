"""LoRA fine-tune a small decision model on oracle-labeled Pac-Man states.

  CUDA_VISIBLE_DEVICES=1 python training/train.py --data data/sft --out runs/ft_v1

The loss reads the model exactly the way Ollama's /v1/systemone scorer does:
take the logits of the option letters (A, B, C, ...) at the first answer
position after the chat-templated prompt, softmax over those letters only,
and compare with the oracle's soft target (cross-entropy). Only that one
position is projected through the output layer, which keeps memory small
despite the large vocabulary.
"""
import argparse
import json
import math
import os
import random
import time

os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

from unsloth import FastLanguageModel  # noqa: E402,I001
import torch  # noqa: E402

# Every language-side linear layer of Qwen3.5: attention, Gated DeltaNet, MLP.
TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]


def read_jsonl(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class Batcher:
    def __init__(self, tokenizer, letter_ids, max_len):
        self.tok, self.letter_ids, self.max_len = tokenizer, letter_ids, max_len

    def prompt(self, row):
        return self.tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def __call__(self, rows, device):
        texts = [self.prompt(r) for r in rows]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, padding_side="right")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if ids.shape[1] > self.max_len:
            raise ValueError(f"prompt of {ids.shape[1]} tokens exceeds --max-len {self.max_len}")
        last = mask.sum(dim=1) - 1
        k = max(len(r["keys"]) for r in rows)
        cand = torch.zeros(len(rows), k, dtype=torch.long)
        valid = torch.zeros(len(rows), k, dtype=torch.bool)
        target = torch.zeros(len(rows), k)
        for i, r in enumerate(rows):
            n = len(r["keys"])
            cand[i, :n] = torch.tensor(self.letter_ids[:n])
            valid[i, :n] = True
            target[i, :n] = torch.tensor(r["target"])
        return ids.to(device), mask.to(device), last.to(device), cand.to(device), valid.to(device), target.to(device)


def option_logits(model, lm_weight, ids, mask, last, cand):
    """Letter logits at each row's answer position: final hidden state times the letters' output rows."""
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, logits_to_keep=1)
    hidden = out.hidden_states[-1]
    h = hidden[torch.arange(ids.shape[0], device=ids.device), last]
    w = lm_weight[cand]  # [B, K, H]
    return torch.einsum("bh,bkh->bk", h.float(), w.float()), out


def masked_log_softmax(logits, valid):
    return torch.log_softmax(logits.masked_fill(~valid, -1e9), dim=-1)


@torch.no_grad()
def evaluate(model, lm_weight, batcher, rows, device, bs):
    model.eval()
    n = correct = decisive = decisive_ok = 0
    ce_sum = 0.0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, mask, last, cand, valid, target = batcher(chunk, device)
        logits, _ = option_logits(model, lm_weight, ids, mask, last, cand)
        logp = masked_log_softmax(logits, valid)
        ce_sum += float(-(target * logp.masked_fill(~valid, 0)).sum())
        pred = logp.argmax(dim=-1)
        best = target.argmax(dim=-1)
        for j in range(len(chunk)):
            n += 1
            correct += int(pred[j] == best[j])
            if float(target[j].max()) > 0.9:
                decisive += 1
                decisive_ok += int(pred[j] == best[j])
    model.train()
    return {"ce": ce_sum / n, "acc": correct / n, "acc_decisive": decisive_ok / max(1, decisive), "n": n, "n_decisive": decisive}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--data", default="data/sft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20, help="print loss, elapsed time, and the time left every N steps")
    ap.add_argument("--val-limit", type=int, default=2000)
    ap.add_argument("--train-limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    device = "cuda"

    model, tok = FastLanguageModel.from_pretrained(model_name=args.base, max_seq_length=args.max_len, load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
    tok = getattr(tok, "tokenizer", tok)  # multimodal processors wrap the text tokenizer
    letter_ids = []
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        ids = tok.encode(ch, add_special_tokens=False)
        assert len(ids) == 1, f"letter {ch} is not a single token: {ids}"
        letter_ids.append(ids[0])

    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [t for t in TARGETS if t in present]
    print("LoRA targets:", targets, flush=True)
    model = FastLanguageModel.get_peft_model(model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0, target_modules=targets,
                                             bias="none", use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=args.max_len)
    lm_weight = model.get_output_embeddings().weight

    train = read_jsonl(os.path.join(args.data, "train.jsonl"), args.train_limit)
    val = read_jsonl(os.path.join(args.data, "val.jsonl"), args.val_limit)
    batcher = Batcher(tok, letter_ids, args.max_len)
    print("example prompt:\n" + batcher.prompt(train[0]), flush=True)

    # Sanity check: the one-position projection must match the model's own last-position logits.
    ids, mask, last, cand, valid, target = batcher(train[:4], device)
    with torch.no_grad():
        logits, out = option_logits(model, lm_weight, ids, mask, last, cand)
        full_last = out.logits[:, -1, :].float()
        longest = int(mask.sum(dim=1).argmax())
        ref = full_last[longest, cand[longest]]
        diff = float((logits[longest] - ref).abs().max())
    print(f"projection check: max |diff| = {diff:.4f}", flush=True)
    assert diff < 0.5, "hidden-state projection does not match the model's logits"

    log_path = os.path.join(args.out, "log.jsonl")
    base_metrics = evaluate(model, lm_weight, batcher, val, device, args.batch)
    print("before training:", base_metrics, flush=True)
    with open(log_path, "w") as f:
        f.write(json.dumps({"step": 0, **base_metrics}) + "\n")

    steps = math.ceil(len(train) * args.epochs / args.batch)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    order = list(range(len(train)))
    rng = random.Random(0)
    model.train()
    t0 = time.time()
    pos = len(order)
    for step in range(1, steps + 1):
        if pos + args.batch > len(order):
            rng.shuffle(order)
            pos = 0
        rows = [train[i] for i in order[pos:pos + args.batch]]
        pos += args.batch
        ids, mask, last, cand, valid, target = batcher(rows, device)
        logits, _ = option_logits(model, lm_weight, ids, mask, last, cand)
        logp = masked_log_softmax(logits, valid)
        loss = -(target * logp.masked_fill(~valid, 0)).sum(dim=-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == steps:
            elapsed = time.time() - t0
            left = elapsed / step * (steps - step)
            print(f"step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {elapsed:.0f}s "
                  f"({elapsed / step:.1f}s/step, about {left / 60:.0f} min left)", flush=True)
        if step % args.eval_every == 0 or step == steps:
            m = evaluate(model, lm_weight, batcher, val, device, args.batch)
            print(f"eval step {step}: {m}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss.item(), **m}) + "\n")

    adapter = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter)
    tok.save_pretrained(adapter)
    print("saved adapter ->", adapter, flush=True)


if __name__ == "__main__":
    main()
