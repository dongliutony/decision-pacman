# Plan: how good can a decision model get on one task?

Status: approved. Bases: Qwen3.5-0.8B, Qwen3.5-4B, Qwen3.8-27B (bf16). Servers: 8x H200 and 8x RTX Pro 6000, Node 20+, vLLM for inference. Track A's first batch of code is in place (see the list at the end); A0 is the next step, run with [guides/distill-oracle-gpu.md](guides/distill-oracle-gpu.md).

Two tracks, run separately so their answers stay separate:

- **Track A, the ceiling.** The exact rollout search `oracle-pellets-5s` labels the data. It sees 5 s of the future and plays at 1,171 pellets per game. Students read only the current state. How close can a student get as model size, data, and input grow?
- **Track B, no simulator.** An LLM teacher (DeepSeek, hosted on our own GPUs) labels the same states from the current state alone, with no rollouts. How good a student can a teacher without a simulator make, and how far is it from Track A at the same size and data?

Everything runs on the GPU servers: data generation, training, serving, and evaluation. The laptop only reads results.

## Fixed across every experiment

- **Validation set:** `data/p1/val` (8,204 states from seeds 900-944), never trained on. Agreement is always measured against `oracle-pellets-5s` on these states, in both tracks, so all numbers share one yardstick.
- **Evaluation seeds:** 100-109, never used for data.
- **Game protocol:** 10 games, 5-minute cap, 1x speed. Lockstep (the game waits for every answer: decision quality alone) and realtime (a slow answer costs game time: what a player would see). Realtime varies by up to about 150 pellets between 10-game runs, so realtime numbers are the mean of 3 runs, and compared models alternate in one session.
- **Serving:** vLLM on one evaluation GPU, so latency is comparable. The game and the scorers speak Ollama's `/v1/systemone`, which vLLM does not have, so a small adapter serves `/v1/systemone` on top of vLLM (see below). The decision prompt is the one in `training/prompt.py`, byte for byte. Each decision costs two vLLM calls (tokenize, then score) plus the adapter, where Ollama made one, so realtime numbers from this setup compare with each other, not with the README ladder or earlier laptop runs.
- **Training recipe:** LoRA rank 16, alpha 32, lr 1e-4 (3% warmup, cosine), batch 32, bf16, loss on the option-letter logits. Changes to the recipe are their own experiment, never mixed into another comparison.
- **Lookahead rule:** labels may come from rollouts; a student's input never contains them.

Reference points: random 27.6 pellets, greedy 200.7, Jev 177-203, the published `pacman-0.8b-qwen` 425 (LLM teacher with lookahead), the earlier oracle student `pacman-0.8b` 386, and the oracle itself 1,171.

## Track A: the ceiling

### A0. Pipeline check and speed calibration (first, small)

Train the two p1 students (0.8B, `features` and `features-ghosts`, 2,000 steps) with `training/train_students.sh`, evaluate on the server, and record seconds per step on each GPU type. Expected: about 50-55% agreement, in line with the earlier oracle student. This also gives the measured step times that every later estimate below depends on.

### A1. Data scaling at 0.8B

Same 0.8B base, both encodings, at four data sizes. The mix of players stays the same as p1 (oracle, greedy, random, 2:3:3 by games); only the number of games grows.

| Size | Scale vs p1 | Games (oracle / greedy / random) | Train seeds | Approx. states |
|---|---|---|---|---|
| p1 | 1x | 20 / 30 / 30 | 1000-1019, 2000-2029, 3000-3029 | 57k (31k rows after dedup) |
| p2 | 3.3x | 66 / 100 / 100 | 1000-1065, 2000-2099, 3000-3099 | 190k |
| p3 | 10x | 200 / 300 / 300 | 1000-1199, 2000-2299, 3000-3299 | 570k |
| p4 | 33x | 660 / 1000 / 1000 | 1000-1659, 2000-2999, 3000-3999 | 1.9M |

All of these stay inside the existing per-player seed blocks (1000-1999, 2000-2999, 3000-3999), away from 100-199 and 900-999. Row counts after dedup will be lower than the state counts; each dataset records its own.

Each run trains about 1 epoch, at least 2,000 steps. 8 runs. Question answered: does more data keep helping a small model, and where does it flatten?

### A2. Model scaling

At the best data size from A1 (or the largest affordable one), both encodings:

| Base | Why |
|---|---|
| Qwen3.5-0.8B | from A1 |
| a ~4B model (Qwen3.5-4B, the base of `tev1:4b`) | the middle of the curve |
| Qwen3.8-27B | the largest; also the Track B baseline student |

4 new runs. Question answered: how much does model size buy when the input stays the same? If the curve flattens early, the limit is the information in the input, not the model.

Each new base needs a check before training: `training/prompt.py` (used by both training and the serving adapter) must render the same text as that model's own chat template with thinking off, and the option letters must be single tokens. `train.py`'s startup checks cover the letters; the template check has been done for Qwen3.5 only, so Qwen3.5-4B and Qwen3.8-27B need it.

### A3. DAgger: train on the student's own mistakes

Take the best A2 student. Let it play (lockstep, training seeds only), label every state it visits with the oracle, add those rows to its data, and train again. Two rounds. Question answered: how much of the remaining gap is distribution shift (the student reaching states the data never showed it)?

### A4. Optional knobs, one at a time

- Soft-target temperature (25, 50, 100) or hard labels.
- Oracle horizon and samples (`oracle-pellets-8s-x2`) for better labels.
- Full fine-tuning instead of LoRA, at 0.8B.
- Shorter prompts for `features-ghosts` (latency in realtime).

## Track B: an LLM teacher without a simulator

### B0. How good is the teacher?

Ask DeepSeek on the p1 validation states and score it against the oracle, with no rollouts in its prompt. Then let it play 10 lockstep games as a player. Conditions: `features` and `features-ghosts` input, thinking off and on. This uses the existing `teacher` policy (any OpenAI-compatible endpoint via `TEACHER_BASE_URL` and `TEACHER_MODEL`). For comparison, Qwen3.8-27B without lookahead lost all its lives within 39-93 s in section 9 of `docs/model_evaluation.md`.

### B1. Label the same states

Label the same training states as Track A with DeepSeek in its best B0 condition (start with p2, 100k). Labels are hard; with several samples per state (temperature above 0), the share of each answer gives a soft target instead.

### B2. Train and compare

Train students on the DeepSeek labels at the same sizes and data as Track A (0.8B, and the best A2 size). The gap between a Track B student and its Track A twin is the price of having no simulator.

## Serving with vLLM

The adapter (`training/systemone_vllm.py`) answers `POST /v1/systemone` the way Ollama does, so the game, `training/score_ollama.py`, and DAgger play need no change:

1. Render the request with `training/prompt.py`: the same system prompt, user message, and chat template the student was trained on.
2. Ask vLLM's `/v1/completions` for one token at temperature 0 with its log-probabilities, restricted to the option letters (A, B, C, ...) of this request.
3. Take a softmax over those letters only and return the choice, every option's probability, and a confidence, in Ollama's response shape.

Students are served as the base model plus their LoRA adapters (`--enable-lora`), so one vLLM process can host several students of one size. If vLLM cannot attach LoRA to Qwen3.5's Gated DeltaNet projections, the fallback is to merge each adapter (`training/merge.py`) and serve the merged weights. Two checks before any evaluation: the adapter's probabilities match `train.py`'s own validation numbers on the same states (parity), and the letter log-probabilities come from the raw model output, not a distribution already renormalized by the restriction. GGUF export stays available for the laptop and iPhone demos but is no longer part of the evaluation.

DeepSeek, also on vLLM, already speaks the OpenAI chat API that the project's `teacher` policy uses (`TEACHER_BASE_URL`, `TEACHER_MODEL`, `TEACHER_EXTRA_BODY` for the thinking switch).

## Compute estimates (to be replaced by A0's measured times)

Assumptions: 0.8B at about 1 s per step (the RTX 5090 ran 0.9 s with the shorter `features` prompt), roughly 5x that for 4B and 30x for 27B, and 1.5x for `features-ghosts`.

| Runs | Steps each | Estimate |
|---|---|---|
| A0: 2x 0.8B | 2,000 | under 1 hour each |
| A1: 8x 0.8B | 2,000-31,000 | from under 1 hour (p1) to about 9-14 hours (p4) each |
| A2: 2x 4B | about 3,000 (p2) | about 4-6 hours each |
| A2: 2x 27B | about 3,000 (p2) | about 25-40 hours each on one GPU; more GPUs or less data shorten it |
| A3: 2 rounds | as A2 | as A2, plus lockstep play to collect states |
| B1: labels | 100k states | depends on DeepSeek's throughput on our GPUs |

Data generation is CPU work: about 185 oracle labels per second per process on an M4, so p4's 1.9M states take about 3 hours on one process and minutes across a many-core server.

## Code

Done (first batch):

1. `scripts/make_oracle_dataset.sh p1|p2|p3|p4`: size presets, parallel seed chunks, p1's validation set for all.
2. `training/train_students.sh` with `BASE`, `SIZE`, `GRAD_CKPT`, `MICRO_BATCH`, `STUDENTS`, `GPUS`.
3. `training/systemone_vllm.py` (the adapter), `scripts/serve_vllm.sh` (vLLM with LoRA students, or merged models), `scripts/eval_models.sh` and `scripts/collect_results.py` (the fixed protocol and one results table).

Still to write, after A0 runs:

4. The template check for a new base (Qwen3.5-4B, Qwen3.8-27B) against `training/prompt.py`.
5. A DAgger script: a student plays, the oracle labels, the rows join the data.
6. Track B: teacher probes against the oracle on the validation set, labeling at scale with retries, and sampled soft labels.

## Open questions

1. The exact model IDs to train: is there a bf16 checkpoint of Qwen3.8-27B (not only NVFP4), and which ~4B base?
2. The DeepSeek endpoint: URL, model name, how to turn thinking on and off, and how many parallel requests it takes. (Filled in on the server, in `.env`.)
3. The servers: how many GPUs of each type, on how many machines, and is Node 20+ available there?
4. vLLM: which version, can we start our own vLLM processes on the training servers for evaluation, and is LoRA serving (`--enable-lora`) allowed?
