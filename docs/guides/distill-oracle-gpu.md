# Train and evaluate oracle students on GPU servers

This guide runs one experiment of [plan_ceiling.md](../plan_ceiling.md) end to end on a GPU server: build an oracle-labeled dataset, train two students on it, serve them with vLLM, and evaluate them. It starts with A0: two 0.8B students on `p1`. Later experiments change the dataset (`p2`-`p4`) or the base model (`BASE`, `SIZE`) and nothing else.

The two students of a dataset learn from the same states and the same labels and differ only in what they read:

| Student | Input encoder | What it sees about ghosts |
|---|---|---|
| `pacman-<size>-<tag>-features` | `features` | the nearest normal ghost per option (as the published model) |
| `pacman-<size>-<tag>-ghosts` | `features-ghosts` | every ghost: `ghost2`-`ghost4` per option, and each ghost's name, state, heading, and steps, with the scatter/chase timer |

The labels come from `oracle-pellets-5s`, the rollout search valued by pellets eaten (1,171 pellets per game on the ladder seeds, against 1,116 for the points-valued `oracle-5s`). They are soft targets: a softmax over the oracle's per-option values at temperature 50 (about five pellets). Lookahead labels the data only; the students read the current state.

```
make_oracle_dataset.sh p1 -> train_students.sh -> serve_vllm.sh (vLLM + /v1/systemone adapter) -> eval_models.sh -> results.md
```

## Status of these scripts

Run end to end on an Apple M4, with fakes where a GPU is needed: the dataset builder (rebuilding `p1` from scratch gave byte-identical rows, also when split across processes), the `/v1/systemone` adapter (unit tests against a fake vLLM, and the game client playing through it), `serve_vllm.sh` in both modes with a fake `vllm` command, `eval_models.sh` with the results table, and `train_students.sh`'s orchestration with a fake trainer. Not yet run on a GPU: `setup_cuda.sh`, real training, and real vLLM. If a step fails, [If a step fails](#if-a-step-fails) lists what to send back.

## Requirements on the server

- An NVIDIA driver that supports CUDA 12.8 (Blackwell needs it): check with `nvidia-smi`.
- [uv](https://docs.astral.sh/uv/), Node 20+ with npm, git, and python3.
- vLLM: the one already on the servers, or `WITH_VLLM=1 training/setup_cuda.sh` installs one into `training/.venv-vllm`.

GPU notes: the H200 is Hopper (compute capability 9.0); the RTX Pro 6000 and the RTX 5090 are Blackwell (12.0). All train in bf16. The published model trained at about 0.9 s per step on one RTX 5090.

## 1. Set up (once per server)

```bash
git clone -b local/learning-poc https://github.com/dongliutony/decision-pacman.git
cd decision-pacman
npm ci
training/setup_cuda.sh               # WITH_VLLM=1 to also install vLLM
```

`setup_cuda.sh` creates `training/.venv` with a CUDA 12.8 PyTorch and the packages in `training/pyproject.toml`, and runs a bf16 matrix multiply on every visible GPU, so a PyTorch build without the GPU's architecture fails here and not an hour into training. The output should end with `kernel check passed on every GPU`.

## 2. Build the dataset

```bash
scripts/make_oracle_dataset.sh p1        # JOBS=<n> sets the parallel processes (default: all cores)
```

It writes `data/p1/` (raw states) and `data/sft_p1/{sft_features,sft_ghosts}/{train,val}.jsonl`: for p1, 30,566 and 35,711 training rows, 4,850 and 6,414 validation rows. The ghost encoding dedups less, since states that look the same in `features` can differ in ghosts further away. The builder is deterministic: any machine builds the same rows. Every preset uses p1's validation states.

## 3. Train

Inside `tmux`, so a dropped SSH connection does not stop it:

```bash
GPUS="0 1" training/train_students.sh data/sft_p1
```

Defaults: base `Qwen/Qwen3.5-0.8B`, 2,000 steps of 32 rows for both students (equal steps mean both see the same number of rows), LoRA rank 16, alpha 32, lr 1e-4 with 3% warmup and cosine decay, bf16, no gradient checkpointing, a validation check every 200 steps on 2,000 rows. Output goes to `runs/students_p1_0.8b/`. The GPUs may differ in speed; that changes wall time only.

Other runs of the plan change only the environment:

```bash
BASE=Qwen/Qwen3.5-4B SIZE=4b GPUS="2 3" training/train_students.sh data/sft_p2
BASE=<Qwen3.8-27B bf16 id> SIZE=27b GPUS="4 5" GRAD_CKPT=true training/train_students.sh data/sft_p2
STUDENTS=sft_ghosts GPUS=6 training/train_students.sh data/sft_p1      # one student only
```

`STEPS`, `MICRO_BATCH`, and `EXTRA_ARGS` (for example `EXTRA_ARGS="--lr 2e-4"`) change the rest. A new base needs the check in [plan_ceiling.md](../plan_ceiling.md#a2-model-scaling) first.

Watch with `tail -f runs/students_p1_0.8b/*.log`. In order:

- `projection check: max |diff| = ...`: must be under 0.5 (it was 0.006 on a T4).
- `before training: {'ce': ..., 'acc': ...}`: about 0.30 agreement, close to guessing.
- `step N/2000 loss ... (X s/step, about M min left)` every 20 steps. The first step is slow while kernels compile.
- `eval step 200: ...` and so on: `acc` is agreement with the oracle's best option on validation states, `acc_decisive` the same on states where the oracle's target is above 0.9, and `ce` the cross-entropy against the soft target. The lowest `ce` possible on p1's targets is about 0.58.

At the end the script prints the validation curve and the command that serves these students, also saved as `serve.txt` in the output folder.

## 4. Serve with vLLM

In a second `tmux` window, on a free GPU:

```bash
GPU=7 $(cat runs/students_p1_0.8b/serve.txt)
```

That runs `scripts/serve_vllm.sh Qwen/Qwen3.5-0.8B pacman-0.8b-p1-features=<adapter> pacman-0.8b-p1-ghosts=<adapter>`: one vLLM process with the base and both LoRA adapters, plus the `/v1/systemone` adapter on `http://127.0.0.1:11500`. Set `VLLM=training/.venv-vllm/bin/vllm` to use the vLLM from step 1. It stays in the foreground; Ctrl+C stops both.

The adapter (`training/systemone_vllm.py`) renders each request with `training/prompt.py`, has vLLM apply the model's own chat template with thinking off (as `train.py` does), appends each option letter, and reads that letter's log-probability at the answer position. It returns a softmax over the letters and Ollama's confidence, in Ollama's response shape, so the game and the scorers work unchanged.

If vLLM refuses LoRA on this base (for example on Qwen3.5's Gated DeltaNet projections), merge the adapters and serve the merged weights, one GPU each:

```bash
source training/.venv/bin/activate
for s in features ghosts; do python training/merge.py --adapter runs/students_p1_0.8b/sft_$s/adapter --out runs/students_p1_0.8b/sft_$s/merged; done
MERGED=1 GPUS="6 7" scripts/serve_vllm.sh - pacman-0.8b-p1-features=runs/students_p1_0.8b/sft_features/merged pacman-0.8b-p1-ghosts=runs/students_p1_0.8b/sft_ghosts/merged
```

Before an evaluation that takes hours, send one request and check that every option gets a probability:

```bash
curl -s http://127.0.0.1:11500/v1/systemone -d '{"model":"pacman-0.8b-p1-features","state":{"lives":3},"questions":{"move":{"type":"choice","instructions":"test","criteria":{"up":null,"left":null,"back":"turn around now"}}}}'
```

The adapter rests on three vLLM behaviors that the tests on the Mac could only fake: `/tokenize` applies the chat template with `chat_template_kwargs` (thinking off) and returns the same token ids as training; `/v1/completions` with `prompt_logprobs` returns the log-probability of the actual letter token even when it is not the most likely token; and vLLM attaches LoRA to Qwen3.5's `in_proj_*` and `out_proj` layers (if not, `vllm.log` says so at startup, and the merged mode above applies). If any decision fails, the adapter reports it, and `runs/serve_<time>/adapter.log` and `vllm.log` show why.

Then check parity: the agreement that step 5 reports should match the last `eval step` line of training to within about 2 points. The two use different samples of the same validation set (1,000 shuffled rows against the first 2,000), so they do not match exactly.

## 5. Evaluate

```bash
BASE_URL=http://127.0.0.1:11500 scripts/eval_students.sh p1
```

For each student, reading its own encoding, `scripts/eval_models.sh` reports agreement with the oracle on 1,000 validation states, 10 lockstep games (the game waits for every answer: decisions alone), and 3 rounds of 10 realtime games (a slow answer costs game time), all on seeds 100-109 with a 5-minute cap, the students alternating. Results go to `runs/eval_p1-0.8b_<time>/results.md`. Any set of served models can be compared the same way:

```bash
MODELS="pacman-0.8b-p1-features:features pacman-4b-p2-features:features" BASE_URL=http://127.0.0.1:11500 scripts/eval_models.sh mixed
python3 scripts/collect_results.py runs/eval_*      # one table over several evaluations
```

What to compare with: the earlier oracle-labeled student `pacman-0.8b` reached 53% agreement after about 1,200 of 1,975 steps and then flattened, and scored 386 pellets per game (`docs/model_evaluation.md`, sections 11 and 18). A Kaggle T4 run of these two students stopped at 600 steps with 48.8% (`features`) and 40.7% (`ghosts`), both still rising, which is why this run uses 2,000.

## Track B settings

Track B's teacher is DeepSeek on our own vLLM. Put its endpoint in a gitignored `.env` in the repository root on the server, as `.env.example` lists:

```bash
TEACHER_BASE_URL=<DeepSeek vLLM base URL, ending in /v1>
TEACHER_MODEL=<served model name>
TEACHER_EXTRA_BODY=<JSON merged into each request, e.g. the switch that turns thinking off>
```

The Track B scripts come after A0.

## Optional: GGUF for Ollama on a laptop

`WITH_GGUF=1 training/setup_cuda.sh` adds the llama.cpp converter; `EXPORT=gguf training/train_students.sh ...` then also merges each adapter, converts it to Q8_0, and writes a tar bundle. On a Mac, `scripts/import_students.sh <bundle>` imports it into Ollama with a Modelfile built from `training/prompt.py` (with the published GGUF it reproduced the published model's probabilities exactly), and `scripts/eval_students.sh` evaluates it there.

## If a step fails

Send back:

- Setup: the last 30 lines of `training/setup_cuda.sh`'s output, and the `nvidia-smi` table it prints.
- Training: `head -n 60` and `tail -n 40` of `runs/<students>/<student>.log`.
- Serving: `tail -n 60 runs/serve_<time>/vllm.log` (or `vllm_<name>.log`) and `runs/serve_<time>/adapter.log`.
- Evaluation: the terminal output and the failing `games_*.log` in the eval folder.

## Lessons from the Kaggle T4 attempt

- A T4 has no bf16, and Unsloth refuses fp16 for Qwen3.5, so it trains in fp32: about 21 s per step with `--grad-checkpoint true` and an estimated 24 s with the default offloading mode, and out of memory without checkpointing even at 8-row micro-batches. The first step took about 3.5 minutes compiling kernels.
- 600 steps was too few for oracle soft labels. The 600-step plateau in `distill-mlx.md` holds for the teacher's hard labels; oracle-labeled students kept improving past 1,000 steps.
- `/kaggle/working` in an interactive (draft) session lives only as long as the session. A merge step hung, the kernel would not restart, and a page reload showed "Notebook not found", with the adapters not yet downloaded. Download results as soon as training ends, and run long jobs with "Save Version > Save & Run All", or on a server.
- A newly added Kaggle input can mount as an empty folder; `kagglehub.dataset_download` worked. `training/kaggle_train.ipynb` does this.
