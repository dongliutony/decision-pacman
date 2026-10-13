# Train the oracle students on a CUDA server

This guide trains two 0.8B students on the `p1` dataset and compares them. Both learn from the same game states and the same labels; they differ only in what they read:

| Student | Input encoder | What it sees about ghosts |
|---|---|---|
| `pacman-0.8b-p1-features` | `features` | the nearest normal ghost per option (as the published model) |
| `pacman-0.8b-p1-ghosts` | `features-ghosts` | every ghost: `ghost2`-`ghost4` per option, and each ghost's name, state, heading, and steps, with the scatter/chase timer |

The labels come from `oracle-pellets-5s`, the rollout search valued by pellets eaten (1,171 pellets per game on the ladder seeds, against 1,116 for the points-valued `oracle-5s`). They are soft targets: a softmax over the oracle's per-option values at temperature 50 (about five pellets). Lookahead labels the data only; the students read the current state.

The flow: build the data (any machine with Node), train on the server, bring back two GGUF files, import them into Ollama on the Mac, and evaluate there.

```
server: setup_cuda.sh -> make_oracle_dataset.sh -> train_students.sh -> students_p1_*.tar
mac:    import_students.sh students_p1_*.tar -> eval_students.sh
```

## Status of these scripts

The data script and the two Mac-side scripts were run end to end on an Apple M4: rebuilding `p1` from scratch gave byte-identical rows (after sorting run folders by player and seed in `build_sft.py`), and the generated Modelfile served the published GGUF with exactly the published model's probabilities. The two server scripts (`training/setup_cuda.sh`, `training/train_students.sh`) were checked for syntax only; they have not run on a CUDA machine yet. If a step fails, the section [If a step fails](#if-a-step-fails) lists what to send back.

## Requirements on the server

- An NVIDIA driver that supports CUDA 12.8 (Blackwell needs it): check with `nvidia-smi`.
- [uv](https://docs.astral.sh/uv/) for the Python environments.
- Node 20+ with npm, only if the server builds the data itself (step 2).
- git, and about 20 GB of disk for the base model cache, merged weights, and llama.cpp.

GPU notes: the H200 is Hopper (compute capability 9.0), the RTX Pro 6000 and the RTX 5090 are Blackwell (12.0). All three train in bf16. The published model trained at about 0.9 s per step on one RTX 5090.

## 1. Set up (once)

```bash
git clone -b local/learning-poc https://github.com/dongliutony/decision-pacman.git
cd decision-pacman
training/setup_cuda.sh
```

It creates `training/.venv` with a CUDA 12.8 PyTorch and the packages in `training/pyproject.toml`, runs a bf16 matrix multiply on every visible GPU (so a PyTorch build without the GPU's architecture fails here, not an hour into training), and clones llama.cpp at tag `b11298` into `~/llama.cpp` with its own `.venv` for the GGUF converter. Set `LLAMA_CPP_DIR` to put llama.cpp elsewhere. The last lines should read `kernel check passed on every GPU` and `converter ready`.

## 2. Get the data

Either rebuild it on the server (about 3 minutes of CPU on an M4; deterministic, so the rows match the Mac's):

```bash
npm ci
scripts/make_oracle_dataset.sh p1
```

or copy the Mac's copy over (6 MB):

```bash
scp data/sft_p1.zip server:decision-pacman/data/     # on the Mac
cd data && unzip sft_p1.zip && cd ..                 # on the server
```

Training needs `data/sft_p1/{sft_features,sft_ghosts}/{train,val}.jsonl`: 30,566 and 35,711 training rows, 4,850 and 6,414 validation rows. The ghost encoding dedups less, since states that look the same in `features` can differ in ghosts further away.

## 3. Train

Start a `tmux` session first, so a dropped SSH connection does not stop the run.

Both students on one server with two GPUs:

```bash
GPUS="0 1" training/train_students.sh data/sft_p1 runs/students_p1
```

One student per server (for example the H200 on one machine and the RTX Pro 6000 on another):

```bash
STUDENTS=sft_features GPUS=0 training/train_students.sh data/sft_p1 runs/students_p1   # server A
STUDENTS=sft_ghosts   GPUS=0 training/train_students.sh data/sft_p1 runs/students_p1   # server B
```

Settings (from `train_students.sh`): 2,000 steps of 32 rows for both students, LoRA rank 16, alpha 32, lr 1e-4 with 3% warmup and cosine decay, bf16, no gradient checkpointing, a validation check every 200 steps on 2,000 rows. Equal steps mean both students see the same number of rows. The GPUs differ in speed; that changes wall time only, since both students get the same data order, steps, and seed. `STEPS` and `EXTRA_ARGS` (for example `EXTRA_ARGS="--lr 2e-4"`) change the settings.

Watch progress with `tail -f runs/students_p1/*.log`. In order, the log shows:

- `projection check: max |diff| = ...`: must be under 0.5 (it was 0.006 on a T4).
- `before training: {'ce': ..., 'acc': ...}`: about 0.30 agreement, close to guessing.
- `step N/2000 loss ... (X s/step, about M min left)` every 20 steps. The first step is slow while kernels compile.
- `eval step 200: ...` and so on: `acc` is agreement with the oracle's best option on validation states, `acc_decisive` the same on states where the oracle's target is above 0.9, and `ce` the cross-entropy against the soft target. The lowest `ce` possible on these targets is about 0.58.

After training, the script merges each adapter and converts it to Q8_0 (`training/export_gguf.sh`), prints the validation curve, and writes one bundle: `runs/students_p1/students_p1_features-ghosts.tar` (or one per student when trained apart) with the GGUFs, their checksums, the logs, and the adapters.

## 4. Bring the bundles to the Mac

```bash
scp server:decision-pacman/runs/students_p1/students_p1_*.tar .
```

## 5. Import into Ollama

```bash
brew services start ollama
scripts/import_students.sh students_p1_*.tar
```

It checks the checksums and creates `pacman-0.8b-p1-features` and `pacman-0.8b-p1-ghosts`. Each Modelfile takes its system prompt from `training/prompt.py` (the one the students were trained under, byte for byte), uses `TEMPLATE {{ .Prompt }}`, and declares `CAPABILITY decision` so Ollama serves `/v1/systemone`.

## 6. Evaluate

```bash
scripts/eval_students.sh p1 10
```

It needs the raw validation states in `data/p1/val` (from step 2 on the Mac). For each student, reading its own encoding, it reports:

1. Agreement with the oracle on 1,000 validation states (`training/score_ollama.py`).
2. 10 games on the ladder protocol (seeds 100-109, 5-minute cap) in lockstep: the game waits for every answer, so this measures decisions alone.
3. The same in realtime: a slow answer costs game time. The ghost student's prompt is about 600 tokens against 400, which costs about 70 ms per decision on an M4.

The students alternate, so they face the same machine load. On an M4 the full run takes several hours; `N=200 MAX_SECONDS=60 scripts/eval_students.sh p1 2` gives a first look in minutes. Results go to `runs/eval_p1_<time>/`.

What to compare with: the earlier oracle-labeled student `pacman-0.8b` reached 53% agreement after about 1,200 of 1,975 steps and then flattened, and scored 386 pellets per game (`docs/model_evaluation.md`, sections 11 and 18). A Kaggle T4 run of these two students stopped at 600 steps with 48.8% (`features`) and 40.7% (`ghosts`), both still rising, which is why this run uses 2,000.

## If a step fails

Send back:

- Setup: the last 30 lines of `training/setup_cuda.sh`'s output, and the `nvidia-smi` table it prints.
- Training: `head -n 60 runs/students_p1/<student>.log` and `tail -n 40 runs/students_p1/<student>.log`.
- Export: `tail -n 40 runs/students_p1/<student>.export.log`.
- Import or evaluation on the Mac: the full terminal output.

## Lessons from the Kaggle T4 attempt

- A T4 has no bf16, and Unsloth refuses fp16 for Qwen3.5, so it trains in fp32: about 21 s per step with `--grad-checkpoint true` and an estimated 24 s with the default offloading mode, and out of memory without checkpointing even at 8-row micro-batches. The first step took about 3.5 minutes compiling kernels.
- 600 steps was too few for oracle soft labels. The 600-step plateau in `distill-mlx.md` holds for the teacher's hard labels; oracle-labeled students kept improving past 1,000 steps.
- `/kaggle/working` in an interactive (draft) session lives only as long as the session. A merge step hung, the kernel would not restart, and a page reload showed "Notebook not found", with the adapters not yet downloaded. Download results as soon as training ends, and run long jobs with "Save Version > Save & Run All", or on a server.
- A newly added Kaggle input can mount as an empty folder; `kagglehub.dataset_download` worked. `training/kaggle_train.ipynb` does this.
