# The Duck 🦆

The Duck is the ARC3 inference harness in this repo: a tool-using solver that
plays ARC-AGI-3 games through TAAF.

It ties together:

- TAAF `Benchmark` / `GameAPI` execution
- a local OpenAI-compatible vLLM server, or OpenRouter
- the duck's single ephemeral `python` tool
- structured run artifacts for scoring, viewing, and trace export

The Python package lives under `inference/`. The run viewer lives under
`viewer/`.

## Quick Start

You need Python 3.12 and `uv`.

```bash
make install
```

Run with the default local vLLM config. `make serve` submits the model-serving allocation through `coe-hpc3`; once it is ready, run the harness against it.

```bash
make serve
make interactive
```

Submit the default Slurm run:

```bash
make sbatch
```

Run through OpenRouter instead:

```bash
export OPENROUTER_API_KEY=<your-openrouter-api-key>
CONFIG_PATH=configs/inference.openrouter.json make interactive
```

Open the viewer:

```bash
make view
```

If your runs are under the checked-in default experiment root, point the viewer
there:

```bash
make view VIEW_RUNS_DIR=/shared/arc_3_results/$USER
```

## What The Duck Does

For each TAAF game run, the harness starts a `HarnessSolver`. The solver gives
the duck the latest game state, valid actions, history, and a Python tool. The
duck inspects the board, writes small bits of code to reason about it, and calls
`action(...)` from inside Python to execute real game actions.

The duck can use:

- `current_frame.ascii` for a compact symbolic grid
- `current_frame.segmentation` for connected components, object hashes,
boundaries, containment, and adjacency
- `history`, `previous_frame`, `transitions`, and `last_transition` for
before/after reasoning
- `valid_actions` for the current action set
- `last_action_result` for fields such as `board_changed`, `level_completed`,
`game_over`, `run_complete`, and `reward`

The raw numeric grid is intentionally hidden from the Python tool. The preferred
view is `current_frame.segmentation`; `current_frame.ascii` is there for small
local checks.

The model-facing actions are:

- `UP`, `DOWN`, `LEFT`, `RIGHT`
- `SPACE`
- `MOUSE(row=..., col=...)`

`MOUSE` uses `row` and `col`. Legacy `x` / `y` mouse fields are rejected.

Every Python tool call starts fresh. It can import a small allowlist of standard
library modules, print compact summaries, assign a final value to `result`, and
call `action(...)` once or many times. The tool call timeout defaults to 30
seconds.

## Configuration

The main config is strict JSON. Comments are not supported.

- `configs/inference.json` is the default local-vLLM / Slurm config.
- `configs/inference.openrouter.json` uses OpenRouter.
- `configs/eval.json` selects runs for `make eval`.
- `configs/significance.json` selects score files for `make significance`.

Use a different config with:

```bash
CONFIG_PATH=/path/to/config.json make interactive
CONFIG_PATH=/path/to/config.json make sbatch
```

Useful sections in `configs/inference.json`:

- `shared.*`: model name, base URL, provider, and context window.
- `experiments.root_dir`: where timestamped run directories are written.
- `environment.*`: games, tags, passes, concurrency, and runtime limits.
- `deployment.*`: inline vs Slurm and source repos bundled into Slurm jobs.
- `deployment.slurm.*`: GPU, walltime, partition, local-server startup, and
extra `sbatch` flags.
- Kaggle runs are configured by CLI/Make overrides because the notebook slug
and source dataset are usually per run.
- `server.*`: vLLM model-serving settings.
- `analyzer.*`: duck sampling/tool settings. This key is still named
`analyzer` for compatibility with existing code and configs.
- `chat.*`: direct chat probing with `make chat`.
- `viewer.port`: default viewer port.
- `multimodal.*`: image context for the current grid.

The checked-in default config currently runs the official tagged game set with
20 passes, 45 minutes per game, and `concurrent_jobs=16`. On Slurm it requests
two B200 GPUs and starts one local vLLM server per GPU. In that mode
`concurrent_jobs` is interpreted per GPU/server, so the effective concurrency is
32.

## Running Games

Run inline in the current process:

```bash
make interactive
```

Submit to Slurm:

```bash
make sbatch
```

Launch the validated duck harness on Kaggle:

```bash
make kaggle-duck \
  RUN_NAME=duck-harness-20260527 \
  KAGGLE_KERNEL_SLUG=taaf-duck-harness-20260527 \
  KAGGLE_DATASET_REF=driessmit1/taaf-kaggle-source-duck-harness-20260527
```

Name a run:

```bash
make sbatch RUN_NAME=baseline-qwen
```

Run one game or short-prefix:

```bash
make interactive GAME=taps N_PASSES=1 MAX_RUNTIME_MINUTES=10
```

Run the official tag set with a whole-experiment cap:

```bash
make sbatch GAME=[] GAME_TAGS=official MAX_EXPERIMENT_RUNTIME_MINUTES=360
```

Run the duck locally against TAAF's competition Arcade simulator:

```bash
make interactive \
  GAME=[] GAME_TAGS=official \
  SIMULATE_COMPETITION_ARCADE=true \
  COMPETITION_CLONE_RUNS=110 \
  N_PASSES=1
```

The simulator is inline-only. `COMPETITION_CLONE_RUNS=110` repeats the selected
25 official games with unique competition-safe IDs, which catches submission
Arcade issues without waiting for a Kaggle rerun.

Common overrides:

- `GAME`: one game, comma-separated games, or a JSON list.
- `GAME_TAGS`: include tags such as `official`.
- `EXCLUDE_GAME_TAGS`: exclude tags.
- `N_PASSES`: TAAF passes per selected game.
- `CONCURRENT_JOBS`: TAAF concurrency. With Slurm local servers this is per
GPU/server.
- `MAX_ACTIONS`: optional per-game action cap.
- `MAX_RUNTIME_MINUTES`: per-game wall-clock cap.
- `MAX_EXPERIMENT_RUNTIME_MINUTES` or `MAX_EXPERIMENT_RUNTIME_HOURS`: whole-run
wall-clock budget. If the per-game cap is unset, the runner derives it from
the number of games, passes, and effective concurrency.
- `EXPERIMENTS_DIR`: base directory for timestamped runs.
- `EXPERIMENT_DIR`: exact output directory for one run.

List resolved official games without running them:

```bash
uv run --no-sync inference-taaf-run --include-tags official --list-games
```

## Local vLLM

`make serve` is the normal command from the login node. `make server` is the
lower-level target that starts vLLM only after a Slurm GPU allocation already
exists; with the default Apptainer runtime it refuses to run outside Slurm.

One-time setup:

```bash
# Run these on the login node, where internet works.
bash container/scripts/01_pull_image.sh
bash container/scripts/03_download_model.sh

# Build the Apptainer image on a node with Apptainer.
bash container/scripts/02_build_sif.sh
```

That is the normal flow on this cluster: pull the Docker image and model weights
on the login node, then build the Apptainer image on a node that has Apptainer.
If your current shell is on the login node and Apptainer only exists on
`coe-hpc3`, run the build step through SSH:

```bash
ssh coe-hpc3 'cd ~/arc-agi-3/duck-harness/ARC3-Inference && bash container/scripts/02_build_sif.sh'
```

Submit a long-lived model-serving Slurm job through `coe-hpc3`:

```bash
make serve
```

`make serve` SSHes to `coe-hpc3` when needed, submits the allocation with `sbatch --wrap`, then runs `make serve-hold` inside the allocation. `serve-hold` calls the normal `make server` target and keeps the allocation alive while the server process is running. Slurm stdout/stderr goes to `container/logs/serve-<jobid>.out`; vLLM logs go to `.cache/arc3_runtime/arc3-inference-server.log`.

If you are already inside a suitable Slurm GPU allocation, start the server directly:

```bash
make server
```

Check or stop a server owned by the current state directory:

```bash
make check-server
make stop-server
```

The default local base URL is `http://127.0.0.1:1234/v1`. `make server`
generates a local server API key unless `SERVER_REQUIRE_API_KEY=false`.
By default it uses:

- `SERVER_APPTAINER_SIF=.cache/vllm-container/vllm-openai.sif`
- `SERVER_APPTAINER_HF_HOME=.cache/vllm-container/hf`
- `SERVER_APPTAINER_ATTENTION_BACKEND=FLASHINFER`

Override those variables if your image or model cache lives elsewhere. If you
are already inside a newer Linux container where the pinned vLLM wheel is
compatible, use `SERVER_RUNTIME=venv make server`; that path installs the
`server` extra via `make install-server`.

On cluster machines, the Makefile moves Hugging Face, Torch, Triton, and related
caches under `/shared/<user>` when that directory exists.

## Slurm Flow

`make sbatch` runs through TAAF's Slurm deployment. The job directory includes
the run config, generated Slurm script, dependency override file, benchmark
artifacts, diagnostics, and logs.

For local-vLLM Slurm runs, the solver starts local server processes inside the
allocation before the duck begins playing. Each run gets its own API key and
run-scoped localhost ports, so a run either talks to its own server or fails
fast. The server is started from the bundled `src/ARC3-Inference` snapshot in
the run directory, using the worker's per-run virtualenv.

`deployment.source_repos` is bundled into the Slurm job directory. The worker
uses a generated dependency override file so it installs those bundled repos
instead of fetching private dependencies from GitHub.

## Kaggle Flow

`make kaggle-duck` runs through TAAF's Kaggle deployment. It packages the
current TAAF and ARC3-Inference sources into a Kaggle source dataset, pushes a
private Kaggle notebook, and attaches the duck solver's declared model and vLLM
wheelhouse datasets. The duck solver owns the Kaggle setup hooks, so the
launcher only chooses the run shape.

By default the target uses the 16 public games from the duck harness validation,
`model=local`, 16 concurrent games, 75 minutes per game, and a 90-minute Kaggle
runtime. Add `DEPLOYMENT_WAIT=true` if you want the command to block and pull
the finished Kaggle output back into the run directory.

The equivalent direct CLI form is:

```bash
uv run --no-sync inference-taaf-run \
  --deployment-target kaggle \
  --kaggle-duck-public-harness \
  --agent duck-harness \
  --model local \
  --run-name duck-harness-20260527 \
  --kaggle-kernel-slug taaf-duck-harness-20260527 \
  --kaggle-dataset-ref driessmit1/taaf-kaggle-source-duck-harness-20260527 \
  --max-runtime-minutes 75 \
  --max-experiment-runtime-minutes 90 \
  --concurrent-jobs 16 \
  --analyzer-timeout 900
```

## Run Artifacts

Each run writes a timestamped directory under `experiments.root_dir`, or under
`EXPERIMENTS_DIR` / `EXPERIMENT_DIR` when overridden.

Important files include:

- `run_config.json`: resolved games, passes, concurrency, runtime caps, model,
Slurm settings, and hardware metadata.
- `benchmark.json`: saved TAAF benchmark and per-game `GameRun` state.
- `diagnostics.html`: TAAF diagnostics.
- `artifacts/*_viewer_data.json`: compact viewer payloads.
- `artifacts/*_events.jsonl`: append-only full viewer event sidecars.
- duck transcript HTML/text files linked from the viewer.
- `stdout.log` and `stderr.log` for Slurm jobs.
- `requests.jsonl` files when `analyzer.save_request_logs` is true.

## Viewer

Start the viewer on the default port from `configs/inference.json`:

```bash
make view
```

Override the port:

```bash
make view VIEW_PORT=8012
```

Point at a run root:

```bash
make view VIEW_RUNS_DIR=/shared/arc_3_results/$USER
```

Point at one exact run:

```bash
make view VIEW_RUN_DIR=/shared/arc_3_results/$USER/<run-name>
```

The viewer shows run summaries, per-game progress, boards, actions, rewards,
level transitions, and the duck's transcript.

## Scoring

Score one run directory:

```bash
make score_run SCORE_RUN_DIR=/path/to/run
```

Evaluate runs from `configs/eval.json`:

```bash
make eval
```

Write a score file somewhere specific:

```bash
make score_run SCORE_RUN_DIR=/path/to/run SCORE_OUTPUT_PATH=docs/candidate-score.json
```

The scorer reads TAAF `benchmark.json`, uses persisted `final_score` values when
present, and otherwise asks TAAF's `GameRun` scorer to compute the score from
the saved state. It writes `evaluation.json` plus the lightweight `score.json`
format used by significance checks.

## Significance

Compare a candidate score file against a current best:

```bash
make significance \
  BASELINE_SCORE=docs/current-best-score.json \
  CANDIDATE_SCORE=docs/candidate-score.json
```

Or configure those paths in `configs/significance.json` and run:

```bash
make significance
```

The comparison aligns by `game_id`, averages repeated trials within each game,
and uses games as the paired unit. It checks runtime budget, hardware, dataset
metadata, and trial counts before reporting whether the candidate passes the
internal-highscore threshold:

```text
P(true_delta > 0 | results) >= 0.90
```

The output also includes win rate, a bootstrap 90% interval, and TAAF paired
test p-values as robustness checks.

## Trace Export

Export machine-readable per-episode duck traces:

```bash
make traces
```

For runs outside `runs/`, call the tool directly:

```bash
uv run --no-sync inference-traces --runs-dir /shared/arc_3_results/$USER
```

Traces are written in live-chat `messages` format. They preserve assistant
reasoning, tool calls, compact tool results, actions, scores, and level
transitions linked back to message indices.

## Useful Commands

- `make install`: create `.venv` and install base locked dependencies.
- `make install-server`: add the vLLM server extra for `SERVER_RUNTIME=venv`.
- `make prepare-ci`: run Ruff and the test suite.
- `make serve`: submit a standalone model-serving Slurm job through `coe-hpc3`.
- `make server`: start local vLLM via Apptainer inside an existing Slurm GPU allocation.
- `make interactive`: run through TAAF inline deployment.
- `make sbatch`: submit through TAAF Slurm deployment.
- `make chat PROMPT="..."`: send a direct chat probe to the configured model.
- `make view`: serve the run viewer.
- `make score_run SCORE_RUN_DIR=...`: score one saved run.
- `make eval`: score runs selected by an eval config.
- `make significance`: compare two score files.
- `make traces`: export trace JSON.
- `make zip`: zip the local `runs/` directory.
- `make clean`: remove local `runs/` artifacts.

## Repo Map

- `inference/framework/run.py`: CLI entry point and TAAF deployment setup.
- `inference/framework/solver.py`: TAAF solver adapter, action execution,
viewer events, transcripts, and local-server orchestration.
- `inference/agent/tool_agent.py`: OpenAI-compatible tool-calling duck.
- `inference/agent/python_tool_sandbox.py`: isolated Python tool runtime.
- `inference/utils/segmentation.py`: connected-component board segmentation.
- `inference/tools/eval.py`: TAAF score export.
- `inference/tools/significance.py`: paired score comparison.
- `inference/tools/traces.py`: trace export.
- `viewer/`: local browser UI for saved runs.
- `tests/`: unit coverage for config, duck runtime, TAAF runner, viewer,
scoring, significance, and traces.

