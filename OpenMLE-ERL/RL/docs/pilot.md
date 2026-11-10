# Frozen-model feasibility pilot

Use `sopo_pilot.py` before spending a full RL training budget. This CLI never
loads an optimizer or updates weights. It calls an existing OpenAI-compatible
model server (default model name Qwen/Qwen3-8B) and the real OpenMLE sandbox.
The existing training entrypoints remain available for the later RL comparison.

## Prerequisites

Python >=3.10; install `requirements-pilot.txt` in the intended environment.
Serve a fixed 4B/8B checkpoint separately and pass its exact served model name
and `/v1` URL. GPU hosting configuration depends on the server; this script does
not reserve Slurm GPUs or start a model service. Keep weights frozen across phases.
Set `GPU_BASE_URL`, `SANDBOX_API_KEY`, `GPT_BASE_URL`, `GPT_API_KEY` for the existing
sandbox and independent code-validity judge. The pilot validates these before generating
any candidate. On the user’s cluster, only Apptainer is currently available; the released
OpenMLE Sandbox deployment requires a controller/worker service and Docker. Therefore the
user must obtain an administrator-provided compatible sandbox endpoint and judge credentials
before submitting this pilot. The current cluster inventory does not provide them. Do not
replace these with mock scores or run generated programs directly on the shared login node. Optional `MODEL_API_KEY` authenticates
the generation server. No secrets go in the input manifest. Random proxy sandbox
scores are explicitly rejected. Generated programs only run in the sandbox.
The released sandbox helper requests GPU resources even for tabular programs;
this pilot does not claim to provide a CPU-only sandbox backend.

Create a real JSONL state file using `configs/pilot_states.example.jsonl` as a
schema example (its placeholders are not runnable). Each row is one fixed task,
parent program and operator prompt. `messages` is the complete first-step prompt;
`task_messages` is the original public task without the original parent or private
scores. Continuations append only their own preceding response to this public task.
Provide `task_id`, unique `state_id`, sandbox `data_dir`, metric direction and fixed
`score_min`/`score_max`. Use identical score bounds for every state of the same task,
chosen before seeing these runs. Utilities are clipped to [0,1]; inspect raw scores
for saturation. Invalid programs receive zero utility and remain repairable.

## Small pilot (2 states, 8 siblings, 2 continuation steps)

Run from the repository root, replacing states.jsonl and model settings:

```bash
python3.12 OpenMLE-ERL/RL/sopo_pilot.py --states states.jsonl --output pilot-run --plan-only
python3.12 OpenMLE-ERL/RL/sopo_pilot.py --states states.jsonl --output pilot-run
python3.12 OpenMLE-ERL/RL/sopo_pilot.py --states states.jsonl --output pilot-run --phase replicate
python3.12 OpenMLE-ERL/RL/sopo_pilot.py --states states.jsonl --output pilot-run --phase fresh
python3.12 OpenMLE-ERL/RL/sopo_pilot.py --output pilot-run --phase report
```

Discovery: at most 48 generations/evaluations. Replication: another 96 (all 16
roots, each continued twice, independently three times). Fresh candidates:
another 48. Total: at most 192, not 48. Judge-invalid/empty code consumes generation
budget but does not execute. The independent judge also consumes separate API calls.
Default maximum sandbox job time is 300 seconds; wait time adds 300 seconds.
Default generation limit is 8192 tokens, concurrency 2. These are budget controls,
not promised completion times. Inspect truncated-response and validity rates.

Scale to 24 states with `--num-states 24`: discovery 576, replication 1152, fresh
576, total 2304 possible sandbox evaluations. Select 8 tasks with 3 fixed states
per task in the JSONL. Repeats are adjustable using `--repeats`. Running only the
first phase is allowed; the report then emits just the first chart.

Use the same configuration for every phase. Per-node cache permits resuming a
failed run; changed states/model/sampling/budget settings require a new output
directory. Cache is written only after the evaluation returns. A crash during a
remote job can leave that job running; a retry can submit it again. Infrastructure
failures stop the run, while invalid generated code is a measured failure.

## Outputs and interpretation

- `01_initial_vs_future.png`: discovery first-step vs best-of-three utility.
- `02_independent_selection.png`: select candidates using discovery only; compare
  those candidates using independently generated continuations. Negative effects
  are retained. Roots are shared, so this is conditional continuation validation.
- `03_variance_repeatability.png`: discovery variance vs variance of a fresh group
  of candidates from the same state. No adaptive sampling is used in this pilot,
  avoiding preferential selection bias in the diagnostic comparison.
- `summary.json`, `report.md`, phase JSON files and per-node caches retain results.

These plots diagnose ranking changes and curriculum signal, not learned policy
improvement or causal credit. Best-of-three includes the root and is upward-biased.
Two-state smoke tests are insufficient evidence; inspect repeated effects across
tasks, validity rates, raw-score saturation and generation/execution cost. No
statistical significance claim or automatic go/no-go is made. Subsequent RL should
compare single-step+uniform, lookahead+uniform and lookahead+curriculum under equal
execution/time budgets on held-out tasks and multiple seeds.
