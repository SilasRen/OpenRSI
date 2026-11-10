# SOPO: fixed two-step lookahead

The four OpenRSI launch scripts enable SOPO by default. To make the settings
explicit in an RL `.env` configuration, use:

```bash
SOPO_CURRICULUM=1
SOPO_STEPS=2
SOPO_REWARD_MODE=best
SOPO_LAMBDA=1.0
```

All four launch profiles forward these settings to Ray workers. Start with the
synchronous profile so continuation evaluations do not span asynchronous model
weight updates. Set `SOPO_STEPS=0` to disable lookahead or
`SOPO_CURRICULUM=0` to disable state resampling. Both are enabled by default in
these launch scripts. The ordinary evaluation path is unchanged.

Each training sibling is evaluated, then continued along its own immediate
predecessor for exactly two additional generations and sandbox evaluations.
Regressions are retained as continuation parents. The continuation instruction
asks for improvement and permits repairing errors. There is no branching or
selection of a different database parent during this lookahead.
Only the original response tokens and log probabilities enter policy training.
Descendants are auxiliary evaluations, excluded from the original group reward
barrier and database selection pool. Their responses, scores, status, and static
values are stored in the root's `sopo_trajectory` metadata. Original sandbox,
hack checks, validation requirements and validation-gap penalties are reused.
Held-out scores are not included in continuation prompts.

## Reward modes

`best` uses the largest eligible static value among the root and its descendants.
Example: 0.70 -> 0.60 -> 0.95 receives 0.95. Lambda does not apply in this mode.

`credit` uses the original single-step reward plus
`lambda * max(0, v1-v0) + lambda**2 * max(0, v2-max(v0,v1))`.
Only valid finite-scored nodes can establish records. Lambda zero recovers the
original reward, although extra lookahead still runs; use steps zero to avoid it.
The original reward can use adaptive group bounds, while the added improvements
use a fixed static score mapping. Their relative scale should be checked before
large experiments. Records are branch-local, not global database records, so
completion order cannot determine which sibling gets credit.

Invalid/hacked roots retain their original penalty and cannot gain reward through
descendants. They still receive the same continuation budget. Invalid descendants
do not establish records, but remain parents for the next repair/improvement step.
A generation abort raises an error rather than silently using incomplete lookahead;
existing asynchronous workers discard failed groups. Transport failures follow the
existing sandbox handling. There is no extra SOPO retry mechanism.

Budget is approximately three times the generations and sandbox jobs, not
necessarily three times wall time. Two steps measure short-horizon potential,
not causal proof that the root enabled a descendant's success. No curriculum
bandit, value model, or search distillation is included in this implementation.

## Local checks

```bash
python3.12 -m unittest discover -s OpenMLE-ERL/RL/tests -v
```

Mock tests cover the branch chain, first-response preservation, disabled mode,
discounting, and configuration rejection. Real SGLang/sandbox/GPU training must
still be validated on the target server before a full run.

## Joint lookahead and curriculum

To override the defaults explicitly, add to the existing launch configuration:

```bash
SOPO_CURRICULUM=1
SOPO_STEPS=2
SOPO_REWARD_MODE=best
SOPO_EXPLORATION=0.3
SOPO_PRIORITY_DECAY=0.99
```

The launch scripts select `sopo_data_source.SopoDataSource`. Arms represent a
specific task, parent code(s), operator and generated operator prompt. Replay
restores that state instead of selecting another parent from the live database.
Fresh groups have new sample/group identifiers. Each completed sibling group
updates its arm with the population variance of final SOPO rewards; subsequent
observations use an EMA (80% previous estimate, 20% current variance).

With probability 0.3 sampling follows the original dataset stream (shuffled if
configured); otherwise it selects a recorded state proportionally to variance,
decayed by 0.99 per completed curriculum group since its last update. If no useful
record exists, it falls back to the dataset. This increases the rate of states
where candidate choices affect later results, not merely high-scoring tasks.
The pool retains up to 512 recently updated states. Incomplete groups never
update priorities; pending groups are bounded to 4096. A lock protects shared
state between asynchronous rollout and sampling. The current rollout manager
constructs the data source and runs reward evaluation in the same process.

Bandit priorities and RNG state are saved with the dataset checkpoint. The
original dataset stream keeps its existing checkpoint behavior. Evaluation does
not update the curriculum. These weights do not claim causal contribution:
continuation sampling noise can also produce reward variance. Validate against
a fixed-budget uniform-sampling baseline on the server.
