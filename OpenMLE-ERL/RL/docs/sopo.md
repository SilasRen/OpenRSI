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
Regressions are retained as continuation parents. Each continuation uses the
repository's own operator prompt (`prompt_builder.build_prompt`), so lookahead
measures what the search will actually do with this program: a node that ran and
scored is continued with **Improve**, a node that crashed is continued with **Debug**
(both show the node's code plus its sandbox feedback, exactly like RL parent
prompts). There is no branching or selection of a different database parent
during this lookahead.
Only the original response tokens and log probabilities enter policy training.
Descendants are auxiliary evaluations, excluded from the original group reward
barrier and database selection pool. Their responses, scores, status, and static
values are stored in the root's `sopo_trajectory` metadata. Original sandbox,
hack checks, validation requirements and validation-gap penalties are reused.
Held-out scores are not included in continuation prompts.

## Running the original baseline

The launch scripts enable SOPO by default. For the control run use the same
config with `SOPO_STEPS=0 SOPO_CURRICULUM=0`; that path is byte-for-byte the
original OpenMLE reward and data source.

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

Empty, hacked and unverifiable roots (`code_category != valid`) retain their
original penalty, cannot gain reward through descendants, and therefore skip the
lookahead entirely (no wasted model or sandbox calls). A valid root that crashed
or produced no score is still continued (Debug, then Improve) and is credited with
the best value its branch reaches; if no node on the branch scores, it keeps its
original reward. Invalid descendants do not establish records, but remain parents
for the next repair/improvement step.

The Program Database stores each root's own single-step reward (`reward` column),
not the SOPO credit. Lookahead descendants are not inserted, so their credit must
not move a program between the Improve (reward > 0) and Debug (reward <= 0) parent
pools. The SOPO credit is still returned for training and logged as `final_reward`.
A generation abort raises an error rather than silently using incomplete lookahead;
existing asynchronous workers discard failed groups. Transport failures follow the
existing sandbox handling. There is no extra SOPO retry mechanism.

Budget is at most three times the generations and sandbox jobs, not
necessarily three times wall time. Two steps measure short-horizon potential,
not causal proof that the root enabled a descendant's success. No value model or
search distillation is included; the curriculum bandit is described below.

## Local checks

```bash
python3.12 -m unittest discover -s OpenMLE-ERL/RL/tests -v
```

Mock tests cover the branch chain (with the real Improve/Debug prompts),
first-response preservation, both reward modes and lambda=0, crashed-root repair
credit, skipped lookahead for hacked roots, crash-excluded curriculum variance,
disabled mode, discounting, and configuration rejection. They need the RL
`requirements.txt` (prompt_builder imports httpx). Real SGLang/sandbox/GPU training must
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
updates its arm with the population variance of final SOPO rewards among the
siblings whose own program ran and scored; subsequent observations use an EMA
(80% previous estimate, 20% current variance). Crashed or hacked siblings are left
out of the variance: a group where half the siblings crash looks high-variance,
but all it teaches is "don't crash", not which method leads further. Groups with
fewer than two scored siblings record zero priority.

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
