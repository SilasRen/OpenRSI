"""SOPO: fixed-budget lookahead credit for every sibling; only the root response is optimized.

Each of the 16 siblings is continued along its own branch for SOPO_STEPS more operator
calls (Improve if it runs, Debug if it crashed) and its reward becomes the credit of
that branch. SOPO_STEPS=0 (or credit mode with SOPO_LAMBDA=0) recovers the original reward.
"""
import copy
import math
import os


def settings():
    mode = os.getenv("SOPO_REWARD_MODE", "best")
    if mode not in {"best", "credit"}:
        raise ValueError("SOPO_REWARD_MODE must be best or credit")
    steps = int(os.getenv("SOPO_STEPS", "2"))
    decay = float(os.getenv("SOPO_LAMBDA", "1.0"))
    if steps < 0 or not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError("SOPO_STEPS must be >= 0 and SOPO_LAMBDA must be in [0, 1]")
    return steps, decay


def lookahead_bonus(nodes, decay):
    """Discount improvements over the best valid value on this branch so far."""
    best = float(nodes[0]["static_value"]) if nodes[0]["eligible"] else 0.0
    bonus = 0.0
    for depth, node in enumerate(nodes[1:], 1):
        if node["eligible"]:
            value = float(node["static_value"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite SOPO reward")
            bonus += decay ** depth * max(0.0, value - best)
            best = max(best, value)
    return bonus


def node_program(node, task_name=""):
    """Wrap a lookahead node as the parent Program shown to the real operator prompt."""
    from program_database import Program
    return Program(task_name=task_name, status_code=node.get("status_code") or 0,
                   payload=copy.deepcopy(node.get("payload") or {}), code=node.get("code") or "",
                   score=node.get("score"))


def continuation_messages(source, public_prompt, node):
    """Build the same Improve/Debug operator prompt that RL and search use."""
    from prompt_builder import build_prompt
    system = next(m["content"] for m in public_prompt if m["role"] == "system")
    user = next(m["content"] for m in public_prompt if m["role"] == "user")
    mode = "improve" if node["eligible"] else "debug"
    system_prompt, user_prompt = build_prompt(
        mode=mode,
        task_description=source.get("task_description") or "",
        data_description=source.get("data_description") or "",
        public_system_prompt=system,
        public_user_prompt=user,
        virtual_data_dir=source.get("data_dir") or "",
        parent_program=node_program(node, source.get("task_name") or ""),
    )
    return mode, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def combine_reward(nodes, immediate_reward, mode, decay):
    """Return (training_reward, bonus) for a root and its fixed-budget lookahead."""
    bonus = lookahead_bonus(nodes, decay)
    eligible = [float(node["static_value"]) for node in nodes if node["eligible"]]
    if mode == "best":
        # Simple version: credit = best value reached from this root.
        return (max(eligible) if eligible else float(immediate_reward)), bonus
    return float(immediate_reward) + bonus, bonus


async def extend_reward(args, sample, immediate_reward):
    steps, decay = settings()
    if not steps or sample.metadata.get("generation_aborted") or sample.metadata.get("group_generation_aborted"):
        return immediate_reward
    mode = os.getenv("SOPO_REWARD_MODE", "best")
    root = copy.deepcopy(sample.metadata["sopo_node"])
    nodes = [root]
    sample.metadata.update(sopo_reward_mode=mode, sopo_steps=steps, sopo_lambda=decay,
                           sopo_immediate_reward=immediate_reward)
    if root.get("code_category") != "valid":
        # Empty / hacked / unverifiable roots keep their penalty and cannot launder
        # it through descendants, so spending lookahead budget on them is wasted.
        sample.metadata.update(sopo_trajectory=_compact(nodes), sopo_bonus=0.0, sopo_lookahead_skipped=True)
        return immediate_reward

    # Import lazily: generate_mle invokes this after the first sandbox evaluation.
    from generate_mle import GenerateState, Sample, post, reward_func

    source = sample.metadata["sopo_source_metadata"]
    params = sample.metadata["sopo_sampling_params"]
    state = GenerateState(args)
    for depth in range(1, steps + 1):
        parent = nodes[-1]
        # Each branch follows its own immediate predecessor, including regressions:
        # runnable nodes get the Improve operator, crashed nodes get Debug.
        op_mode, messages = continuation_messages(source, sample.prompt, parent)
        async with state.semaphore:
            output = await post(
                f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
                {"text": state.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                 "sampling_params": copy.deepcopy(params), "return_logprob": False},
            )
        if output["meta_info"]["finish_reason"]["type"] == "abort":
            # Never silently reward a shorter path as if it received the full budget.
            raise RuntimeError("SOPO continuation aborted; discard/retry this rollout group")
        metadata = copy.deepcopy(source)
        for key in list(metadata):
            if key.startswith(("parent_", "crossover_", "reward_group_", "sopo_")):
                metadata.pop(key)
        metadata.update(sopo_continuation=True, evaluation=False, generation_mode=op_mode,
                        task_id=sample.metadata["task_id"], parent_code=parent["code"])
        child = Sample(prompt=messages, response=output["text"], metadata=metadata)
        await reward_func(args, child)
        child_node = copy.deepcopy(child.metadata["sopo_node"])
        child_node["operator"] = op_mode
        nodes.append(child_node)
    reward, bonus = combine_reward(nodes, immediate_reward, mode, decay)
    sample.metadata.update(sopo_trajectory=_compact(nodes), sopo_bonus=bonus)
    sample.metadata["sopo_node"].pop("payload", None)
    return reward


def _compact(nodes):
    """Sandbox payloads are only needed to prompt the next step; keep stored metadata small."""
    return [{k: v for k, v in node.items() if k != "payload"} for node in nodes]


def program_snapshot(program):
    """Avoid recursively copying earlier lookahead/replay metadata into ancestors."""
    if program is None:
        return None
    import json
    result = copy.deepcopy(program.to_dict())
    metadata = result.get("metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    result["metadata"] = {k: v for k, v in metadata.items() if not k.startswith("sopo_")}
    return result
