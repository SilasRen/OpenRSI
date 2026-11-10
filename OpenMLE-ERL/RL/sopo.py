"""Fixed-budget lookahead credit; only the original response is optimized."""
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


async def extend_reward(args, sample, immediate_reward):
    steps, decay = settings()
    if not steps or sample.metadata.get("generation_aborted") or sample.metadata.get("group_generation_aborted"):
        return immediate_reward
    # Import lazily: generate_mle invokes this after the first sandbox evaluation.
    from generate_mle import GenerateState, Sample, post, reward_func

    source = sample.metadata["sopo_source_metadata"]
    params = sample.metadata["sopo_sampling_params"]
    nodes = [copy.deepcopy(sample.metadata["sopo_node"])]
    state = GenerateState(args)
    initial_messages = sample.metadata.get("sopo_initial_messages", sample.prompt)
    system = next(m["content"] for m in initial_messages if m["role"] == "system")
    task = next(m["content"] for m in initial_messages if m["role"] == "user")
    for depth in range(1, steps + 1):
        parent = nodes[-1]
        # Each branch follows its own immediate predecessor, including regressions.
        # Only public task text and generated code are shown, never held-out scores.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task + "\n\nImprove the following candidate program. "
             "Repair errors if necessary. Return a complete runnable program in a Python code block. "
             "Follow all original task and validation requirements.\n\n" + parent["response"]},
        ]
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
        metadata.update(sopo_continuation=True, evaluation=False, generation_mode="improve",
                        task_id=sample.metadata["task_id"], parent_code=parent["code"])
        child = Sample(prompt=messages, response=output["text"], metadata=metadata)
        await reward_func(args, child)
        nodes.append(copy.deepcopy(child.metadata["sopo_node"]))
    # Hacked/invalid roots cannot launder their penalty through descendants.
    bonus = lookahead_bonus(nodes, decay) if nodes[0]["eligible"] else 0.0
    mode = os.getenv("SOPO_REWARD_MODE", "best")
    if not nodes[0]["eligible"]:
        reward = float(immediate_reward)
    elif mode == "best":
        reward = max(float(node["static_value"]) for node in nodes if node["eligible"])
    else:
        reward = float(immediate_reward) + bonus
    sample.metadata.update(sopo_reward_mode=mode, sopo_trajectory=nodes, sopo_steps=steps, sopo_lambda=decay,
                           sopo_immediate_reward=immediate_reward, sopo_bonus=bonus)
    return reward


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
