"""Draft / Improve / Debug operator prompts, program database and single-step reward.

Prompt wording follows OpenMLE-ERL/RL/prompt_builder.py so the small experiment trains
the same operator interface as the full system.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from executor import Result
from tasks import Task

PUBLIC_SYSTEM = (
    "Respond with a short plan followed by ONE complete, runnable Python program in a single ```python code block. "
    "The program reads train.csv and test.csv from the current directory and writes submission.csv.")
PUBLIC_USER = "Return the full program, not a diff."

DRAFT_SYSTEM = ("You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, "
                "you need to come up with an excellent and creative plan for a solution and then implement this "
                "solution in Python.\n")
IMPROVE_SYSTEM = ("You are a Kaggle Grandmaster attending a high-stakes competition. In order to win this competition, "
                  "you need to come up with an excellent and creative plan for a solution and then implement this "
                  "solution in Python, that improves upon an existing solution to the task.\n")


@dataclass
class Node:
    id: int
    task: str
    op: str
    code: str
    response: str
    result: Result
    parent: int | None = None
    children: list[int] = field(default_factory=list)

    @property
    def utility(self) -> float:
        return self.result.utility if self.result.eligible else 0.0


def build_messages(op: str, task: Task, parent: Node | None = None) -> list[dict]:
    head = ("We will now provide a description of the task.\n\nTask Description:\n\n"
            f"{task.description}\n\nData Description:\n\n{task.data_description}\n\n")
    if op == "draft":
        return [{"role": "system", "content": DRAFT_SYSTEM + PUBLIC_SYSTEM},
                {"role": "user", "content": head + "Draft Guidance:\n\nPlease focus on proposing an advanced idea "
                 "addressing this task.\n\n" + PUBLIC_USER}]
    assert parent is not None
    guidance = (
        "Improve Guidance:\n\nBased on the previous attempt, its score, and execution output, please analyze what "
        "worked well and what didn't, then improve the solution.\n\n" if op == "improve" else
        "Debug Guidance:\n\nBased on the previous attempt and its execution output, identify the root cause of the "
        "failure and fix the code so it runs successfully and writes a valid submission. After it runs, improve the "
        "solution if possible.\n\n")
    return [{"role": "system", "content": IMPROVE_SYSTEM + PUBLIC_SYSTEM},
            {"role": "user", "content": head + "A previous solution was attempted with the following results:\n\n"
             f"Code:\n\n```python\n{parent.code}\n```\n\nFeedback:\n\n{parent.result.feedback(task.metric)}\n\n"
             + guidance + PUBLIC_USER}]


def immediate_reward(op: str, result: Result, parent: Node | None, bonus_coef: float = 0.5) -> float:
    """Single-step reward of the original OpenMLE RL (compute_mode_rewards, static mapping)."""
    if result.category == "empty":
        return -1.0
    if result.category == "hack":
        return -0.5
    if not result.eligible:
        return 0.0
    u = result.utility
    if op in ("improve", "debug") and parent is not None:
        if result.code.strip() == parent.code.strip():
            return 0.0
        return u + bonus_coef * max(0.0, u - parent.utility)
    return u


class ProgramDB:
    """Per-task programs; parent selection mirrors AIRAEvoSearch (improve: scored, debug: failed)."""

    def __init__(self, rng: random.Random):
        self.nodes: dict[int, Node] = {}
        self.by_task: dict[str, list[int]] = {}
        self.rng = rng
        self._next = 0

    def add(self, task: str, op: str, code: str, response: str, result: Result, parent: Node | None) -> Node:
        node = Node(self._next, task, op, code, response, result, parent.id if parent else None)
        self._next += 1
        self.nodes[node.id] = node
        self.by_task.setdefault(task, []).append(node.id)
        if parent is not None:
            parent.children.append(node.id)
        return node

    def programs(self, task: str) -> list[Node]:
        return [self.nodes[i] for i in self.by_task.get(task, [])]

    def best(self, task: str) -> Node | None:
        scored = [n for n in self.programs(task) if n.result.eligible]
        return max(scored, key=lambda n: n.utility) if scored else None

    def pick_improve(self, task: str, temperature: float = 0.2) -> Node | None:
        scored = [n for n in self.programs(task) if n.result.eligible]
        if not scored:
            return None
        # Softmax on utility with a visit penalty: exploit good programs, keep exploring.
        w = [math.exp((n.utility - 0.15 * len(n.children)) / temperature) for n in scored]
        return self.rng.choices(scored, weights=w, k=1)[0]

    def pick_debug(self, task: str) -> Node | None:
        failed = [n for n in self.programs(task) if n.result.category == "valid" and not n.result.eligible]
        fresh = [n for n in failed if not n.children] or failed
        return self.rng.choice(fresh) if fresh else None

    def select(self, task: str, probs: dict[str, float]) -> tuple[str, Node | None]:
        op = self.rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
        if op == "improve":
            parent = self.pick_improve(task)
            if parent is None:
                parent, op = self.pick_debug(task), "debug"
        elif op == "debug":
            parent = self.pick_debug(task)
            if parent is None:
                parent, op = self.pick_improve(task), "improve"
        else:
            parent = None
        if parent is None:
            op = "draft"
        return op, parent
