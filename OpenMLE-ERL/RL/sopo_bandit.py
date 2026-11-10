"""Process-local, thread-safe curriculum over (task, parents, operator) states."""
import copy
import hashlib
import json
import math
import random
import threading
from collections import OrderedDict


def credit_variance(entries):
    """Spread of SOPO credit among siblings whose own program ran and scored.

    Crash/hack penalties would otherwise dominate the variance (8 crashes at -0.5 and
    8 runs at 0.8 look "useful" but only teach "don't crash"). States with fewer than
    two scored siblings carry no information about which *method* leads further.
    """
    values = [v for v, ok in entries if ok]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


class StateBandit:
    def __init__(self):
        self.lock = threading.RLock()
        self.arms = OrderedDict()
        self.pending = OrderedDict()
        self.clock = 0
        self.rng = random.Random(42)

    def observe(self, sample, reward, max_states=512):
        m = sample.metadata
        selection = m['sopo_state_selection']
        identity = [m.get('task_id'), selection['mode'], selection['messages'],
                    [(p or {}).get('code') for p in (selection['parent'], selection['secondary'])]]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        gid = m['reward_group_id']
        with self.lock:
            pending = self.pending.setdefault(gid, {'key': key, 'values': {}, 'expected': int(m['reward_group_expected_count'])})
            if pending['key'] != key:
                raise ValueError('SOPO siblings must share task, parents and operator')
            trajectory = m.get('sopo_trajectory') or [{}]
            eligible = bool(trajectory[0].get('eligible'))
            pending['values'][int(m['reward_group_member_index'])] = (float(reward), eligible)
            # Incomplete failed groups never update the curriculum; bound their memory.
            while len(self.pending) > 4096:
                self.pending.popitem(last=False)
            if len(pending['values']) != pending['expected']:
                return
            entries = list(pending['values'].values())
            del self.pending[gid]
            if len(entries) < 2 or not all(math.isfinite(v) for v, _ in entries):
                return
            variance = credit_variance(entries)
            self.clock += 1
            previous = self.arms.get(key)
            priority = variance if previous is None else .8 * previous['priority'] + .2 * variance
            source = copy.deepcopy(m['sopo_source_metadata'])
            for field in list(source):
                if field.startswith(('sopo_', 'reward_group_')):
                    source.pop(field)
            source['sopo_replay_selection'] = copy.deepcopy(selection)
            self.arms[key] = dict(priority=priority, updated=self.clock,
                                  prompt=copy.deepcopy(sample.prompt), metadata=source)
            self.arms.move_to_end(key)
            while len(self.arms) > max_states:
                self.arms.popitem(last=False)

    def draw(self, exploration=.3, decay=.99):
        with self.lock:
            if not self.arms or self.rng.random() < exploration:
                return None
            arms = list(self.arms.values())
            weights = [a['priority'] * decay**(self.clock-a['updated']) for a in arms]
            if sum(weights) <= 1e-12:
                return None
            return copy.deepcopy(self.rng.choices(arms, weights=weights, k=1)[0])

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(dict(arms=list(self.arms.items()), clock=self.clock, rng=self.rng.getstate()))

    def restore(self, state):
        with self.lock:
            self.arms = OrderedDict(copy.deepcopy(state['arms']))
            self.clock = state['clock']
            self.rng.setstate(state['rng'])
            self.pending.clear()


bandit = StateBandit()
