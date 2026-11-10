"""Curriculum data source shared by synchronous and asynchronous rollouts."""
import copy
import os
from slime.rollout.data_source import RolloutDataSourceWithBuffer
from slime.utils.types import Sample
from sopo_bandit import bandit


class SopoDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args):
        super().__init__(args)
        if int(os.getenv('SOPO_STEPS', '2')) < 1:
            raise ValueError('SOPO curriculum requires SOPO_STEPS >= 1')
        self.exploration = float(os.getenv('SOPO_EXPLORATION', '.3'))
        self.decay = float(os.getenv('SOPO_PRIORITY_DECAY', '.99'))
        if not (0 < self.exploration <= 1 and 0 < self.decay <= 1):
            raise ValueError('SOPO exploration and priority decay must be in (0, 1]')

    def get_samples(self, num_samples):
        groups = self._get_samples_from_buffer(num_samples)
        while len(groups) < num_samples:
            arm = bandit.draw(self.exploration, self.decay)
            if arm is None:
                groups.extend(super().get_samples(1))
                continue
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = Sample(prompt=copy.deepcopy(arm['prompt']), metadata=copy.deepcopy(arm['metadata']))
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def save(self, rollout_id):
        self.metadata['sopo_bandit'] = bandit.snapshot()
        super().save(rollout_id)

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if 'sopo_bandit' in self.metadata:
            bandit.restore(self.metadata['sopo_bandit'])
