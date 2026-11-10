import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sopo_bandit import StateBandit


def sample(task, gid, member):
    return SimpleNamespace(prompt=[{'role':'user','content':task}], metadata=dict(
        task_id=task, reward_group_id=gid, reward_group_expected_count=2,
        reward_group_member_index=member, sopo_source_metadata={'uuid':task},
        sopo_state_selection={'mode':'draft','messages':[{'role':'user','content':task}], 'parent':None,'secondary':None}))


class BanditTests(unittest.TestCase):
    def test_incomplete_groups_do_not_change_weights(self):
        b=StateBandit()
        b.observe(sample('a','g',0), 0)
        self.assertEqual(len(b.arms),0)
        b.observe(sample('a','g',1),1)
        self.assertEqual(next(iter(b.arms.values()))['priority'],.25)

    def test_high_variance_state_is_replayed_more(self):
        b=StateBandit()
        for task, scores in [('flat',[.5,.5]),('useful',[0,1])]:
            for i, score in enumerate(scores): b.observe(sample(task,task,i), score)
        results=[b.draw(exploration=.2) for _ in range(1000)]
        self.assertGreater(sum(r is None for r in results),100)
        replay=[r for r in results if r is not None]
        self.assertTrue(all(r['metadata']['uuid']=='useful' for r in replay))
        self.assertNotIn('reward_group_id',replay[0]['metadata'])
        self.assertIn('sopo_replay_selection',replay[0]['metadata'])

    def test_checkpoint_reproduces_draws(self):
        a=StateBandit()
        for i,v in enumerate([0,1]): a.observe(sample('x','g',i),v)
        b=StateBandit(); b.restore(a.snapshot())
        self.assertEqual([a.draw() for _ in range(20)], [b.draw() for _ in range(20)])

    def test_mixed_states_rejected(self):
        b=StateBandit(); b.observe(sample('a','g',0),0)
        with self.assertRaises(ValueError): b.observe(sample('b','g',1),1)

if __name__=='__main__': unittest.main()
