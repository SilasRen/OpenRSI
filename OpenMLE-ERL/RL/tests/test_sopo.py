import asyncio
import os
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prompt_builder  # noqa: F401  (load once; tests patch sys.modules)
from sopo import combine_reward, continuation_messages, extend_reward, lookahead_bonus, settings


def node(value, code, eligible=True, category='valid'):
    return dict(static_value=value, eligible=eligible, code=code, response=code, code_category=category,
                status_code=200 if eligible else 500, score=value if eligible else None,
                payload={'status': 'done', 'result': {'score': value, 'run_log': 'log-' + code}})


class SopoTests(unittest.TestCase):
    def test_discount_counts_only_new_records(self):
        nodes = [node(.7, 'a'), node(.6, 'b'), node(.95, 'c')]
        self.assertAlmostEqual(lookahead_bonus(nodes, .5), .25 * .25)
        self.assertEqual(lookahead_bonus(nodes, 0), 0)

    def test_chain_and_root_tokens_preserved(self):
        async def run():
            root = types.SimpleNamespace(prompt=[dict(role='system', content='sys'), dict(role='user', content='task')],
                tokens=[1, 2], response='a', metadata=dict(sopo_node=node(.7, 'a'),
                sopo_source_metadata={'uuid': 'task'}, sopo_sampling_params={'temperature': 1}, task_id='task'))
            before = deepcopy((root.tokens, root.response, root.prompt))
            prompts = []
            async def post(url, payload):
                prompts.append(payload['text'])
                return dict(text=['b', 'c'][len(prompts)-1], meta_info={'finish_reason': {'type': 'stop'}})
            async def evaluate(args, child):
                self.assertTrue(child.metadata['sopo_continuation'])
                self.assertNotIn('reward_group_id', child.metadata)
                child.metadata['sopo_node'] = node(.6 if child.response == 'b' else .95, child.response)
            fake = types.ModuleType('generate_mle')
            fake.Sample = types.SimpleNamespace
            fake.post = post
            fake.reward_func = evaluate
            fake.GenerateState = lambda args: types.SimpleNamespace(semaphore=asyncio.Semaphore(1), tokenizer=types.SimpleNamespace(
                apply_chat_template=lambda messages, **kw: messages))
            with patch.dict(sys.modules, generate_mle=fake):
                reward = await extend_reward(types.SimpleNamespace(sglang_router_ip='localhost', sglang_router_port=1), root, .7)
            self.assertEqual(len(prompts), 2)
            # Second step is the repository's Improve operator applied to the first continuation.
            self.assertIn('A previous solution was attempted', prompts[1][1]['content'])
            self.assertIn('Code:\n\nb', prompts[1][1]['content'])
            self.assertIn('**Score**: 0.6', prompts[1][1]['content'])
            self.assertIn('Kaggle Grandmaster', prompts[0][0]['content'])
            self.assertNotIn('payload', root.metadata['sopo_trajectory'][1])
            self.assertAlmostEqual(reward, .95)
            self.assertEqual(before, (root.tokens, root.response, root.prompt))
            self.assertEqual(len(root.metadata['sopo_trajectory']), 3)
        with patch.dict(os.environ, SOPO_STEPS='2', SOPO_LAMBDA='1', SOPO_REWARD_MODE='best'):
            asyncio.run(run())

    def test_reward_modes(self):
        nodes = [node(.7, 'a'), node(.6, 'b'), node(.95, 'c')]
        self.assertAlmostEqual(combine_reward(nodes, .7, 'best', 1)[0], .95)
        self.assertAlmostEqual(combine_reward(nodes, .7, 'credit', .9)[0], .7 + .81 * .25)
        # lambda=0 in credit mode is exactly the original single-step reward.
        self.assertAlmostEqual(combine_reward(nodes, .7, 'credit', 0)[0], .7)
        # A crashed root that Debug repairs earns credit from the repaired branch.
        crashed = [node(0, 'a', eligible=False), node(.8, 'b'), node(.9, 'c')]
        self.assertAlmostEqual(combine_reward(crashed, 0, 'best', 1)[0], .9)
        self.assertEqual(combine_reward([node(0, 'a', eligible=False)] * 3, 0, 'best', 1)[0], 0)

    def test_crashed_root_uses_debug_operator(self):
        mode, messages = continuation_messages({'task_description': 'T'},
            [dict(role='system', content='s'), dict(role='user', content='u')], node(0, 'x', eligible=False))
        self.assertEqual(mode, 'debug')
        mode, _ = continuation_messages({}, [dict(role='system', content='s'), dict(role='user', content='u')], node(.5, 'x'))
        self.assertEqual(mode, 'improve')

    def test_invalid_root_skips_lookahead(self):
        root = types.SimpleNamespace(metadata=dict(sopo_node=node(0, 'a', eligible=False, category='hack')))
        with patch.dict(os.environ, SOPO_STEPS='2', SOPO_REWARD_MODE='best'):
            # No generate_mle import is needed: hacked roots never call the model or sandbox.
            self.assertEqual(asyncio.run(extend_reward(None, root, -.5)), -.5)
        self.assertTrue(root.metadata['sopo_lookahead_skipped'])
        self.assertEqual(len(root.metadata['sopo_trajectory']), 1)

    def test_disabled(self):
        with patch.dict(os.environ, SOPO_STEPS='0'):
            self.assertEqual(asyncio.run(extend_reward(None, types.SimpleNamespace(metadata={}), .3)), .3)

    def test_invalid_configuration(self):
        with patch.dict(os.environ, SOPO_STEPS='2', SOPO_LAMBDA='nan'):
            with self.assertRaises(ValueError): settings()

if __name__ == '__main__': unittest.main()
