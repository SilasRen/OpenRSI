import asyncio
import os
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sopo import extend_reward, lookahead_bonus, settings


def node(value, code):
    return dict(static_value=value, eligible=True, code=code, response=code)


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
            self.assertTrue(prompts[1][1]['content'].endswith('b'))
            self.assertAlmostEqual(reward, .95)
            self.assertEqual(before, (root.tokens, root.response, root.prompt))
            self.assertEqual(len(root.metadata['sopo_trajectory']), 3)
        with patch.dict(os.environ, SOPO_STEPS='2', SOPO_LAMBDA='1', SOPO_REWARD_MODE='best'):
            asyncio.run(run())

    def test_disabled(self):
        with patch.dict(os.environ, SOPO_STEPS='0'):
            self.assertEqual(asyncio.run(extend_reward(None, types.SimpleNamespace(metadata={}), .3)), .3)

    def test_invalid_configuration(self):
        with patch.dict(os.environ, SOPO_STEPS='2', SOPO_LAMBDA='nan'):
            with self.assertRaises(ValueError): settings()

if __name__ == '__main__': unittest.main()
