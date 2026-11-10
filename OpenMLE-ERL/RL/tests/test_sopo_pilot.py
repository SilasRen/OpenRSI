import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sopo_pilot import collect, seed_for, utility, load_states, report

class PilotTests(unittest.TestCase):
    def test_budget_replication_and_no_private_feedback(self):
        async def run():
            state=dict(state_id='s',task_id='t',messages=[{'role':'user','content':'operator'}],
                task_messages=[{'role':'user','content':'public task'}])
            calls=[]
            async def backend(state,messages,identity,seed):
                calls.append((messages,identity))
                return dict(response='candidate code',utility=.7,eligible=True,score=12345)
            first=await collect([state],8,1,'discover',42,backend)
            self.assertEqual(len(calls),24)
            prior={f's/{r["candidate"]}':r for r in first}
            calls.clear()
            repeated=await collect([state],8,3,'replicate',42,backend,prior)
            self.assertEqual(len(calls),48)
            self.assertEqual(len({seed_for(42,*i) for _,i in calls}),48)
            self.assertTrue(all('12345' not in json.dumps(m) for m,_ in calls))
            self.assertEqual(len(repeated),24)
        asyncio.run(run())

    def test_metric_direction(self):
        s=dict(score_min=0.,score_max=10.,higher_is_better=False)
        self.assertAlmostEqual(utility(2,s),.8)
        s['higher_is_better']=True
        self.assertAlmostEqual(utility(2,s),.2)

    def test_independent_report_selection(self):
        try: import matplotlib
        except ImportError: self.skipTest('matplotlib not installed')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            def row(c,initial,best):
                return dict(state_id='s',candidate=c,best=best,nodes=[dict(utility=initial,eligible=True)])
            (path/'discover.json').write_text(json.dumps([row(0,.8,.8),row(1,.5,.95)]))
            (path/'replicate.json').write_text(json.dumps([row(0,.8,.85),row(1,.5,.6)]))
            (path/'fresh.json').write_text(json.dumps([row(0,.4,.4),row(1,.5,.5)]))
            report(path)
            summary=json.loads((path/'summary.json').read_text())
            # Discovery winner loses on independent continuations: do not hide this.
            self.assertAlmostEqual(summary['independent_selection'][0]['heldout_gain'],-.25)
            self.assertTrue((path/'03_variance_repeatability.png').exists())

if __name__=='__main__': unittest.main()
