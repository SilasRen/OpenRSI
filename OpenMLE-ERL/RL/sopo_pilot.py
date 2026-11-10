"""Frozen-model SOPO pilot. No SLIME, Ray, gradients, or optimizer required."""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


def seed_for(seed, *parts):
    return int(hashlib.sha256(json.dumps([seed, *parts]).encode()).hexdigest()[:8], 16) % (2**31)


def load_states(path, limit):
    states = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()][:limit]
    ids = set()
    task_bounds = {}
    for state in states:
        if state['state_id'] in ids:
            raise ValueError('state_id must be unique')
        ids.add(state['state_id'])
        if not isinstance(state['messages'], list) or not state['messages']:
            raise ValueError('Each state needs operator messages')
        if not isinstance(state.get('task_messages'), list) or not state['task_messages']:
            raise ValueError('task_messages must contain the original public task, without parent feedback')
        if not state.get('data_dir') or not state.get('task_id'):
            raise ValueError('Each state needs task_id and sandbox data_dir')
        low, high = float(state['score_min']), float(state['score_max'])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError('Provide finite fixed score_min < score_max')
        state['score_min'], state['score_max'] = low, high
        bounds = (low, high, state['higher_is_better'])
        if state['task_id'] in task_bounds and task_bounds[state['task_id']] != bounds:
            raise ValueError('States of the same task must use identical metric bounds and direction')
        task_bounds[state['task_id']] = bounds
        if not isinstance(state['higher_is_better'], bool):
            raise ValueError('higher_is_better must be boolean')
    if not states:
        raise ValueError('No states supplied')
    return states


def utility(score, state):
    value = (float(score)-state['score_min'])/(state['score_max']-state['score_min'])
    return min(1., max(0., value if state['higher_is_better'] else 1-value))


async def collect(states, candidates, repeats, phase, seed, run_node, previous=None):
    """Each repeat has independent continuation seeds; replicate reuses discovery roots."""
    async def branch(state, candidate, repeat):
        sid = state['state_id']
        key = f'{sid}/{candidate}'
        if phase == 'replicate':
            root = previous[key]['nodes'][0]
        else:
            root = await run_node(state, state['messages'], [phase, sid, candidate, repeat, 0], seed)
        nodes = [root]
        for depth in (1, 2):
            messages = list(state['task_messages']) + [{'role':'user', 'content':
                'Improve this candidate; repair errors if necessary. Return the entire Python program '
                'and follow the original task and validation requirements.\n\n' + nodes[-1]['response']}]
            nodes.append(await run_node(state, messages, [phase, sid, candidate, repeat, depth], seed))
        return dict(state_id=sid, task_id=state['task_id'], candidate=candidate,
                    repeat=repeat, phase=phase, nodes=nodes,
                    best=max(n['utility'] for n in nodes))
    return await asyncio.gather(*(branch(s, c, r) for s in states
                                 for c in range(candidates) for r in range(repeats)))


class Backend:
    def __init__(self, args, directory):
        import httpx
        from reward_func_utils import extract_code, get_sandbox_result, hack_check_async
        self.args, self.directory = args, directory
        self.extract, self.evaluate, self.judge = extract_code, get_sandbox_result, hack_check_async
        self.httpx = httpx
        self.sem = asyncio.Semaphore(args.concurrency)
        self.calls = 0

    async def __call__(self, state, messages, identity, seed):
        cache = self.directory / (hashlib.sha256(json.dumps(identity).encode()).hexdigest()+'.json')
        if cache.exists():
            return json.loads(cache.read_text())
        async with self.sem:
            started = time.monotonic()
            async with self.httpx.AsyncClient(timeout=self.args.request_timeout) as client:
                response = await client.post(self.args.model_url.rstrip('/')+'/chat/completions',
                    headers={'Authorization': 'Bearer '+os.getenv('MODEL_API_KEY', 'EMPTY')},
                    json=dict(model=self.args.model, messages=messages, max_tokens=self.args.max_tokens,
                              temperature=self.args.temperature, seed=seed_for(seed, *identity)))
                response.raise_for_status()
                choice = response.json()['choices'][0]
                text = choice['message']['content'] or ''
            code = self.extract(text)
            valid, category, reason = (False, 'empty', '')
            if code:
                valid, category, reason = await self.judge(code, require_holdout_validation=True)
                # Existing judge helper can fail open; pilot must not treat that as valid evidence.
                if any(x in reason.lower() for x in ('failed', 'error', 'not available', 'unavailable')):
                    raise RuntimeError('Judge unavailable; resume after restoring judge service')
            score, status = None, None
            if valid:
                async with self.httpx.AsyncClient(base_url=os.environ['GPU_BASE_URL'], timeout=100) as client:
                    status, payload = await self.evaluate(client, code, state['data_dir'],
                        job_timeout=self.args.job_timeout, wait_timeout=self.args.job_timeout+300)
                if status != 200:
                    raise RuntimeError(f'Sandbox infrastructure status {status}; resume after checking service')
                score = (payload.get('result') or {}).get('score')
            eligible = valid and score is not None and math.isfinite(float(score))
            node = dict(response=text, score=score, eligible=eligible, category=category,
                        finish_reason=choice.get('finish_reason'), utility=utility(score,state) if eligible else 0.,
                        wall_seconds=time.monotonic()-started, seed=seed_for(seed,*identity))
            temp = cache.with_suffix('.tmp')
            temp.write_text(json.dumps(node, ensure_ascii=False)); temp.replace(cache)
            self.calls += 1
            print(f'Completed node {identity}; valid={eligible}; elapsed={node["wall_seconds"]:.1f}s', flush=True)
            return node


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    discovery = json.loads((directory/'discover.json').read_text())
    rep_path, fresh_path = directory/'replicate.json', directory/'fresh.json'
    replicated = json.loads(rep_path.read_text()) if rep_path.exists() else []
    fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else []
    grouped = {}
    for row in discovery: grouped.setdefault(row['state_id'], []).append(row)
    summary = dict(states=len(grouped), branches=len(discovery),
        valid_node_fraction=statistics.mean(n['eligible'] for r in discovery for n in r['nodes']),
        note='Frozen-policy pilot, not an RL result. Discovery maxima are optimistic. '
             'Replication shares root evaluations; fresh groups use new candidates. Compare within tasks.')
    fig, ax = plt.subplots()
    for sid, rows in grouped.items():
        ax.scatter([r['nodes'][0]['utility'] for r in rows], [r['best'] for r in rows], label=sid)
    ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Initial utility',ylabel='Best of three (discovery)')
    fig.savefig(directory/'01_initial_vs_future.png',dpi=160); plt.close(fig)
    gains=[]
    if replicated:
        fig, ax=plt.subplots()
        for sid, rows in grouped.items():
            eligible=[r for r in rows if r['nodes'][0]['eligible']]
            if not eligible: continue
            immediate=max(eligible,key=lambda r:r['nodes'][0]['utility'])['candidate']
            future=max(eligible,key=lambda r:r['best'])['candidate']
            outcomes={c:[r['best'] for r in replicated if r['state_id']==sid and r['candidate']==c]
                      for c in (immediate,future)}
            if not all(outcomes.values()): continue
            x,y=statistics.mean(outcomes[immediate]),statistics.mean(outcomes[future])
            ax.scatter(x,y); gains.append(dict(state_id=sid,immediate_candidate=immediate,
                lookahead_candidate=future,heldout_gain=y-x))
        ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Immediate-selected: replicated utility',
                                           ylabel='Lookahead-selected: replicated utility')
        fig.savefig(directory/'02_independent_selection.png',dpi=160); plt.close(fig)
    summary['independent_selection']=gains
    variance=[]
    if fresh:
        fig,ax=plt.subplots()
        for sid,rows in grouped.items():
            values=[r['best'] for r in fresh if r['state_id']==sid]
            if len(values)<2: continue
            x,y=statistics.pvariance(r['best'] for r in rows),statistics.pvariance(values)
            variance.append(dict(state_id=sid,discovery_variance=x,fresh_variance=y));ax.scatter(x,y)
        ax.set(xlabel='Discovery group variance',ylabel='Fresh-candidate group variance')
        fig.savefig(directory/'03_variance_repeatability.png',dpi=160);plt.close(fig)
    summary['variance_repeatability']=variance
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (directory/'report.md').write_text('# Frozen-model SOPO pilot\n\n'+summary['note']+
        '\n\n![Initial vs future](01_initial_vs_future.png)\n\n'+
        ('![Independent selection](02_independent_selection.png)\n\n' if replicated else '')+
        ('![Variance repeatability](03_variance_repeatability.png)\n\n' if fresh else '')+
        f'Valid node fraction: {summary["valid_node_fraction"]:.3f}. See summary.json for per-state results.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['discover','replicate','fresh','report'],default='discover')
    p.add_argument('--states',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-states',type=int,default=2)
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3,help='Independent continuation repeats for replicate phase')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--model',default='Qwen/Qwen3-8B')
    p.add_argument('--model-url',default='http://127.0.0.1:8000/v1')
    p.add_argument('--max-tokens',type=int,default=8192)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--job-timeout',type=int,default=300)
    p.add_argument('--request-timeout',type=int,default=600)
    p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.phase=='report': report(args.output); return
    if not args.states: p.error('--states is required')
    if min(args.num_states,args.candidates,args.repeats,args.concurrency,args.max_tokens,args.job_timeout)<=0:
        p.error('Counts, concurrency and budgets must be positive')
    if args.candidates<2: p.error('At least two candidates are needed to estimate variance')
    states=load_states(args.states,args.num_states)
    config={k:v for k,v in vars(args).items() if k not in ('phase','states','output','plan_only','repeats')}
    config['states']=states
    repeats=args.repeats if args.phase=='replicate' else 1
    count=len(states)*args.candidates*(2 if args.phase=='replicate' else 3)*repeats
    print(f'{args.phase}: at most {count} new model generations and sandbox evaluations; no optimizer updates')
    if args.plan_only: return
    if os.getenv('USE_PROXY_SANDBOX', '0') != '0':
        p.error('USE_PROXY_SANDBOX must be disabled; random proxy scores are not experimental evidence')
    required = ('GPU_BASE_URL', 'SANDBOX_API_KEY', 'GPT_BASE_URL', 'GPT_API_KEY')
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        p.error('Cannot start a real pilot: missing ' + ', '.join(missing) +
                '. This backend requires a reachable OpenMLE Sandbox API and the code-review judge. '
                'The HPC node currently has Apptainer but no configured sandbox controller/worker. '
                'Do not submit the pilot until an administrator-provided sandbox endpoint and judge credentials are configured.')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=args.output/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=config: raise ValueError('Run configuration changed; use a new output directory')
    else: manifest.write_text(json.dumps(config,indent=2))
    previous=None
    if args.phase=='replicate':
        rows=json.loads((args.output/'discover.json').read_text())
        previous={f'{r["state_id"]}/{r["candidate"]}':r for r in rows}
    cache=args.output/'nodes';cache.mkdir(exist_ok=True)
    rows=asyncio.run(collect(states,args.candidates,repeats,args.phase,args.seed,Backend(args,cache),previous))
    (args.output/f'{args.phase}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
