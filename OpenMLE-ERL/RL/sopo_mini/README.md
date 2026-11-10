# sopo_mini：离线小规模验证 SOPO 是否有效

目的只有一个：在**一个集群节点、无 Docker、无沙盒服务、无任何外部 API** 的条件下，回答
“用 SOPO 奖励（看分支往后走得多远）训练出来的算子，在固定搜索预算下是否比原始单步奖励更好”。

它不依赖 slime / SGLang / Ray / Megatron / OpenMLE 沙盒 / o3-mini 代码审查，全部替换为本地等价物：

| 原系统 | 这里的替代 |
|---|---|
| MLE-bench 任务 + Docker 沙盒 | `tasks.py` 生成的表格任务（sklearn 合成 + 自带数据集），`executor.py` 本地子进程执行（rlimit 限内存/CPU/文件大小、超时杀进程组、单线程 BLAS） |
| API 代码审查（hack_check） | 静态规则：禁止 socket / requests / subprocess / os.system / ctypes / 读 `_private` 或 `../` 等，命中记为 hack（-0.5） |
| 排行榜分数 | utility = clip((score − 傻瓜基线) / (GBM 参考 − 傻瓜基线), 0, 1.5)，跨任务可比 |
| Qwen3 大模型 + GSPO 多机 | Qwen2.5-Coder-1.5B-Instruct，单卡 HF 训练 + 生成，on-policy 一步更新（重要性比 = 1，退化为组归一化策略梯度） |

SOPO 的核心逻辑与主线 `sopo.py` 相同：能跑通的兄弟节点做 Improve→Improve，报错的不展开（只拿即时奖励）；
`best` 模式奖励 = 分支上最高 utility，`credit` 模式 = 即时奖励 + Σλ^d·新纪录增量；课程按“跑通兄弟”的奖励方差采样状态。

## 实验臂

| arm | 含义 |
|---|---|
| `control` | 原始单步奖励（`--sopo-steps 0`） |
| `control_3x` | 同样奖励，3 倍 RL 步数 ≈ 与 SOPO 相同的程序执行次数（排除“只是算力更多”） |
| `sopo` | SOPO 简单版：best 模式，K=2 |
| `sopo_cur` | sopo + 方差课程 |
| `sopo_credit`（可选） | credit 模式，λ=0.9，`ARMS="... sopo_credit"` 加入 |

另外未训练的 base 模型也跑同样的评测。每个 arm 默认 2 个训练种子，评测在**留出任务**（含训练中没见过的任务类型和真实数据集）上、每任务 3 个搜索种子，所有 arm 用完全相同的 (任务, 种子) 配对比较。

## 准备（集群通常不能联网）

```bash
pip install -r requirements.txt          # 或在有网的机器上建好 conda/venv/Apptainer 镜像再拷过去
# 在有网的机器上下载模型，再拷到集群：
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir /path/models/Qwen2.5-Coder-1.5B-Instruct
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct --local-dir /path/models/Qwen2.5-Coder-0.5B-Instruct
export HF_HUB_OFFLINE=1
```

任务数据由 sklearn 本地生成/自带，不需要下载。

## 运行

```bash
cd OpenMLE-ERL/RL/sopo_mini

# 0) 无 GPU 也能跑的流水线自检（假模型，几分钟）
FAKE=1 OUT=/tmp/sopo_fake bash run_experiment.sh

# 1) 冒烟（0.5B，少量步数，约 1-2 小时），先确认集群环境没问题
QUICK=1 MODEL_QUICK=/path/models/Qwen2.5-Coder-0.5B-Instruct OUT=$PWD/runs_quick bash run_experiment.sh

# 2) 正式实验（8 卡：4 arm × 2 种子同时跑，每卡一个任务）
MODEL=/path/models/Qwen2.5-Coder-1.5B-Instruct OUT=$PWD/runs bash run_experiment.sh
# 或 Slurm：
sbatch --gres=gpu:8 --cpus-per-task=64 --mem=400G --time=24:00:00 \
  --export=ALL,MODEL=/path/models/Qwen2.5-Coder-1.5B-Instruct,OUT=$PWD/runs run_experiment.sh
```

脚本可断点续跑：已有 `final/` 的训练和已有结果的评测会跳过。常用环境变量：
`STEPS`(60) `SEEDS`(2) `N_GROUPS`(4) `GROUP_SIZE`(8) `LR`(1e-6) `MAX_NEW_TOKENS`(1536) `TIMEOUT`(90)
`EVAL_BUDGET`(16) `EVAL_SEEDS`(3) `ARMS` `TRAIN_PER_KIND`(10) `EVAL_PER_KIND`(4) `PY`。

显存：1.5B 用 fp32 主权重 + AdamW + 梯度检查点，单卡约 25-35GB，60GB 卡足够。
CPU：执行器并行数 = CPU 数 / 同时运行的任务数，建议节点 ≥ 32 核。

## 怎么读结论

结果在 `$OUT/report/summary.md`（和 `summary.json`、`search_curves.png`）：

- **best@k**：搜索执行 k 个程序后找到的最高 utility（各任务/种子平均）。看 `best@16`。
- **Δbest@final vs control**：配对差值 + 按任务聚类 bootstrap 的 95% CI，以及 W/T/L。
- **Δ(gain after drafts)**：扣掉开头几个 draft 后搜索带来的提升差，直接对应 SOPO 想改善的“作为搜索算子的能力”。
- 判定：`sopo` 相对 `control` **且** 相对 `control_3x` 的 CI 都在 0 以上，才算“有效且不是因为算力多”。
  只有 best@1/4 提升而 best@16 相同，说明它让搜索更快到达，但没有到达更好的程序。

训练过程看 `$OUT/train/<arm>_s<seed>/train_log.jsonl`（每步的 valid 率、奖励、执行次数）和 `$OUT/logs/`。

## 注意

- 隔离弱于 Docker：模型生成的代码以当前用户在本地子进程执行，靠静态黑名单 + rlimit + 临时目录。
  请在计算节点、无重要文件的账号/目录下运行；如需更强隔离，可在 Apptainer 容器里运行整个脚本（`apptainer exec --containall --nv img.sif bash run_experiment.sh`）。
- 1.5B 模型首轮 valid 率可能偏低（30-60%），这是正常的；若长时间接近 0，先看 `logs/train_*.log` 里的报错（多半是模型路径或依赖）。
- 规模较小时可能得到“CI 包含 0”的结论，此时可加 `SEEDS=3`、`EVAL_SEEDS=5` 或 `STEPS=100` 提高统计功效。
- 测试：`python -m unittest tests/test_sopo_mini.py`（在 `OpenMLE-ERL/RL` 下）。
