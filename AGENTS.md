# AGENTS.md

## 项目概况

- 本仓库是 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的个人 fork（`Vincent-Huang-2000/minimind`），用于学习和实践从零训练小型语言模型 MiniMind（主线约 64M 参数）。
- 覆盖完整训练链路：Pretrain → SFT → LoRA / DPO / PPO / GRPO / 蒸馏 / Agentic RL，核心代码均为 PyTorch 原生实现。
- 个人改动在 `custom` 分支上，`master` 跟随上游。

## 必须遵守的规则

- **不在本地训练。** 所有训练只在 Google Colab 上进行；本地（Windows 笔记本）只用来阅读、修改和提交代码，最多做语法 / import 检查或极小样例的冒烟测试。
- **不在本地下载完整数据集。** 只想看数据结构时用 `scripts/peek_dataset.py`。
- **不要自行运行收费的 API 调用**：`augment_probe.py` 的 `concepts` / `questions` / `answers` 阶段（包括 `--stage all`），`topic_map.py` 的 `label` 阶段。
- **不提交数据文件。** `dataset/*.jsonl` 和各工具在 `dataset/` 下的输出目录都已被 `.gitignore` 忽略。
- **改保存 / 续训逻辑要格外小心**：Colab 上的长时间训练依赖 `--from_resume 1`。
- 给训练或性能优化建议时以 Colab 为准，不针对本地 GPU 或 Windows 做适配。

## Colab 训练环境

- 单卡环境，DDP / 多卡相关的优化优先级很低。
- 运行时会断开、`/content` 下的文件会被清空：训练产物应保存到 Google Drive，长时间训练用 `--from_resume 1` 续训。
- GPU 型号不固定：T4 不原生支持 bf16，用 `--dtype float16`（脚本会自动启用 GradScaler）；L4 / A100 可用默认的 `bfloat16`。

## 目录与运行位置

| 目录 | 内容 | 说明 |
|---|---|---|
| `trainer/` | 训练脚本 | 必须在 `trainer/` 下运行，脚本里的 `../dataset`、`../out`、`../checkpoints` 都是相对路径 |
| `dataset/` | 训练数据及各工具的输出 | 用 `scripts/download_dataset.py` 下载；不要提交 |
| `eval_data/` | 评测题库 | 与训练数据分开，不在 `.gitignore` 范围内，可以提交 |
| `out/` | 推理权重、训练曲线 | 可用 `--save_dir` 修改，Colab 上指向 Drive |
| `checkpoints/` | 续训文件 | 路径在脚本中写死，要放到 Drive 需建软链接或改代码 |
| `scripts/` | 数据、评测、分析工具 | 在仓库根目录运行 |
| `docs/` | 各功能的详细文档 | 改动对应功能时同步更新 |

- `eval_llm.py` 在仓库根目录运行。
- 本地 `.venv` 装有 numpy、transformers、openai，没有 torch，用来冒烟不依赖 torch 的脚本。

## 工具速查

| 用途 | 命令 | 运行位置 | 详细文档 |
|---|---|---|---|
| 下载数据集 | `python scripts/download_dataset.py` | 根目录 | — |
| 查看数据结构 | `python scripts/peek_dataset.py` | 根目录 | — |
| 选 `--max_seq_len` | `python scripts/token_length_stats.py --data dataset/sft_t2t_mini.jsonl --current 768` | 根目录 | `docs/token_length_stats.md` |
| 预训练装箱打包 | `python train_pretrain.py --pack 1 --pack_cache_dir <Drive 目录>` | `trainer/` | `docs/pretrain_packing.md` |
| 训练曲线出图 | `python ../scripts/plot_metrics.py ../out/pretrain_metrics.jsonl` | `trainer/` | `docs/training_metrics.md` |
| 对话评测 | `python eval_llm.py --weight full_sft` | 根目录 | — |
| 题库批量作答 | `python scripts/run_probe.py --weight full_sft` | 根目录 | — |
| 题库扩增成 SFT 数据 | `python scripts/augment_probe.py --stage all`（收费） | 根目录 | `docs/augment_probe.md` |
| 数据主题分布 | `python scripts/topic_map.py --stage all --data dataset/sft_t2t_mini.jsonl --work_dir <Drive 目录>` | 根目录（Colab） | `docs/topic_map.md` |
| 查看主题的完整样本 | `python scripts/view_topic_samples.py --work_dir <topic_map 输出目录> --topics 129,494` | 根目录 | 脚本开头的说明 |

## 各工具要点

### 数据下载与查看

- `download_dataset.py` 默认下载 mini 组合，`--list` 查看全部。
- `peek_dataset.py`（可加 `--files`、`-n`）从 HuggingFace 流式读取每个文件的前 N 行（默认 20），存到 `dataset/peek/` 并打印字段和首条样本。

### token 长度分布 `scripts/token_length_stats.py`

- 只需 `transformers`，不需要 torch，本地或 Colab 都能跑；可一次传多个文件，`--limit` 冒烟。
- 按训练时的拼接方式计算长度，字段自动识别；PPO / GRPO / Agent RL 数据要加 `--type prompt`，只统计 prompt。
- 训练数据集把每条样本补齐到 `max_seq_len` 且不拼接，所以输出里的 padding 占比基本就是浪费掉的算力。

### 预训练装箱打包 `--pack 1`

- 默认 `--pack 0`，行为不变；只影响预训练。
- 多篇完整文档共用一行，超长文档按 `max_seq_len` 切段（不再截断丢弃）；没有文档隔离掩码，同一行的文档能互相看到。
- 首次运行分词并缓存到 `--pack_cache_dir`（默认数据文件同目录的 `pack_cache/`），Colab 上应放 Drive；数据、`max_seq_len`、`seed`、tokenizer 不变时直接复用。
- **续训时不要改 `--pack` / `--max_seq_len` / `--data_path` / `--seed`**，否则断点步数对应到别的数据，且不会报错。
- 打包核心 `dataset/pretrain_packing.py` 只依赖 numpy 和 tokenizer，本地可用 `.venv` 冒烟。

### 训练曲线

- 各训练脚本在每个 `log_interval` 把指标追加写入 `{save_dir}/{save_weight}_metrics.jsonl`（LoRA 用 `lora_name`），训练结束自动生成同名 `.png`（`trainer_utils.MetricsLogger`）。
- 依赖 matplotlib（Colab 自带），缺失时只告警，不影响训练。
- 新训练覆盖旧文件；`--from_resume 1` 保留断点之前的记录，丢弃断点之后会被重跑的记录。
- Colab 里显示：`from IPython.display import Image; Image('../out/pretrain_metrics.png')`。

### 评测与题库作答

- 题库放在 `eval_data/`，如知识探针 `small_llm_knowledge_probe_142.jsonl`（`run_probe.py` 和 `augment_probe.py` 的默认题库，换题库用 `--data`）。
- `run_probe.py` 每题追加写入 `{save_dir}/{weight}_{题库名}_answers.jsonl`（原字段 + `answer` / `gen_tokens` / `truncated`），断连后加 `--resume 1` 续跑。

### 题库扩增成 SFT 数据 `scripts/augment_probe.py`

- 五个阶段 `concepts` → `merge` → `questions` → `answers` → `build`，可用 `--stage` 单独跑；`merge` 和 `build` 不调 API。
- API key 读环境变量 `DEEPSEEK_API_KEY`，也可写在根目录 `.env`（模板 `.env.example`，`.env` 已被忽略）。
- 输出在 `dataset/probe_aug_YYYYMMDD/`：新跑用当天日期，同一天重跑会覆盖；其余阶段和 `--resume 1` 接着用最近一次的目录。
- 重跑 `merge` 会覆盖人工改过的 `2_concepts.jsonl`。
- `--max_seq_len` 要和训练时一致（`train_full_sft.py` 默认 768）。
- 不训练，本地或 Colab 都能跑；本地需 `pip install openai==1.59.6 transformers==4.57.6 jinja2==3.1.2`，不需要 torch。

### 数据主题分布 `scripts/topic_map.py`

- 用途：分析预训练 / SFT 数据的整体构成，回答"有哪些主题、各占多少"；不统计具体概念的出现次数（概念覆盖核对是另一个计划中的脚本，见 `docs/topic_map.md` 开头）。
- 向量化需要 GPU，在 Colab 上跑；本地只用 `--limit` 冒烟。
- 流程：`embed`（sentence-transformers，默认 `intfloat/multilingual-e5-small`，支持 `--resume 1`）→ `cluster`（球面 K-Means，`--k` 个主题、`--k_top` 个大类）→ `keywords`（jieba + c-TF-IDF）→ `report`（`report.md`、`topics.csv`）。
- `label` 调 DeepSeek 起中文名，收费，不在 `all` 里。
- 默认输出到 `dataset/topic_map/{数据文件名}/`，Colab 上用 `--work_dir` 指向 Drive。
- 看某个主题的完整样本用 `scripts/view_topic_samples.py`（只读，不写文件）。
