# AGENTS.md

## 项目用途

- 本仓库是 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的个人 fork（`Vincent-Huang-2000/minimind`），用于学习和实践从零训练小型语言模型 MiniMind（主线约 64M 参数）。
- 覆盖完整训练链路：Pretrain → SFT → LoRA / DPO / PPO / GRPO / 蒸馏 / Agentic RL，核心代码均为 PyTorch 原生实现。
- 个人改动在 `custom` 分支上，`master` 跟随上游。

## 训练环境：只在 Google Colab 中训练

**所有训练都只在 Colab 上进行，不在本地机器上训练。** 本地（Windows 笔记本）只用来阅读、修改和提交代码。

对 agent 的要求：

- 不要在本地启动训练脚本，也不要为了"验证"在本地下载完整数据集；本地最多做语法 / import 检查或极小样例的冒烟测试。
- 给训练或性能优化建议时，以 Colab 环境为准，不要针对本地 GPU 或 Windows 做适配。
- Colab 为单卡环境，DDP / 多卡相关的优化优先级很低。
- Colab 运行时会断开、`/content` 下的文件会被清空：训练产物应保存到 Google Drive，长时间训练依赖 `--from_resume 1` 续训，改动保存 / 续训逻辑时要格外小心。
- Colab 分配的 GPU 型号不固定：T4 不原生支持 bf16，应使用 `--dtype float16`（脚本会自动启用 GradScaler）；L4 / A100 可使用默认的 `bfloat16`。

## 目录与运行约定

- 训练脚本在 `trainer/`，需在 `trainer/` 目录下运行，脚本中的 `../dataset`、`../out`、`../checkpoints` 都是相对路径。
- 推理权重输出到 `out/`（可用 `--save_dir` 修改）；续训文件写入 `checkpoints/`，该路径在脚本中写死，要放到 Drive 需建软链接或改代码。
- 数据集放在 `dataset/`，用 `python scripts/download_dataset.py` 下载（默认 mini 组合，`--list` 查看全部）；`dataset/*.jsonl` 已加入 `.gitignore`，不要提交数据文件。
- 评测：`python eval_llm.py --weight full_sft`。
- 评测题库放在 `eval_data/`（如知识探针 `small_llm_knowledge_probe_142.jsonl`），与训练数据 `dataset/` 分开；不在 `.gitignore` 范围内，可以提交。
- 题库批量作答：`python scripts/run_probe.py --weight full_sft`（默认读 `eval_data/small_llm_knowledge_probe_142.jsonl`，换题库用 `--data`），在仓库根目录运行；每题追加写入 `{save_dir}/{weight}_{题库名}_answers.jsonl`（原字段 + `answer` / `gen_tokens` / `truncated`），断连后加 `--resume 1` 续跑。
- 题库扩增成 SFT 数据：`python scripts/augment_probe.py --stage all`（在仓库根目录运行，API key 读环境变量 `DEEPSEEK_API_KEY`，也可写在根目录 `.env` 里，模板 `.env.example`，`.env` 已被 `.gitignore` 忽略；默认 `deepseek-flash`、关闭思考模式）。每次运行的文件都写在 `dataset/probe_aug_YYYYMMDD/`（已被 `.gitignore` 忽略）：新跑（`concepts` / `all` 且不加 `--resume`）用当天日期，同一天重跑会覆盖；其余阶段和 `--resume 1` 默认接着用最近一次的目录，也可用 `--work_dir` 指定。分五步，可用 `--stage` 单独跑，文件按步骤编号：`concepts` 从每道题提取核心 / 相关概念（`1_concepts_raw.jsonl`）→ `merge` 按概念名去重成概念表（`2_concepts.jsonl`，不调 API，可先人工删改；重跑会覆盖修改）→ `questions` 围绕每个概念按题型出题，把来源原题作为禁止清单传给模型（`3_questions.jsonl`）→ `answers` 生成回答（`4_answers.jsonl`）→ `build` 丢弃截断、重复、与原题相似（`--sim_threshold`）、超过 `--max_seq_len - --reserve_tokens` 的样本，输出训练数据 `5_sft.jsonl`（只含 `conversations`，元数据在 `5_sft_meta.jsonl`，丢弃明细在 `5_sft_dropped.jsonl`）。回答长度由 `--answer_chars`（提示词软限制）和 `--answer_max_tokens`（API 硬上限）控制，`--max_seq_len` 要和训练时一致（`train_full_sft.py` 默认 768）。调 API 的阶段支持 `--limit` 冒烟、`--resume 1` 续跑。该脚本不训练，本地或 Colab 都能跑（本地需 `pip install openai==1.59.6 transformers==4.57.6 jinja2==3.1.2`，不需要 torch）；调用 API 会产生费用，agent 不要自行运行调 API 的阶段。
- 训练曲线：各训练脚本在每个 `log_interval` 把指标追加写入 `{save_dir}/{save_weight}_metrics.jsonl`（LoRA 用 `lora_name`），训练结束自动生成同名 `.png`（`trainer_utils.MetricsLogger`，绘图在 `scripts/plot_metrics.py`，依赖 matplotlib，Colab 自带；缺失时只告警不影响训练）。新训练覆盖旧文件，`--from_resume 1` 续训保留断点之前的记录、丢弃断点之后会被重跑的记录。训练中途或训练后手动出图：`python ../scripts/plot_metrics.py ../out/pretrain_metrics.jsonl`；在 Colab 里显示：`from IPython.display import Image; Image('../out/pretrain_metrics.png')`。指标文件跟权重一起放在 `save_dir`，指向 Drive 时曲线也会保存在 Drive。详细用法见 `docs/training_metrics.md`。
