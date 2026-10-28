# 题库扩增成 SFT 数据

`scripts/augment_probe.py` 从评测题库（默认 `eval_data/small_llm_knowledge_probe_142.jsonl`）里提取知识概念，围绕每个概念调用 DeepSeek 出题、作答，过滤后组装成 `train_full_sft.py` 能直接读取的 SFT 数据。脚本本身不训练。

目标是补充题库考到的知识，而不是让模型背题：出题时把来源原题作为禁止清单传给模型，组装时再丢掉和原题太像的问题。

**调用 API 会产生费用**。`concepts` / `questions` / `answers` 三个阶段调 API，`merge` / `build` 不调。

## 1. 准备

在仓库根目录运行，本地或 Colab 都可以。本地需要：

```bash
pip install openai==1.59.6 transformers==4.57.6 jinja2==3.1.2
```

不需要 torch。API key 读环境变量 `DEEPSEEK_API_KEY`，也可以把 `.env.example` 复制成根目录的 `.env` 再填入（`.env` 已被 `.gitignore` 忽略；系统里已设置的同名环境变量优先）。默认模型 `deepseek-flash`，关闭思考模式。

## 2. 快速开始

先用 `--limit` 冒烟，确认每步输出正常：

```bash
python scripts/augment_probe.py --stage all --limit 3
```

再完整运行：

```bash
python scripts/augment_probe.py --stage all
```

最终训练数据是 `dataset/probe_aug_YYYYMMDD/5_sft.jsonl`，可以直接作为 `train_full_sft.py` 的 `--data_path`。

## 3. 五个阶段

可以用 `--stage` 单独运行某一步，文件按步骤编号：

| 阶段 | 调 API | 输出 | 做什么 |
|---|---|---|---|
| `concepts` | 是 | `1_concepts_raw.jsonl` | 从每道题提取 2–4 个核心概念和 2–3 个相关概念，各带一句解释 |
| `merge` | 否 | `2_concepts.jsonl` | 按概念名（去掉空白和标点后）去重成概念表，记录来源题 |
| `questions` | 是 | `3_questions.jsonl` | 每个概念按 `--questions_per_concept` 种题型出题，来源原题作为禁止清单 |
| `answers` | 是 | `4_answers.jsonl` | 逐题生成回答 |
| `build` | 否 | `5_sft.jsonl` 等 | 过滤并组装训练数据 |

- **可以人工删改概念表。** `merge` 之后先检查 `2_concepts.jsonl`，删掉不想要的概念再跑 `questions`。`answers` 和 `build` 只处理概念表里还在的概念。注意**重跑 `merge` 会覆盖你的修改**。
- **题型。** 从定义、举例、对比、原因、应用、误区中按概念抽取，种子由 `--seed` 和概念 id 决定，续跑时不变；按 `--false_premise_ratio` 的比例把最后一题换成「错误前提」题，训练模型先指出错误再回答。
- **概念 id。** 由概念名决定，重跑 `merge` 或删改概念表后 id 不变，续跑不会错位。

`build` 依次丢弃：

| 原因 | 条件 |
|---|---|
| `truncated` | 回答撞上 `--answer_max_tokens` 或为空 |
| `duplicate` | 问题（去掉空白和标点后）和已保留的重复 |
| `similar_to_probe` | 和任一原题的字级 TF-IDF 余弦相似度 ≥ `--sim_threshold`，或较短的一句被另一句包含 |
| `too_long` | 套 chat template 后超过 `--max_seq_len - --reserve_tokens` |

输出三个文件：

- `5_sft.jsonl`：训练数据，每行只有 `conversations`（`SFTDataset` 用固定字段加载，多余的顶层字段会导致加载失败）。
- `5_sft_meta.jsonl`：和训练数据逐行对应的元数据（概念、题型、token 数、来源题 id）。
- `5_sft_dropped.jsonl`：被丢弃的样本和原因，`similar_to_probe` 会附上匹配到的原题 id 和相似度。

## 4. 工作目录

每次运行的文件都写在 `dataset/probe_aug_YYYYMMDD/`（已被 `.gitignore` 忽略）：

- 新跑（`--stage concepts` 或 `all`，且不加 `--resume`）用当天日期的目录，**同一天重跑会覆盖**。
- 其余阶段和 `--resume 1` 默认接着用最近一次的目录，跨天也能续上。
- 用 `--work_dir` 可以指定其他目录；用 `--output` 可以改最终数据的路径（元数据和丢弃明细跟着它命名）。

## 5. 回答长度

三个参数一起控制：

- `--answer_chars`（默认 300）：写进提示词的字数上限，是软限制。
- `--answer_max_tokens`（默认 512）：API 单次输出上限，是硬限制；撞上限的回答在 `build` 时丢弃。
- `--max_seq_len`（默认 768）：**要和训练时的 `--max_seq_len` 一致**（`train_full_sft.py` 默认 768）。`--reserve_tokens`（默认 64）给训练时随机添加的 system 预留空间。

生成后可以用 `scripts/token_length_stats.py --data dataset/probe_aug_YYYYMMDD/5_sft.jsonl` 再核对一遍长度分布。

## 6. 续跑与失败

调 API 的阶段支持 `--resume 1`：跳过输出文件里已有的条目，在后面追加；断连时写了一半的最后一行会被清掉。失败的条目不写入，结束时会打印出来，加 `--resume 1` 重跑即可再试。不加 `--resume` 会覆盖旧输出。

`--limit N` 让每个调 API 的阶段只跑前 N 条，可以和 `--resume 1` 配合逐步扩大。

## 7. 其他参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data` | `eval_data/small_llm_knowledge_probe_142.jsonl` | 题库，每行至少有 `id` 和 `question` |
| `--model` | `deepseek-flash` | 也可用 `deepseek-v4-pro` |
| `--base_url` | `https://api.deepseek.com` | API 地址 |
| `--env_file` | 根目录 `.env` | 启动时读取的环境变量文件 |
| `--workers` | 8 | 并发请求数 |
| `--temperature` | 1.0 | 采样温度 |
| `--max_retries` / `--timeout` | 5 / 300 | 429、5xx、超时的自动重试次数和单次超时（秒） |
| `--max_attempts` | 3 | 返回的 JSON 无效时重新请求的次数 |
| `--json_max_tokens` | 1024 | `concepts` / `questions` 阶段的单次输出上限 |
| `--questions_per_concept` | 5 | 每个概念出几道题（每题一种题型，最多 6） |
| `--false_premise_ratio` | 0.2 | 把最后一题换成错误前提题的概念比例 |
| `--sim_threshold` | 0.5 | 与原题的相似度阈值，越小越严格 |
| `--seed` | 42 | 决定每个概念抽到的题型 |
