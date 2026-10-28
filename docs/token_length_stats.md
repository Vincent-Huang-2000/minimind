# 数据 token 长度分布

`scripts/token_length_stats.py` 按训练时的拼接方式统计数据的 token 长度，用来选择训练的 `--max_seq_len`：看各分位数，以及每个候选长度下有多少样本被截断、有多少算力浪费在 padding 上。

训练数据集会把每条样本补齐到 `max_seq_len`，且不做拼接（预训练打包除外，见 `docs/pretrain_packing.md`），所以 padding 占比基本就是浪费掉的算力。

## 1. 快速开始

在仓库根目录运行，只需要 `transformers`，不需要 torch，本地或 Colab 都能跑：

```bash
python scripts/token_length_stats.py --data dataset/sft_t2t_mini.jsonl --current 768
```

可以一次传多个文件，每个文件单独出一份报告。本地只想冒烟时加 `--limit 1000`。

## 2. 长度怎么算

tokenizer 读取仓库里的 `model/`，数据类型默认按第一条样本的字段判断（`--type auto`）：

| 类型 | 判断依据 | 统计的文本 |
|---|---|---|
| `pretrain` | 有 `text` | `text` 本身，加上 bos / eos 两个 token |
| `sft` | 有 `conversations` | 套 chat template，去掉空思考标签，不加随机 system |
| `dpo` | 有 `chosen` | `chosen` 和 `rejected` 分别按 SFT 方式统计，取较长的 |
| `prompt` | 需手动指定 | PPO / GRPO / Agent RL 数据：去掉最后一条回答，只统计 prompt（带生成提示） |

和训练时的细微差别：

- 训练时 SFT 有 20% 的概率随机加一条 system，这里不加，所以结果略偏短。
- 训练时约 80% 的情况会去掉空思考标签，这里一律去掉。
- 对话里字符串形式的 `tools` / `tool_calls` 会先解析成 JSON 再套模板，和 `lm_dataset.py` 一致。

## 3. 参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--data` | 必填 | jsonl 路径，可以传多个 |
| `--type` | `auto` | `auto` / `pretrain` / `sft` / `dpo` / `prompt` |
| `--sample` | 20000 | 蓄水池抽样条数，会扫一遍全文件但不整份读进内存；`0` 为全部统计 |
| `--limit` | 0 | 只读前 N 行，冒烟用；和 `--sample` 同时给时先截前 N 行再抽样 |
| `--candidates` | 256 340 384 512 640 768 1024 1536 2048 | 要对比的 `max_seq_len` 候选值 |
| `--target_pct` | 95 | 推荐值要覆盖的分位数 |
| `--current` | 0 | 当前训练用的 `max_seq_len`，会在表里标出来 |
| `--seed` | 42 | 抽样的随机种子 |

## 4. 输出怎么看

每个文件输出：

- 平均长度，p50 / p75 / p90 / p95 / p99 和最大值。
- 中位数的「字符 / token」比，用来估算按字数写的数据（比如生成数据时的字数限制）大约占多少 token。
- 一张表，行是候选长度、覆盖 `--target_pct` 的推荐值（向上取到 64 的倍数）和 `--current`：

| 列 | 含义 |
|---|---|
| 截断样本 | 长度超过该值的样本比例 |
| 被截掉的 token | 被截掉的 token 占全部 token 的比例 |
| padding 占比 | 补齐的 pad 占全部位置（样本数 × 该长度）的比例 |
| 备注 | `当前默认`、`覆盖 p95` 等标记 |

选长度时在截断和 padding 之间权衡：截断会让 SFT 回答没有结尾，padding 会浪费算力。
