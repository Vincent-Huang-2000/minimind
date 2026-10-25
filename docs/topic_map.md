# 数据集主题分布分析指南

`scripts/topic_map.py` 用来回答"这个 jsonl 数据集里都是些什么内容、各占多少"。流程分四步：

1. **向量化**：用 sentence-transformers 模型把每条样本转成一个向量，分片存盘。
2. **聚类**：用 K-Means 把全部样本分成 `--k` 个主题，再把主题中心聚成 `--k_top` 个大类。
3. **关键词**：每个主题抽样，用 jieba 分词，按 c-TF-IDF 提关键词，并挑出代表样本。
4. **报告**：生成 Markdown 报告和 CSV 表。

另有一个可选步骤：调 DeepSeek 给主题和大类起中文名。

前两步处理全量数据，后两步只在抽样上做，所以 GB 级数据集也能跑完。脚本不训练模型，但向量化需要 GPU，**建议在 Colab 上运行**。本地只适合用 `--limit` 跑几百条做冒烟测试。

## 1. 快速开始（Colab）

Colab 自带 torch 和 sentence-transformers，另外需要装 jieba：

```bash
pip install jieba
```

在仓库根目录运行。输出目录指向 Drive，断连后可以续跑：

```bash
python scripts/topic_map.py --stage all --data dataset/sft_t2t_mini.jsonl --work_dir /content/drive/MyDrive/minimind/topic_map/sft_t2t_mini
```

向量化每写完一个分片（默认 10 万条）会打印累计条数和速度，可以据此估算全量要多久。Colab 断连后重连，加 `--resume 1` 重新运行同一条命令，已经写好的分片会跳过：

```bash
python scripts/topic_map.py --stage all --data dataset/sft_t2t_mini.jsonl --work_dir /content/drive/MyDrive/minimind/topic_map/sft_t2t_mini --resume 1
```

跑完后在 notebook 里查看报告：

```python
from IPython.display import Markdown
Markdown(open('/content/drive/MyDrive/minimind/topic_map/sft_t2t_mini/report.md', encoding='utf-8').read())
```

可选：给主题起中文名（调 API 会产生费用，不包含在 `all` 里），起完名后重新生成报告：

```bash
python scripts/topic_map.py --stage label --work_dir /content/drive/MyDrive/minimind/topic_map/sft_t2t_mini
python scripts/topic_map.py --stage report --work_dir /content/drive/MyDrive/minimind/topic_map/sft_t2t_mini
```

`label` 需要的 API 调用次数是"主题数 + 大类数"，默认约 220 次。API key 的配置方式与 `augment_probe.py` 相同（环境变量 `DEEPSEEK_API_KEY` 或根目录 `.env`）。

## 2. 输出文件

全部写在 `--work_dir` 下。默认目录是 `dataset/topic_map/{数据文件名}/`，已被 `.gitignore` 忽略。

| 文件 | 阶段 | 内容 |
|---|---|---|
| `meta.json` | embed | 数据路径、条数、向量模型和文本提取参数；后续阶段按它读取原始数据 |
| `1_embeddings/part_00000.npy …` | embed | 向量分片，float16，每片 `--shard_size` 条，已归一化 |
| `2_labels.npy` | cluster | 每条样本的主题编号（int32），第 i 个值对应数据文件的第 i 个非空行 |
| `2_scores.npy` | cluster | 每条样本与所属主题中心的余弦相似度（float16） |
| `2_centroids.npy` | cluster | 主题中心向量，主题按样本数从大到小编号，0 号最大 |
| `2_groups.npy` | cluster | 每个主题所属的大类编号，大类也按样本数从大到小编号 |
| `3_topics.jsonl` | keywords | 每个主题一行：大类、条数、占比、平均相似度、关键词、代表样本、随机样本 |
| `3_groups.jsonl` | keywords | 每个大类一行：条数、占比、包含的主题、关键词 |
| `4_topic_labels.jsonl` / `4_group_labels.jsonl` | label | DeepSeek 起的名字和一句话说明 |
| `report.md` | report | 主要看这个：大类分布表 + 每个大类下的主题表 |
| `topics.csv` | report | 每个主题一行的汇总表，可以用 Excel 打开 |

`3_topics.jsonl` 的一行大致如下：

```json
{"topic": 12, "group": 3, "size": 18234, "ratio": 0.0152, "score": 0.8731,
 "keywords": ["函数", "python", "列表", "代码", ...],
 "examples": [{"row": 88123, "score": 0.9412, "text": "如何用 Python 把列表去重？"}, ...],
 "random_examples": [...]}
```

`examples` 是与主题中心最相似的几条，最能代表这个主题；`random_examples` 是随机抽的，能看出主题内部有多杂。`row` 是样本在原始数据里的行号（只计非空行）。

找出某个主题的全部样本：

```python
import json, numpy as np
labels = np.load('.../2_labels.npy')
rows = set(np.where(labels == 12)[0])
with open('dataset/sft_t2t_mini.jsonl', encoding='utf-8') as f:
    samples = [json.loads(l) for i, l in enumerate(x for x in f if x.strip()) if i in rows]
```

## 3. 文本怎么取

| 数据格式 | 取哪个字段 |
|---|---|
| 预训练 `{"text": ...}` | `text` |
| SFT / RLAIF / Agent RL `{"conversations": [...]}` | 默认只取用户的发言；加 `--text_mode all` 时，用户和助手的发言都取（见下文） |
| DPO `{"chosen": [...], "rejected": [...]}` | 只取 `chosen`，规则同上；`rejected` 不取 |
| 题库 `{"question": ...}` | `question` |

字段不在上表时，用 `--field 字段名` 指定。

### 用户和助手指什么

对话数据的 `conversations` 是一个消息列表，每条消息用 `role` 字段标明是谁说的：

| `role` | 指什么 | 默认（`--text_mode user`） | `--text_mode all` |
|---|---|---|---|
| `user` | 用户：人提出的问题或指令 | 取 | 取 |
| `assistant` | 助手：模型给出的回答，也就是 SFT 时模型要学着生成的部分 | 不取 | 取 |
| `system` | 系统提示词；Agent RL 数据的工具定义（`tools` 字段）也放在这条消息里 | 不取 | 不取 |

两种模式都只取每条消息的 `content` 字段。思考过程 `reasoning_content`、工具定义 `tools`、工具调用 `tool_calls` 都不取。

选中的发言按原来的顺序用换行拼成一段文本。比如这条两轮的 SFT 样本：

```json
{"conversations": [
  {"role": "user", "content": "什么是质数？"},
  {"role": "assistant", "content": "质数是大于 1、只能被 1 和它本身整除的整数，比如 2、3、5、7。"},
  {"role": "user", "content": "1 是质数吗？"},
  {"role": "assistant", "content": "不是。质数必须大于 1，所以 1 既不是质数也不是合数。"}
]}
```

默认只取用户的发言，得到：

```
什么是质数？
1 是质数吗？
```

加 `--text_mode all`，得到：

```
什么是质数？
质数是大于 1、只能被 1 和它本身整除的整数，比如 2、3、5、7。
1 是质数吗？
不是。质数必须大于 1，所以 1 既不是质数也不是合数。
```

### 用户和助手的发言都取

完整命令。和第 1 节相比，只多了 `--text_mode all`，并换了一个输出目录：

```bash
python scripts/topic_map.py --stage all --data dataset/sft_t2t_mini.jsonl --work_dir /content/drive/MyDrive/minimind/topic_map/sft_t2t_mini_all --text_mode all
```

断连后续跑，`--text_mode all` 也要带上：

```bash
python scripts/topic_map.py --stage all --data dataset/sft_t2t_mini.jsonl --work_dir /content/drive/MyDrive/minimind/topic_map/sft_t2t_mini_all --text_mode all --resume 1
```

注意事项：

- **要换一个输出目录。** `--text_mode` 决定了向量表示的是什么内容。在原来的目录里不加 `--resume 1` 重跑，会覆盖只取用户发言的结果；加了 `--resume 1`，会报"续跑参数与已有向量不一致"。换个目录，两份结果还能对比着看。
- **只有 `embed` 阶段需要指定。** 之后的 `cluster`、`keywords`、`label`、`report` 阶段都从 `meta.json` 读取 `text_mode`，单独运行时不用再加。
- **回答长，会挤掉后面的轮次。** 文本先截断到 `--max_chars`（默认 1000 字），模型再只看前 `--max_seq_length`（默认 256）个 token。回答通常比问题长得多，所以多轮对话的向量基本由第一轮的问题和回答决定。想让模型看到更多内容，可以加 `--max_seq_length 512`（e5-small 的上限），代价是向量化更慢。
- **RLAIF 和 Agent RL 数据用 `all` 差别不大。** 这两类数据是强化学习的提示，最后一条助手消息是空的，回答要在训练时由模型生成。所以 `all` 只会多取前面几轮已有的回答；单轮样本（如 `agent_rl_math`）两种模式得到的文本完全一样。
- **报告内容也会变。** `keywords` 阶段同样用拼接后的文本统计关键词、挑代表样本，所以关键词里会混进回答里的常用词，代表样本也会带上回答。

现在每一行（一整段对话）只算一个向量、一个主题。跨话题的多轮对话会被平均掉；统计单位也是对话数，不是问题数或训练 token 数。后续的改进想法（按轮拆分、按 token 加权等）见 [topic_map_ideas.md](topic_map_ideas.md)。

**怎么选：** 想看"数据里问了哪些知识"，保持默认的只取用户提问。回答通常很长，取进来会让聚类偏向回答的写法，而不是问题的主题。想看"模型学到了哪些回答内容"（比如回答里讲了什么知识、用了什么格式），再用 `--text_mode all`。

文本先截断到 `--max_chars` 个字，再由模型按 `--max_seq_length` 个 token 截断。所以预训练数据里很长的样本，主要按开头部分归类。

## 4. 参数怎么选

**向量模型（`--model`）**

| 模型 | 说明 |
|---|---|
| `intfloat/multilingual-e5-small`（默认） | 多语言，中英文同一个话题能聚到一起；MiniMind 的 SFT 数据里混有英文，推荐用它。需要 `--prefix 'query: '`（默认值） |
| `BAAI/bge-small-zh-v1.5` | 中文效果好，英文一般。换用时加 `--prefix ''` |
| `minishlab/potion-multilingual-128M` | 静态向量，比 transformer 快一到两个数量级，CPU 也能跑全量，质量差一些。需要较新的 sentence-transformers（3.x 以上），加 `--prefix ''` |

**主题数**

- `--k`：主题数，默认 200。几十万条数据用 100～300；百万级以上想看细一点，可以用 500～1000。主题越多，每个主题越窄。
- `--k_top`：大类数，默认 20。
- 改 `--k` 或 `--k_top` 只需要重跑 `cluster` 及之后的阶段，不用重新向量化：

```bash
python scripts/topic_map.py --stage cluster --work_dir ... --k 500 --k_top 30
python scripts/topic_map.py --stage keywords --work_dir ...
python scripts/topic_map.py --stage report --work_dir ...
```

注意：重跑 `cluster` 会让主题编号全部重排，之前 `label` 生成的名字就对不上了，要重新跑 `label`。

**速度相关**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch_size` | 256 | 显存够可以调大 |
| `--max_seq_length` | 256 | 调小更快，但长文本被截断得更多 |
| `--train_size` | 500000 | K-Means 只在这么多条抽样上训练，训练完再给全量分配主题 |
| `--kw_samples` | 200 | 每个主题抽多少条做分词；jieba 是单线程，`--k` 很大时可以调小 |

## 5. 各阶段单独运行

| `--stage` | 需要 GPU | 依赖 | 说明 |
|---|---|---|---|
| `embed` | 是（CPU 很慢） | torch、sentence-transformers | 最耗时；支持 `--resume 1` 续跑、`--limit N` 冒烟 |
| `cluster` | 推荐 | torch | 全量数据分片读取，内存里同时只放一片 |
| `keywords` | 否 | jieba | 会再读一遍原始数据，只取被抽中的样本 |
| `label` | 否 | openai | 调 API，收费；支持 `--resume 1`、`--limit N` |
| `report` | 否 | 无 | 有 `label` 的结果就用中文名，没有就用关键词 |
| `all` | | | `embed` → `cluster` → `keywords` → `report` |

`embed` 之后的阶段只需要 `--work_dir`，数据路径从 `meta.json` 读取。换了一个 Colab 会话、数据路径变了时，用 `--data` 指定新路径。文件大小与向量化时不同会报错，因为行号已经对不上了。

## 6. 常见问题

**续跑时报"续跑参数与已有向量不一致"。** `--model`、`--prefix`、`--field`、`--text_mode`、`--max_chars`、`--max_seq_length`、`--shard_size` 这几个参数决定了向量的含义，续跑时必须和原来一致。想换参数就去掉 `--resume 1`，从头向量化。

**主题看起来很杂。** 看报告里的"相似度"列，值低的主题比较松散，调大 `--k` 能拆得更细。也可以看 `3_topics.jsonl` 里的 `random_examples`，判断是真的杂还是关键词没提好。

**关键词里有没意义的词。** 在脚本开头的 `STOPWORDS` 里加上这些词，再重跑 `keywords` 和 `report`。

**想对比两个数据集。** 目前每个数据集单独聚类，主题编号不能直接对比。可以把两个文件合并后一起分析，再按行号区间统计各主题里两边的占比。
