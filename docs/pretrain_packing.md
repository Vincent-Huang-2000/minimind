# 预训练装箱打包

`train_pretrain.py --pack 1` 会把多篇完整文档装进同一个 `max_seq_len` 序列，减少 padding 浪费的算力；默认 `--pack 0`，行为和原来完全一样。只影响预训练，SFT 等其他阶段不变。

## 1. 为什么要打包

默认的 `PretrainDataset` 每篇文档单独占一行：开头加 bos、结尾加 eos，超长就截断丢掉，不够长就用 pad 补到 `max_seq_len`。pad 不算 loss，但前向和反向计算照样要做。短文档越多，浪费越多；超长文档被截掉的部分也直接丢了。

用 `scripts/token_length_stats.py` 可以看到这两项的比例（`padding 占比`、`被截掉的 token`）。用 `pretrain_t2t_mini` 前 20 行估算，`max_seq_len 340` 时约 38% 是 padding，约 6% 的 token 被截掉；20 行只能粗看，全量要在 Colab 上统计。

## 2. 快速开始（Colab）

```bash
cd trainer
python train_pretrain.py --pack 1 --pack_cache_dir /content/drive/MyDrive/minimind/pack_cache \
    --save_dir /content/drive/MyDrive/minimind/out --dtype float16
```

第一次运行会先分词并装箱，日志里有 `[pack] 已分词 ...` 的进度，结束时打印行数和填充率。结果缓存在 `--pack_cache_dir`（默认是数据文件同目录下的 `pack_cache/`，已被 `.gitignore` 忽略）。之后用同样的数据、`--max_seq_len`、`--seed` 再运行，直接读缓存，不再分词。

`/content` 在 Colab 断开后会被清空，缓存目录建议放在 Drive 上，否则每次新会话都要重新分词。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--pack` | 0 | 1 开启装箱打包 |
| `--pack_cache_dir` | 数据文件同目录的 `pack_cache/` | 打包缓存目录 |

## 3. 怎么打包

实现在 `dataset/pretrain_packing.py`（只依赖 numpy 和 tokenizer），`dataset/lm_dataset.py` 的 `PackedPretrainDataset` 负责读取。

1. **分词**：逐行读取 `text`，每篇写成 `[bos] 文本 [eos]`，首尾相接追加到 `*.tokens.bin`（uint16）。和 `PretrainDataset` 一样不额外加特殊 token。
2. **切段**：超过 `max_seq_len` 的文档按 `max_seq_len` 切成多段，最后一段带 eos。原来被截断丢掉的部分现在也会参与训练。
3. **装箱（best-fit）**：用 `--seed` 打乱段的顺序，逐段放进“剩余空间最小且放得下”的行，都放不下就开新行。短文档不会被切开，每行都从一篇文档（或长文档的一段）的开头开始。
4. **读取**：`__getitem__` 取出一行的各段拼起来，行尾不足 `max_seq_len` 的部分补 pad，标签设为 -100；其余位置的标签就是 token 本身。
5. **缓存**：`{数据名}_pack{max_seq_len}_{指纹}.tokens.bin` / `.index.npz` / `.stats.json`。指纹由数据文件大小与首尾各 1MB 内容、`max_seq_len`、`--seed`、tokenizer 共同决定，任何一项变了都会生成新缓存。`.index.npz` 最后写入，它存在才表示缓存完整，构建中途断开会在下次重建。

`.stats.json` 记录文档数、切出的段数、总 token 数、行数和填充率（有效 token / 行数 × `max_seq_len`）。

## 4. 对训练的影响

**好处**

- 每一步的有效 token 变多，同样的数据步数和时间都会减少（减少的比例约等于原来的 padding 占比）。
- 超长文档不再截断丢弃。
- `--max_seq_len` 可以调大（如 512、768）而不用付 padding 的代价，模型能看到更长的上下文。
- 短文档（例如概念扩写生成的 CPT 短文）可以直接用，不必要求每篇都写到接近 `max_seq_len`。

**需要注意**

- **同一行里的文档能互相看到**：没有加文档隔离的注意力掩码，后一篇能注意到前一篇的内容，文档之间只靠 eos / bos 分隔。早期 GPT 类预训练普遍这样做；也有研究发现加文档隔离掩码能带来小幅提升。在 340–768 这样的短上下文里影响一般很小，但没有在 MiniMind 上验证过。
- **位置不从 0 开始**：除了每行第一段，其他文档的 RoPE 位置从行内偏移开始。RoPE 编码的是相对位置，影响不大。
- **步数变了，曲线按 token 对比**：一个 epoch 的步数会变少，和不打包的训练曲线对比时，要按"看过的 token 数"对齐，不要按步数。loss 都是按有效 token 平均的，数值本身可比。
- **有效 batch 变大**：每步 token 更多，优化器更新次数更少。学习率调度按总步数计算会自动适应；`--learning_rate` 是否需要调整，用小规模对比确认。

## 5. 续训

`--from_resume 1` 照常可用：缓存内容和行的顺序是确定的，每个 epoch 的打乱和跳过已训练批次的逻辑与原来相同。

**续训时不要改 `--pack`、`--max_seq_len`、`--data_path` 或 `--seed`**。这些变了，数据集的行数和顺序就变了，断点里记录的步数会对应到别的数据上，脚本不会报错。要换配置，就开一次新的训练。

## 6. 建议的效果对比

在 Colab 上用同一份数据各训一次（打包和不打包），比较数据利用率、按 token 对齐的 loss 曲线和留出集 loss。步骤和脚本见 [`experiment/pack_compare/README.md`](../experiment/pack_compare/README.md)。还想看下游效果，可以再分别做 SFT，用 `scripts/run_probe.py` 比较作答。

## 7. 常见问题

**第一次运行很久才开始训练。** 在分词，看 `[pack] 已分词` 的进度。完整的 `pretrain_t2t`（10GB）分词会比较久，缓存放 Drive 后只需要一次。

**内存。** token 用 memmap 读取，不会整份读进内存；索引是每段一组整数，千万级文档也只有几百 MB。

**多卡。** 只有 rank 0 构建缓存，其余 rank 等它完成后再读取。
