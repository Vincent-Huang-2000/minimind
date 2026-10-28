# 预训练打包对比实验

用同一份数据各训一次打包（`--pack 1`）和不打包的预训练，比较三件事：

1. **数据利用率**：每步的有效 token、每个 epoch 的步数、不打包时截掉多少 token。
2. **训练曲线**：两条 loss 曲线按"看过的有效 token 数"对齐后放在一张图上。
3. **留出集 loss**：在两次训练都没见过的文档上，用相同的方式计算 loss / perplexity。这一项是质量对比的结论。

打包本身的原理和注意事项见 [`docs/pretrain_packing.md`](../../docs/pretrain_packing.md)。

## 文件

| 脚本 | 作用 | 依赖 | 运行环境 |
|---|---|---|---|
| `split_heldout.py` | 随机切出留出集，其余作为训练集，并去掉训练集里与留出集文本相同的行 | 标准库 | 本地 / Colab |
| `data_efficiency.py` | 构建打包缓存，统计两种模式的行数和有效 token，写入 `efficiency.json` | numpy、transformers | 本地 / Colab |
| `compare_metrics.py` | 读入两次训练的 `*_metrics.jsonl`，按有效 token 对齐出对比图，并打印对齐点的 loss | matplotlib | 本地 / Colab |
| `eval_heldout_loss.py` | 在留出集上计算各个权重的 loss / ppl，写入 `heldout_loss.json` | torch | Colab |

所有脚本都在**仓库根目录**运行。输出统一写到 `--work_dir`（默认 `out/pack_compare/`，已被 `.gitignore` 忽略），Colab 上应指向 Drive。

## 步骤（Colab）

下面的 `D=/content/drive/MyDrive/minimind` 是 Drive 上的目录，按实际情况替换。

### 1. 切出留出集

```bash
python experiment/pack_compare/split_heldout.py --data dataset/pretrain_t2t_mini.jsonl --n 2000
```

会生成 `dataset/pretrain_t2t_mini_train.jsonl` 和 `dataset/pretrain_t2t_mini_heldout.jsonl`。两次训练都只用 `_train`。

切分由 `--seed` 决定，同样的参数重跑结果相同。`/content` 断开后会被清空，可以把两个文件复制到 Drive，也可以下次重新切。

### 2. 统计数据利用率，同时提前建好打包缓存

```bash
python experiment/pack_compare/data_efficiency.py --data dataset/pretrain_t2t_mini_train.jsonl \
    --max_seq_len 340 --batch_size 32 --pack_cache_dir $D/pack_cache --work_dir $D/pack_compare
```

这一步会打印两种模式的对比表，并写出 `$D/pack_compare/efficiency.json`。`--max_seq_len`、`--batch_size`、`--seed`、`--pack_cache_dir` 要和后面训练时用的一致：这样打包训练会直接复用这里建好的缓存，`compare_metrics.py` 换算 token 数也才准确。

光看这张表就能知道打包能省多少：两种模式每步的张量形状相同，每步耗时基本一样，所以训练时间之比约等于"每 epoch 步数"之比。

### 3. 两次训练

```bash
cd trainer
python train_pretrain.py --pack 0 --save_weight pretrain_nopack --data_path ../dataset/pretrain_t2t_mini_train.jsonl \
    --epochs 1 --save_dir $D/out --dtype float16
python train_pretrain.py --pack 1 --save_weight pretrain_pack --data_path ../dataset/pretrain_t2t_mini_train.jsonl \
    --epochs 1 --pack_cache_dir $D/pack_cache --save_dir $D/out --dtype float16
cd ..
```

- **`--save_weight` 必须不同。** 权重、`*_metrics.jsonl` 和 `../checkpoints/` 里的续训文件都按这个名字区分。
- 除了 `--pack` / `--pack_cache_dir`，其余参数保持一致。
- T4 用 `--dtype float16`，L4 / A100 可以去掉，使用默认的 bfloat16。
- 断开后加 `--from_resume 1` 续训，参数保持不变。
- 训练时间太长的话，两次都换成同一个更小的子集（先截取子集，再从第 1 步重新做）。

### 4. 对比训练曲线

```bash
python experiment/pack_compare/compare_metrics.py --nopack $D/out/pretrain_nopack_metrics.jsonl \
    --pack $D/out/pretrain_pack_metrics.jsonl --work_dir $D/pack_compare
```

终端会打印共同 token 范围内 25% / 50% / 75% / 100% 处平滑后的 loss，并生成 `$D/pack_compare/compare_tokens.png`。加 `--x step` 可以按步数画图，但两边的步数含义不同，只用来看训练进度。

如果某次训练的步数和 `efficiency.json` 对不上，脚本会给出警告，通常是数据或参数不一致。

在 notebook 里查看：

```python
from IPython.display import Image
Image('/content/drive/MyDrive/minimind/pack_compare/compare_tokens.png')
```

**训练 loss 只看趋势，不能直接比数值高低**：

- 打包时同一行的文档能互相看到。
- 长文档的后半段也参与训练，两边训练的 token 构成不同。

### 5. 留出集 loss

```bash
python experiment/pack_compare/eval_heldout_loss.py --data dataset/pretrain_t2t_mini_heldout.jsonl \
    --weights pretrain_nopack,pretrain_pack --save_dir $D/out --work_dir $D/pack_compare
```

每篇留出文档写成 `[bos] 文本 [eos]`，按 `--max_seq_len` 切成窗口，每个窗口单独计算。loss 按 token 平均，并拆成三列：

| 列 | 含义 |
|---|---|
| `all` | 全部窗口 |
| `head` | 每篇的第一个窗口，也就是不打包训练时保留下来的部分，两边都训练过 |
| `tail` | 超长文档后续的窗口，不打包时被截断丢掉，只有打包训练过；窗口单独计算，看不到前文 |

结果写入 `$D/pack_compare/heldout_loss.json`。重跑某个权重只覆盖它自己的那一条；换了留出集或窗口长度，会整份重写。

## 怎么解读

- **`head` loss**：判断打包对"原来就训练的那部分"有没有副作用。两边差不多或打包更低，就说明文档互相可见、位置偏移这些影响可以忽略。
- **`tail` loss**：打包额外带来的收益，不打包的模型没见过这些位置的内容，一般会明显更差。
- **训练时间**：按第 2 步的步数比估算，或者直接对比两次训练的日志。
- **结论**：`head` 不变差，同时训练时间按预期缩短，就可以把打包作为预训练的默认设置。

如果还想看下游效果，可以两个权重分别做 SFT，再用 `scripts/run_probe.py` 比较作答。预训练阶段的差别经过 SFT 后可能被冲淡，成本也最高，放到最后。
