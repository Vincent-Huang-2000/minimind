# 训练曲线使用指南

训练脚本会把 loss、reward 等指标记录到本地文件，并在训练结束时自动画成图；也可以随时用脚本对记录文件出图。不依赖 wandb / swanlab，原有的 `--use_wandb` 不受影响，两者可以同时使用。

## 1. 快速开始（Colab）

照常启动训练，不需要加任何参数：

```bash
cd trainer
python train_pretrain.py --save_dir /content/drive/MyDrive/minimind/out --dtype float16
```

训练过程中，每到 `log_interval` 就会在 `save_dir` 下追加一行记录：

```
/content/drive/MyDrive/minimind/out/pretrain_metrics.jsonl
```

训练结束时，终端会打印 `指标曲线已保存: ...`，并在同目录生成 `pretrain_metrics.png`。在 notebook 里查看：

```python
from IPython.display import Image
Image('/content/drive/MyDrive/minimind/out/pretrain_metrics.png')
```

`save_dir` 指向 Drive 时，记录文件和图片跟权重一起保存在 Drive，Colab 断连后也不会丢。

## 2. 文件命名

| 训练脚本 | 记录文件名前缀 | 默认 `log_interval` |
|---|---|---|
| `train_pretrain.py` | `--save_weight`（默认 `pretrain`） | 100 |
| `train_full_sft.py` | `--save_weight`（默认 `full_sft`） | 100 |
| `train_lora.py` | `--lora_name`（默认 `lora_medical`） | 10 |
| `train_dpo.py` | `--save_weight`（默认 `dpo`） | 100 |
| `train_distillation.py` | `--save_weight`（默认 `full_dist`） | 100 |
| `train_grpo.py` | `--save_weight`（默认 `grpo`） | 1 |
| `train_ppo.py` | `--save_weight`（默认 `ppo_actor`） | 每步都记录（忽略 `log_interval`） |
| `train_agent.py` | `--save_weight`（默认 `agent`） | 1 |

完整文件名是 `{save_dir}/{前缀}_metrics.jsonl` 和 `{save_dir}/{前缀}_metrics.png`。

## 3. 记录的指标

每个脚本记录的指标与它的 wandb 日志一致（不含 `epoch_time`）：

| 训练脚本 | 指标 |
|---|---|
| pretrain / full_sft / lora | `loss`、`logits_loss`、`aux_loss`、`learning_rate` |
| dpo | `loss`、`dpo_loss`、`aux_loss`、`learning_rate` |
| distillation | `loss`、`ce_loss`、`aux_loss`、`distill_loss`、`learning_rate` |
| grpo | `reward`、`kl_ref`、`advantages_std`、`advantages_mean`、`policy_loss`、`avg_response_len`、`learning_rate` |
| agent | `reward`、`kl_ref`、`group_reward_std`、`advantages_std`、`advantages_mean`、`policy_loss`、`avg_response_len`、`learning_rate` |
| ppo | `reward`、`kl_ref`、`approx_kl`、`clipfrac`、`critic_loss`、`avg_response_len`、`actor_lr`、`critic_lr` |

说明：

- `loss` 是整个目标的损失，`logits_loss` 是去掉 MoE 负载均衡项 `aux_loss` 后的部分。
- dense 模型的 `aux_loss` 恒为 0，出图时会自动省略；蒸馏时没有 teacher 的话 `distill_loss` 也是这样。
- 横轴是 `epoch * iters + step`，单位是 batch 步，不是优化器步。使用梯度累积时，优化器更新次数是它的 `1 / accumulation_steps`。

## 4. 手动出图

训练中途、训练结束、或 Colab 断连之后，都可以对记录文件出图：

```bash
python ../scripts/plot_metrics.py ../out/pretrain_metrics.jsonl
```

| 参数 | 含义 |
|---|---|
| `path`（位置参数） | `*_metrics.jsonl` 文件路径 |
| `--out` | 图片保存路径，默认是和 jsonl 同名的 `.png` |
| `--smooth` | 平滑系数，默认 `0.9`；`0` 表示不平滑，越接近 1 越平滑 |
| `--show` | 弹窗显示（本地有 GUI 时用；Colab 里不需要） |

图的样子：每个指标一个小图，三列排布；淡色细线是原始值，粗线是平滑后的值，标题里的 `last` 是最后一次记录的原始值。记录点不足约 200 个时平滑会自动减弱，不足约 20 个时关闭，避免平滑线明显滞后于原始值。

在 notebook 里直接内联显示：

```python
import sys; sys.path.append('/content/minimind')   # 仓库根目录
from scripts.plot_metrics import plot_metrics
plot_metrics('/content/drive/MyDrive/minimind/out/pretrain_metrics.jsonl', show=True)
```

训练进程阻塞着当前 cell 时，看不到实时曲线。想边训边看，可以把训练放到后台，再在另一个 cell 里出图：

```bash
cd /content/minimind/trainer
nohup python train_pretrain.py --dtype float16 > train.log 2>&1 &
```

## 5. 续训与覆盖规则

| 场景 | 记录文件的处理 |
|---|---|
| 新训练（不带 `--from_resume 1`，或带了但没有可恢复的 checkpoint） | 同名旧文件被清空重写，和权重文件被覆盖的行为一致 |
| `--from_resume 1` 续训 | 保留断点之前的记录；丢弃断点之后的记录（这些步骤会重新训练）；断连时写了一半的行也会被清掉 |

所以续训之后曲线是连续的，不会有重复或错位的点。想保留旧曲线再重训，先把旧的 `_metrics.jsonl` 和 `.png` 复制或改名。

## 6. 常见问题

**训练结束没有生成 png。** 看训练日志末尾是否有 `[plot] 跳过出图: ...`。最常见原因是没装 matplotlib（Colab 自带，本地需要 `pip install matplotlib`）。出图失败不影响权重和续训文件；装好后手动运行第 4 节的命令即可。

**报错 `没有非零指标可画` 或 `没有可用的指标记录`。** 记录文件是空的或所有指标都是 0。多半是训练刚开始、还没到第一个 `log_interval`，或者 `log_interval` 比总步数还大。

**曲线太抖。** 调大 `--smooth`（比如 `0.97`）。记录的是日志点那一个 batch 的值，不是区间平均，本来就比较抖。点数少于约 200 时平滑会被自动减弱，调大 `--smooth` 帮助有限。

**多卡训练。** 只有 rank 0 写文件和出图，不会重复。

**想给自己的训练循环加指标。** 创建 `metrics = MetricsLogger(save_dir, name, ckp_data)`（在 `trainer/trainer_utils.py`），在日志点调用 `metrics.log(epoch, step, iters, my_metric=value)`，结束时 `metrics.plot()`。出图脚本会自动识别新增的指标名，不需要改绘图代码。

## 7. 记录文件格式

`*_metrics.jsonl` 每行一个 JSON 对象，`epoch` 从 0 开始，`step` 从 1 开始：

```json
{"epoch": 0, "step": 100, "iters": 9532, "loss": 3.4129, "logits_loss": 3.4129, "aux_loss": 0.0, "learning_rate": 0.0004998}
```

可以用 pandas 等工具直接读：`pd.read_json(path, lines=True)`。
