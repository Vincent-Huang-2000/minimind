import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import argparse
import numpy as np
from scripts.plot_metrics import load_metrics, ema, SURFACE, INK, INK_2, MUTED, GRID, AXIS

# 把打包 / 不打包两次预训练的 *_metrics.jsonl 画在同一张图上。两次训练每步的有效 token 不同，
# 横轴默认按"看过的有效 token 数"对齐：全局步数 × batch_size × 每行有效 token（来自 data_efficiency.py 的 efficiency.json）。
# 注意训练 loss 只能看趋势：打包时同一行的文档能互相看到、长文档的后半段也参与训练，两边的 token 构成不同，
# 质量结论以 eval_heldout_loss.py 的留出集 loss 为准。
# 用法（仓库根目录）:
#   python experiment/pack_compare/compare_metrics.py --nopack out/pretrain_nopack_metrics.jsonl --pack out/pretrain_pack_metrics.jsonl
# 图里的文字用英文：Colab 默认没有中文字体，中文会显示成方框
RUNS = (('nopack', '不打包', 'no pack', '#2a78d6'), ('pack', '打包', 'pack', '#eb6834'))  # 分类色板前两位，已用 dataviz 校验器验证
PANELS = (('logits_loss', 'loss (w/o aux_loss)'), ('learning_rate', 'learning rate'))


def load_run(path, mode, eff, x_axis):
    rows = load_metrics(path)
    m = eff[mode]
    if rows[0]['iters'] != m['steps_per_epoch']:
        print(f"[警告] {path} 每个 epoch {rows[0]['iters']} 步，efficiency.json 算出的{mode}是 {m['steps_per_epoch']} 步；"
              f"数据、max_seq_len、batch_size 或 seed 与训练不一致时 token 换算不准")
    steps = np.array([r['epoch'] * r['iters'] + r['step'] for r in rows], dtype=float)
    x = steps * eff['batch_size'] * m['tokens_per_row'] if x_axis == 'tokens' else steps
    return x, {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k, _ in PANELS}


def smoothed(y, smooth):
    s = max(0.0, min(smooth, 1 - 20 / len(y)))  # 与 plot_metrics 相同：点数太少时减弱平滑
    return np.array(ema(list(y), s)) if s > 0 else y


def fmt_tokens(v):
    for unit, d in (('B', 1e9), ('M', 1e6), ('K', 1e3)):
        if abs(v) >= d: return f'{v / d:.3g}{unit}'
    return f'{v:.0f}'


def summary(runs, x_axis, smooth):
    """在两次训练共同覆盖的范围内取几个对齐点，比较平滑后的 loss"""
    end = min(x[-1] for x, _ in runs.values())
    unit = 'token' if x_axis == 'tokens' else 'step'
    print(f'按{unit}对齐的训练 loss（EMA 平滑后插值）：')
    print(f"{unit:>12}" + ''.join(f'{label:>10}' for mode, label, _, _ in RUNS if mode in runs) + f"{'差值(打包-不打包)':>18}")
    for frac in (0.25, 0.5, 0.75, 1.0):
        at = end * frac
        vals = {mode: float(np.interp(at, x, smoothed(ys['logits_loss'], smooth))) for mode, (x, ys) in runs.items()}
        diff = f"{vals['pack'] - vals['nopack']:>+18.4f}" if len(vals) == 2 else ''
        print(f'{fmt_tokens(at) if x_axis == "tokens" else int(at):>12}' + ''.join(f'{vals[m]:>10.4f}' for m, _, _, _ in RUNS if m in vals) + diff)
    for mode, label, _, _ in RUNS:
        if mode in runs:
            x, ys = runs[mode]
            print(f'{label}：共 {fmt_tokens(x[-1]) if x_axis == "tokens" else int(x[-1])} {unit}，最后一次记录的 loss {ys["logits_loss"][-1]:.4f}')


def plot(runs, x_axis, smooth, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6 * len(PANELS), 3.6), squeeze=False, facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for mode, _, label, color in RUNS:
            if mode not in runs: continue
            x, ys = runs[mode]
            y = ys[key]
            ys_s = smoothed(y, smooth) if key != 'learning_rate' else y
            if key != 'learning_rate': ax.plot(x, y, color=color, linewidth=1, alpha=0.25)
            ax.plot(x, ys_s, color=color, linewidth=2, label=label)
            ax.annotate(label, (x[-1], ys_s[-1]), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK_2)
        ax.set_title(title, loc='left', fontsize=10, color=INK)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'): ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_xlabel('effective tokens seen' if x_axis == 'tokens' else 'global step', fontsize=8, color=INK_2)
        if x_axis == 'tokens': ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_tokens(v)))
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.suptitle(f'pretrain: pack vs no pack (aligned by {"effective tokens" if x_axis == "tokens" else "step"})', x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='对比打包 / 不打包两次预训练的训练曲线')
    parser.add_argument('--nopack', type=str, required=True, help='不打包那次训练的 *_metrics.jsonl')
    parser.add_argument('--pack', type=str, required=True, help='打包那次训练的 *_metrics.jsonl')
    parser.add_argument('--work_dir', type=str, default='out/pack_compare', help='对比实验的输出目录（读 efficiency.json，写对比图）')
    parser.add_argument('--efficiency', type=str, default=None, help='data_efficiency.py 生成的文件，默认 {work_dir}/efficiency.json')
    parser.add_argument('--x', type=str, default='tokens', choices=['tokens', 'step'], help='横轴：tokens=有效 token 数（推荐），step=全局步数')
    parser.add_argument('--smooth', type=float, default=0.9, help='滑动平均系数，0 表示不平滑')
    parser.add_argument('--out', type=str, default=None, help='图片路径，默认 {work_dir}/compare_{x}.png')
    args = parser.parse_args()

    with open(args.efficiency or os.path.join(args.work_dir, 'efficiency.json'), encoding='utf-8') as f:
        eff = json.load(f)
    runs = {mode: load_run(getattr(args, mode), mode, eff, args.x) for mode, _, _, _ in RUNS}
    summary(runs, args.x, args.smooth)
    os.makedirs(args.work_dir, exist_ok=True)
    out = args.out or os.path.join(args.work_dir, f'compare_{args.x}.png')
    try:
        print(f'已保存: {plot(runs, args.x, args.smooth, out)}')
    except ImportError as e:
        print(f'[plot] 缺少 matplotlib，跳过出图: {e}')
