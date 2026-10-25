import os
import json
import argparse

# 训练脚本通过 trainer_utils.MetricsLogger 写入 {save_dir}/{name}_metrics.jsonl，这里读取并画图。
# 用法: python plot_metrics.py ../out/pretrain_metrics.jsonl [--smooth 0.9] [--out x.png]
# Colab notebook 里直接看图: plot_metrics('../out/pretrain_metrics.jsonl', show=True)
META_KEYS = {"epoch", "step", "iters"}

# 单色线 + 淡色原始曲线；浅色底，文字用墨色而非线条色
LINE, SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_metrics(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except ValueError: continue  # 跳过断连时写了一半的行
    if not rows: raise SystemExit(f"{path} 里没有可用的指标记录")
    rows.sort(key=lambda r: (r["epoch"], r["step"]))
    return rows


def ema(values, smooth):
    """指数滑动平均，smooth=0 时原样返回"""
    out, prev = [], None
    for v in values:
        prev = v if prev is None else smooth * prev + (1 - smooth) * v
        out.append(prev)
    return out


def plot_metrics(path, out=None, smooth=0.9, show=False):
    import matplotlib
    if not show: matplotlib.use("Agg")  # 无显示环境（训练进程/Colab 脚本）下只存图
    import matplotlib.pyplot as plt

    rows = load_metrics(path)
    xs = [r["epoch"] * r["iters"] + r["step"] for r in rows]
    names = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in META_KEYS]
    series = {k: [(x, r[k]) for x, r in zip(xs, rows) if k in r] for k in names}
    series = {k: v for k, v in series.items() if any(y != 0 for _, y in v)}  # 丢掉全 0 的指标（如 dense 模型的 aux_loss）
    if not series: raise SystemExit(f"{path} 里没有非零指标可画")

    cols = min(3, len(series))
    nrows = -(-len(series) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 3.2 * nrows), squeeze=False, facecolor=SURFACE)
    for ax, (name, pts) in zip(axes.flat, series.items()):
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        s = max(0.0, min(smooth, 1 - 20 / len(y)))  # 点数太少时 EMA 严重滞后，按点数自动减弱，约 20 个窗口才用满 smooth
        ax.set_facecolor(SURFACE)
        ax.plot(x, y, color=LINE, linewidth=1 if s > 0 else 2, alpha=0.3 if s > 0 else 1)
        if s > 0: ax.plot(x, ema(y, s), color=LINE, linewidth=2)
        ax.set_title(f"{name}  (last {y[-1]:.4g})", loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_xlabel("step", fontsize=8, color=INK_2)
    for ax in list(axes.flat)[len(series):]: ax.set_visible(False)
    fig.suptitle(os.path.basename(path).replace("_metrics.jsonl", ""), x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()

    out = out or os.path.splitext(path)[0] + ".png"
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    if show: plt.show()
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制 MiniMind 训练指标曲线")
    parser.add_argument("path", type=str, help="训练产生的 *_metrics.jsonl，例如 ../out/pretrain_metrics.jsonl")
    parser.add_argument("--out", type=str, default=None, help="图片保存路径，默认与 jsonl 同名的 .png")
    parser.add_argument("--smooth", type=float, default=0.9, help="滑动平均系数，0 表示不平滑，越大越平滑")
    parser.add_argument("--show", action="store_true", help="弹窗显示图（本地有 GUI 时使用）")
    args = parser.parse_args()
    print(f"已保存: {plot_metrics(args.path, args.out, args.smooth, args.show)}")
