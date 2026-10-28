import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import argparse
import numpy as np
from transformers import AutoTokenizer
from dataset.pretrain_packing import load_or_build, PackedRows

# 统计同一份数据打包 / 不打包时的行数、每行有效 token、每个 epoch 的步数，写入 {work_dir}/efficiency.json，
# 供 compare_metrics.py 把训练曲线的横轴从 step 换算成"看过的有效 token 数"。
# 打包缓存用 dataset/pretrain_packing.py 构建：--pack_cache_dir、--max_seq_len、--seed 与训练一致时，
# 训练直接复用这里建好的缓存，不会重复分词。不打包的数字从同一份分词结果推出，不再分词第二遍。
# 只依赖 numpy 和 transformers，不需要 torch。
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def efficiency(data, max_seq_len, batch_size, seed, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    prefix = load_or_build(data, tokenizer, max_seq_len, seed, cache_dir)
    with open(prefix + '.stats.json', encoding='utf-8') as f:
        stats = json.load(f)
    rows = PackedRows(prefix)
    tokens = np.memmap(prefix + '.tokens.bin', dtype=np.uint16, mode='r')

    # 每篇文档以 bos 开头，长文档切出的后续段不以 bos 开头；文档第一段的长度 = min(文档长度, max_seq_len)，
    # 正好是 PretrainDataset 截断后保留下来的 token 数
    first = tokens[rows.seg_start] == tokenizer.bos_token_id
    docs = int(first.sum())
    if docs != stats['docs']:
        print(f'[警告] 以 bos 开头的段 {docs} 个，与文档数 {stats["docs"]} 不一致（正文里可能含 bos 字面量），不打包的统计会有偏差')
    nopack_tokens = int(rows.seg_len[first].sum())
    total = stats['tokens']

    def mode(n_rows, n_tokens):
        return {'rows': n_rows, 'tokens': n_tokens, 'tokens_per_row': n_tokens / n_rows,
                'fill_rate': n_tokens / (n_rows * max_seq_len), 'steps_per_epoch': -(-n_rows // batch_size)}

    return {
        'data': os.path.abspath(data), 'max_seq_len': max_seq_len, 'batch_size': batch_size, 'seed': seed,
        'pack_cache': prefix, 'docs': stats['docs'], 'total_tokens': total,
        'nopack': {**mode(stats['docs'], nopack_tokens), 'truncated_tokens': total - nopack_tokens,
                   'truncated_ratio': (total - nopack_tokens) / total},
        'pack': mode(stats['rows'], total),
    }


def report(e):
    n, p = e['nopack'], e['pack']
    print(f"数据 {e['data']}，{e['docs']} 篇，共 {e['total_tokens']} token，max_seq_len={e['max_seq_len']}，batch_size={e['batch_size']}")
    print(f"{'':8}{'行数':>12}{'有效token':>14}{'每行有效':>10}{'填充率':>9}{'每epoch步数':>12}")
    for name, m in (('不打包', n), ('打包', p)):
        print(f"{name:8}{m['rows']:>12}{m['tokens']:>14}{m['tokens_per_row']:>10.1f}{m['fill_rate']:>9.1%}{m['steps_per_epoch']:>12}")
    print(f"不打包截掉 {n['truncated_tokens']} token（{n['truncated_ratio']:.1%}）；"
          f"打包后每 epoch 步数为原来的 {p['steps_per_epoch'] / n['steps_per_epoch']:.1%}，"
          f"每步有效 token 是原来的 {p['tokens_per_row'] / n['tokens_per_row']:.2f} 倍")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='统计预训练数据打包 / 不打包的数据利用率')
    parser.add_argument('--data', type=str, default='dataset/pretrain_t2t_mini_train.jsonl', help='训练用的预训练数据（split_heldout.py 切出的训练集）')
    parser.add_argument('--max_seq_len', type=int, default=340, help='与训练的 --max_seq_len 一致')
    parser.add_argument('--batch_size', type=int, default=32, help='与训练的 --batch_size 一致，用于计算每个 epoch 的步数')
    parser.add_argument('--seed', type=int, default=42, help='与训练的 --seed 一致（决定打包缓存）')
    parser.add_argument('--pack_cache_dir', type=str, default=None, help='与训练的 --pack_cache_dir 一致，默认数据文件同目录的 pack_cache/')
    parser.add_argument('--work_dir', type=str, default='out/pack_compare', help='对比实验的输出目录，Colab 上指向 Drive')
    args = parser.parse_args()

    e = efficiency(args.data, args.max_seq_len, args.batch_size, args.seed, args.pack_cache_dir)
    report(e)
    os.makedirs(args.work_dir, exist_ok=True)
    path = os.path.join(args.work_dir, 'efficiency.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(e, f, ensure_ascii=False, indent=2)
    print(f'已保存: {path}')
