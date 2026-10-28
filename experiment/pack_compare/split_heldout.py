import os
import json
import random
import hashlib
import argparse

# 从预训练数据里随机留出 N 篇作为留出集，其余写成训练集；打包 / 不打包两次训练都只用训练集，
# 留出集交给 eval_heldout_loss.py 比较 loss。和留出集文本完全相同的训练行也一并去掉，避免重复样本泄漏。
# 流式读两遍，不把整个文件读进内存；原始行逐字节写出，不重新序列化。
# 用法（仓库根目录）: python experiment/pack_compare/split_heldout.py --data dataset/pretrain_t2t_mini.jsonl --n 2000


def text_key(line):
    return hashlib.md5(str(json.loads(line)['text']).encode('utf-8')).digest()


def split(data, n, seed, out_dir):
    with open(data, 'rb') as f:
        total = sum(1 for line in f if line.strip())
    if n >= total: raise SystemExit(f'--n {n} 不小于数据行数 {total}')
    picked = set(random.Random(seed).sample(range(total), n))

    name = os.path.splitext(os.path.basename(data))[0]
    out_dir = out_dir or os.path.dirname(os.path.abspath(data))
    train_path = os.path.join(out_dir, f'{name}_train.jsonl')
    heldout_path = os.path.join(out_dir, f'{name}_heldout.jsonl')

    heldout_keys, i = set(), 0
    with open(data, 'rb') as f, open(heldout_path, 'wb') as out:
        for line in f:
            if not line.strip(): continue
            if i in picked:
                out.write(line if line.endswith(b'\n') else line + b'\n')
                heldout_keys.add(text_key(line))
            i += 1

    kept = dup = i = 0
    with open(data, 'rb') as f, open(train_path, 'wb') as out:
        for line in f:
            if not line.strip(): continue
            if i not in picked:
                if text_key(line) in heldout_keys: dup += 1
                else:
                    out.write(line if line.endswith(b'\n') else line + b'\n')
                    kept += 1
            i += 1

    print(f'共 {total} 行：留出 {n} 行 -> {heldout_path}')
    print(f'训练集 {kept} 行 -> {train_path}（去掉与留出集重复的 {dup} 行）')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='把预训练数据切成训练集和留出集')
    parser.add_argument('--data', type=str, default='dataset/pretrain_t2t_mini.jsonl', help='预训练数据 jsonl（每行含 text）')
    parser.add_argument('--n', type=int, default=2000, help='留出的篇数')
    parser.add_argument('--seed', type=int, default=42, help='抽样种子')
    parser.add_argument('--out_dir', type=str, default=None, help='输出目录，默认与数据文件同目录（dataset/ 下的 jsonl 已被 .gitignore 忽略）')
    args = parser.parse_args()
    split(args.data, args.n, args.seed, args.out_dir)
