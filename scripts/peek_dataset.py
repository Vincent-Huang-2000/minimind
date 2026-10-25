import os
import sys
import json
import argparse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_dataset import DATASET_FILES, HF_REPO, resolve_files


def peek_file(name, n, endpoint, out_dir):
    """流式读取远程 jsonl 的前 n 行写入本地，读够就断开连接，不会下载整个文件"""
    url = f"{endpoint}/datasets/{HF_REPO}/resolve/main/{name}"
    path = os.path.join(out_dir, name)
    count = 0
    with urllib.request.urlopen(urllib.request.Request(url), timeout=60) as resp, open(path, 'wb') as f:
        for line in resp:
            f.write(line)
            count += 1
            if count >= n: break
    return path, count


def show_structure(path, show=1):
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= show: break
            sample = json.loads(line)
            print(f"  字段: {list(sample.keys())}")
            print("  " + json.dumps(sample, ensure_ascii=False, indent=2).replace('\n', '\n  ')[:1500])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="只下载 MiniMind 数据集前 N 行，用于查看数据结构")
    parser.add_argument('--files', nargs='+', default=['all'], help="文件名（可省略.jsonl后缀），默认 all")
    parser.add_argument('-n', '--lines', default=20, type=int, help="每个文件保留的行数")
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'), type=str, help="HF 地址，国内可设 https://hf-mirror.com")
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'peek'), type=str, help="保存目录")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name in resolve_files(args.files):
        print(f"== {name} ({DATASET_FILES[name][0]}) ==")
        path, count = peek_file(name, args.lines, args.endpoint.rstrip('/'), out_dir)
        print(f"  已保存前 {count} 行 -> {path}")
        show_structure(path)
