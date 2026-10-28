"""预训练数据装箱打包：把多篇完整文档装进同一个 max_length 序列，减少 padding。

只依赖 numpy 和 tokenizer（不需要 torch），本地可以单独冒烟测试；训练时由 lm_dataset.PackedPretrainDataset 调用。
流程：逐行分词，每篇写成 [bos] + 文本 + [eos] 追加到 tokens.bin（uint16）；超过 max_length 的文档切成多段，
不再像 PretrainDataset 那样截断丢弃；再按固定种子打乱顺序做 best-fit 装箱：每段放进剩余空间最小且放得下的行，
放不下就开新行。短文档不会被切开，每行都从一篇文档（或长文档的一段）的开头开始。
结果缓存到 cache_dir，数据、max_length、种子、tokenizer 不变时直接复用，保证续训时每行内容和顺序不变。
"""
import os
import json
import time
import hashlib
from bisect import bisect_left, insort
import numpy as np

PACK_VERSION = 1


def cache_prefix(data_path, tokenizer, max_length, seed, cache_dir):
    """缓存文件名前缀。数据文件只取大小和首尾各 1MB 计算指纹：Colab 每次复制数据会改变 mtime，不能用 mtime"""
    size = os.path.getsize(data_path)
    h = hashlib.sha256(f'{PACK_VERSION}|{size}|{max_length}|{seed}|{len(tokenizer)}|'
                       f'{tokenizer.bos_token_id}|{tokenizer.eos_token_id}'.encode())
    with open(data_path, 'rb') as f:
        h.update(f.read(1 << 20))
        f.seek(max(0, size - (1 << 20)))
        h.update(f.read(1 << 20))
    name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(cache_dir, f'{name}_pack{max_length}_{h.hexdigest()[:12]}')


def tokenize_to_bin(data_path, tokenizer, max_length, tokens_path, batch_size=10000, log=print):
    """分词写入 tokens_path，返回每段的 (起点, 长度)。长文档按 max_length 切段，最后一段带 eos"""
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    seg_start, seg_len, offset, docs, t0 = [], [], 0, 0, time.time()
    with open(data_path, encoding='utf-8') as f, open(tokens_path, 'wb') as out:
        def flush(texts):
            nonlocal offset, docs
            if not texts: return
            chunks = []
            for ids in tokenizer(texts, add_special_tokens=False).input_ids:
                n = len(ids) + 2
                chunks.append(np.asarray([bos] + ids + [eos], dtype=np.uint16))
                for s in range(0, n, max_length):
                    seg_start.append(offset + s)
                    seg_len.append(min(max_length, n - s))
                offset += n
            out.write(np.concatenate(chunks).tobytes())
            docs += len(texts)
            log(f'[pack] 已分词 {docs} 篇，{offset} token，{time.time() - t0:.0f}s')

        batch = []
        for line in f:
            if not line.strip(): continue
            batch.append(str(json.loads(line)['text']))
            if len(batch) == batch_size:
                flush(batch)
                batch = []
        flush(batch)
    return np.asarray(seg_start, dtype=np.int64), np.asarray(seg_len, dtype=np.int32), docs


def best_fit_pack(seg_len, max_length, seed):
    """按种子打乱后逐段 best-fit 装箱，返回 (row_ptr, row_items)：第 r 行依次是 row_items[row_ptr[r]:row_ptr[r+1]] 这些段"""
    order = np.random.default_rng(seed).permutation(len(seg_len))
    row_of = np.empty(len(seg_len), dtype=np.int64)
    caps, bins_by_cap, n_rows = [], {}, 0  # caps：当前有空位的行的剩余容量（去重、升序）
    for pos, i in enumerate(order.tolist()):
        n = int(seg_len[i])
        k = bisect_left(caps, n)
        if k < len(caps):
            cap = caps[k]
            row = bins_by_cap[cap].pop()
            if not bins_by_cap[cap]:
                del bins_by_cap[cap]
                caps.pop(k)
        else:
            row, cap = n_rows, max_length
            n_rows += 1
        row_of[pos] = row
        cap -= n
        if cap > 0:
            if cap not in bins_by_cap:
                insort(caps, cap)
                bins_by_cap[cap] = []
            bins_by_cap[cap].append(row)
    # 按行分组，行内保持装入顺序
    by_row = np.argsort(row_of, kind='stable')
    row_items = order[by_row]
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(np.bincount(row_of, minlength=n_rows), out=row_ptr[1:])
    return row_ptr, row_items


def load_or_build(data_path, tokenizer, max_length, seed=42, cache_dir=None, build=True, log=print):
    """返回缓存前缀；缓存不存在且 build=True 时构建。index 文件最后原子写入，它存在即表示缓存完整"""
    if len(tokenizer) > np.iinfo(np.uint16).max:
        raise ValueError('词表超过 uint16 范围，不能用打包缓存')
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(data_path)), 'pack_cache')
    prefix = cache_prefix(data_path, tokenizer, max_length, seed, cache_dir)
    if os.path.exists(prefix + '.index.npz') or not build:
        return prefix
    os.makedirs(cache_dir, exist_ok=True)
    log(f'[pack] 构建打包缓存 {prefix}.*（只需一次，之后直接复用）')
    seg_start, seg_len, docs = tokenize_to_bin(data_path, tokenizer, max_length, prefix + '.tokens.bin.tmp', log=log)
    row_ptr, row_items = best_fit_pack(seg_len, max_length, seed)
    os.replace(prefix + '.tokens.bin.tmp', prefix + '.tokens.bin')
    with open(prefix + '.index.npz.tmp', 'wb') as f:
        np.savez(f, seg_start=seg_start, seg_len=seg_len, row_ptr=row_ptr, row_items=row_items)
    os.replace(prefix + '.index.npz.tmp', prefix + '.index.npz')
    total, rows = int(seg_len.sum()), len(row_ptr) - 1
    stats = {'docs': docs, 'segments': len(seg_len), 'split_docs_extra_segments': len(seg_len) - docs,
             'tokens': total, 'rows': rows, 'max_length': max_length, 'fill_rate': round(total / (rows * max_length), 4)}
    with open(prefix + '.stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f'[pack] 完成：{stats}')
    return prefix


class PackedRows:
    """按行读取打包结果；tokens.bin 用 memmap 懒加载，DataLoader 每个 worker 各自打开"""
    def __init__(self, prefix):
        self.prefix = prefix
        index = np.load(prefix + '.index.npz')
        self.seg_start, self.seg_len = index['seg_start'], index['seg_len']
        self.row_ptr, self.row_items = index['row_ptr'], index['row_items']
        self._tokens = None

    def __len__(self):
        return len(self.row_ptr) - 1

    def row(self, i):
        if self._tokens is None:
            self._tokens = np.memmap(self.prefix + '.tokens.bin', dtype=np.uint16, mode='r')
        segs = self.row_items[self.row_ptr[i]:self.row_ptr[i + 1]]
        return np.concatenate([self._tokens[self.seg_start[s]:self.seg_start[s] + self.seg_len[s]] for s in segs])
