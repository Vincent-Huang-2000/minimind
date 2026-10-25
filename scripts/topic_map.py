import os
import re
import csv
import sys
import json
import math
import time
import argparse
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_probe import load_env_file, make_client, chat_json, run_stage, read_jsonl, write_jsonl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAGES = ['embed', 'cluster', 'keywords', 'label', 'report']
# all 不含 label：label 调 API 会产生费用，需要单独运行
ALL_STAGES = ['embed', 'cluster', 'keywords', 'report']
# 这些参数决定每条样本的向量，续跑和后续阶段都以 meta.json 里记录的为准
EMBED_KEYS = ['field', 'text_mode', 'max_chars', 'model', 'prefix', 'max_seq_length', 'shard_size']

# 统计关键词时去掉的高频虚词和问答套话
STOPWORDS = set("""
我们 你们 他们 她们 它们 自己 一个 一些 一下 一种 这个 那个 这些 那些 这样 那样 这里 那里 这是 就是 也是 还是 或者 以及 而且 并且 但是 因为 所以 如果 虽然 然后 那么
什么 怎么 怎样 如何 为什么 是否 是不是 哪些 哪个 多少 可以 能够 需要 应该 可能 已经 非常 比较 没有 不是 不会 不能 时候 进行 通过 使用 根据 对于 关于 其中 之后 之前 以上 以下 同时
问题 回答 请问 帮我 给我 告诉 一段 下面 以下 内容 文本 句子 用户 助手 例如 比如 包括 主要 一般 通常 任何 所有 每个 很多 许多 其他 各种 进一步 不同 相关 具有 提供 方面 情况 方式
the and for with that this these those are was were you your have has had from which what when where who why how can could will would should not but they them their there then than
into about also more most some such only other its our out use used using one two all any each may might must does did been being just like very here make made need answer question please
""".split())
TOKEN_RE = re.compile(r'[一-鿿]{2,}|[a-z][a-z+#\-]{2,}')

TOPIC_PROMPT = """下面是一个训练数据集中被聚成同一类的若干条样本（已截断），以及这一类的高频关键词。请概括这一类样本的主题。
关键词：{keywords}
样本：
{examples}

输出 JSON：{{"name": "主题名", "description": "一句话说明"}}

要求：
1. name 是不超过 12 个字的中文短语，说清楚涉及什么知识或任务（如“Python 列表操作”“古诗词赏析”），不要用“杂项”“综合问题”这类空泛的词。
2. description 不超过 50 个字，可以说明样本的形式（问答、翻译、写作、代码、数学题等）和语言。"""

GROUP_PROMPT = """下面是训练数据集中同一个大类下的若干个小主题（括号里是它在这个大类中的样本占比）：
{topics}

请为这个大类起名，输出 JSON：{{"name": "大类名", "description": "一句话说明"}}

要求：
1. name 是不超过 8 个字的中文短语，能概括占比大的那些小主题，不要用“杂项”“综合”这类空泛的词。
2. description 不超过 50 个字。"""


def extract_text(row, cfg):
    """把一行 jsonl 转成用于向量化的文本：
    预训练数据取 text；SFT / RL 数据取 conversations（DPO 取 chosen），默认只拼接 user 的发言；题库取 question。"""
    value = row.get(cfg['field']) if cfg['field'] else next((row[k] for k in ('conversations', 'chosen', 'text', 'question') if k in row), None)
    if value is None: raise SystemExit(f'无法识别的数据格式，字段为 {list(row)}，请用 --field 指定')
    if isinstance(value, list):
        roles = ('user',) if cfg['text_mode'] == 'user' else ('user', 'assistant')
        value = '\n'.join(str(m.get('content') or '') for m in value if isinstance(m, dict) and m.get('role') in roles)
    return str(value).strip()[:cfg['max_chars']]


def iter_lines(path, limit=0):
    """逐行读取（不整体载入内存），只计非空行，第 i 个非空行就是第 i 条样本。"""
    with open(path, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            if not line.strip(): continue
            if limit and i >= limit: return
            yield i, line
            i += 1


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save_npy(path, arr):
    # 先写临时文件再改名：Colab 断连时不会留下写了一半的文件
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f: np.save(f, arr)
    os.replace(tmp, path)


def load_meta(files):
    if not os.path.exists(files['meta']): raise SystemExit(f'找不到 {files["meta"]}，请先运行 --stage embed')
    meta = read_json(files['meta'])
    if not meta.get('done'): raise SystemExit('向量化还没完成，请先用 --stage embed --resume 1 跑完')
    return meta


def shard_paths(files):
    d = files['emb_dir']
    return sorted(os.path.join(d, p) for p in os.listdir(d) if re.fullmatch(r'part_\d{5}\.npy', p)) if os.path.isdir(d) else []


def resolve_device(args):
    import torch
    return args.device or ('cuda' if torch.cuda.is_available() else 'cpu')


def stage_embed(args, files):
    from sentence_transformers import SentenceTransformer
    cfg = {k: getattr(args, k) for k in EMBED_KEYS}
    part = lambda i: os.path.join(files['emb_dir'], f'part_{i:05d}.npy')
    shards, n_shards = shard_paths(files), 0
    if args.resume and shards:
        old = read_json(files['meta']) if os.path.exists(files['meta']) else {}
        changed = [k for k in EMBED_KEYS if old.get(k) != cfg[k]]
        if changed: raise SystemExit(f'续跑参数与已有向量不一致：{changed}，改回原来的值，或去掉 --resume 1 重新向量化')
        # 只保留从头连续、写满的分片；最后一个不满的分片（上次的结尾或 --limit 冒烟的结果）重算
        while n_shards < len(shards) and shards[n_shards] == part(n_shards) and np.load(shards[n_shards], mmap_mode='r').shape[0] == args.shard_size:
            n_shards += 1
    for p in shards[n_shards:]: os.remove(p)
    os.makedirs(files['emb_dir'], exist_ok=True)
    start = n_shards * args.shard_size
    write_json(files['meta'], {**cfg, 'data': os.path.abspath(args.data), 'done': False})

    device = resolve_device(args)
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    if device.startswith('cuda'): model.half()
    print(f'[向量化] {args.data} -> {files["emb_dir"]}，模型 {args.model}（{device}），已有 {start} 条，从第 {start} 条继续')

    buf, n, st, dim = [], start, time.time(), None

    def flush():
        nonlocal dim, n_shards
        emb = model.encode(buf, batch_size=args.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        dim = emb.shape[1]
        save_npy(part(n_shards), emb.astype(np.float16))
        print(f'[向量化] 第 {n_shards} 片完成，累计 {n} 条，{(n - start) / (time.time() - st):.0f} 条/秒')
        n_shards += 1
        buf.clear()

    for i, line in iter_lines(args.data, args.limit):
        if i < start: continue
        buf.append(args.prefix + extract_text(json.loads(line), cfg))
        n = i + 1
        if len(buf) == args.shard_size: flush()
    if buf: flush()
    if dim is None: dim = np.load(shard_paths(files)[0], mmap_mode='r').shape[1]
    write_json(files['meta'], {**cfg, 'data': os.path.abspath(args.data), 'data_bytes': os.path.getsize(args.data), 'done': True, 'rows': n, 'dim': int(dim)})
    print(f'[向量化] 完成：共 {n} 条，维度 {dim}，耗时 {time.time() - st:.0f}s')


def assign(x, c, chunk=65536):
    """向量已归一化，点积就是余弦相似度，分给最相似的中心。"""
    import torch
    labels, scores = [], []
    for i in range(0, len(x), chunk):
        s, l = (x[i:i + chunk] @ c.T).max(1)
        labels.append(l)
        scores.append(s)
    return torch.cat(labels), torch.cat(scores)


def kmeans(x, k, iters, seed):
    """球面 K-Means：k-means++ 初始化（在至多 64k 个点上抽），按余弦相似度分配，中心取均值后归一化，空簇随机换一个点。"""
    import torch
    g = torch.Generator().manual_seed(seed)
    pool = x[torch.randperm(len(x), generator=g)[:min(len(x), 64 * k)].to(x.device)]
    c = [pool[torch.randint(len(pool), (1,), generator=g).item()]]
    d = (1 - pool @ c[0]).clamp_min(0)
    for _ in range(k - 1):
        p = (d ** 2).cpu() + 1e-12
        c.append(pool[torch.multinomial(p / p.sum(), 1, generator=g).item()])
        d = torch.minimum(d, (1 - pool @ c[-1]).clamp_min(0))
    c = torch.stack(c)
    for it in range(iters):
        labels, scores = assign(x, c)
        counts = torch.bincount(labels, minlength=k)
        c = torch.nn.functional.normalize(torch.zeros_like(c).index_add_(0, labels, x), dim=1)
        empty = (counts == 0).nonzero().flatten()
        if len(empty): c[empty] = x[torch.randint(len(x), (len(empty),), generator=g).to(x.device)]
        if it % 10 == 0 or it == iters - 1: print(f'  迭代 {it + 1}/{iters}：平均相似度 {scores.mean().item():.4f}，空簇 {len(empty)}')
    return c


def sort_by_size(labels, c):
    """把簇按大小重新编号，0 号最大。"""
    counts = np.bincount(labels, minlength=len(c))
    order = np.argsort(-counts, kind='stable')
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels], c[order]


def stage_cluster(args, files):
    import torch
    meta = load_meta(files)
    shards = [np.load(p, mmap_mode='r') for p in shard_paths(files)]
    n = sum(len(s) for s in shards)
    if n < args.k: raise SystemExit(f'样本数 {n} 少于主题数 --k {args.k}')
    device = resolve_device(args)

    # 1) 在随机抽样上训练 K-Means
    idx = np.sort(np.random.default_rng(args.seed).choice(n, min(n, args.train_size), replace=False))
    offsets = np.cumsum([0] + [len(s) for s in shards])
    train = np.concatenate([s[idx[(idx >= lo) & (idx < hi)] - lo] for s, lo, hi in zip(shards, offsets[:-1], offsets[1:])])
    print(f'[聚类] 共 {n} 条，抽 {len(train)} 条训练 k={args.k} 的 K-Means（{device}）')
    st = time.time()
    x = torch.from_numpy(train.astype(np.float32)).to(device)
    c = kmeans(x, args.k, args.iters, args.seed)
    del x

    # 2) 全量分配：逐片读取，内存里只放一片
    labels, scores = [], []
    for s in shards:
        l, sc = assign(torch.from_numpy(np.asarray(s, dtype=np.float32)).to(device), c)
        labels.append(l.cpu().numpy().astype(np.int32))
        scores.append(sc.cpu().numpy().astype(np.float16))
    labels, scores = np.concatenate(labels), np.concatenate(scores)
    labels, c = sort_by_size(labels, c.cpu().numpy())

    # 3) 大类：对主题中心再聚一次
    k_top = min(args.k_top, args.k)
    ct = torch.from_numpy(c).to(device)
    gc = kmeans(ct, k_top, args.iters, args.seed)
    groups = assign(ct, gc)[0].cpu().numpy()
    group_sizes = np.bincount(groups, weights=np.bincount(labels, minlength=args.k), minlength=k_top)
    remap = np.empty(k_top, dtype=np.int64)
    remap[np.argsort(-group_sizes, kind='stable')] = np.arange(k_top)

    save_npy(files['labels'], labels.astype(np.int32))
    save_npy(files['scores'], scores)
    save_npy(files['centroids'], c.astype(np.float32))
    save_npy(files['groups'], remap[groups].astype(np.int32))
    counts = np.bincount(labels, minlength=args.k)
    print(f'[聚类] 完成，耗时 {time.time() - st:.0f}s：{args.k} 个主题，{k_top} 个大类；主题大小 最大 {counts.max()} / 中位 {int(np.median(counts))} / 最小 {counts.min()}，'
          f'样本与所属中心的平均相似度 {scores.astype(np.float32).mean():.4f}')


def tokenize(text):
    import jieba
    return {w for w in (t.lower() for t in jieba.lcut(text)) if TOKEN_RE.fullmatch(w) and w not in STOPWORDS}


def ctfidf(counters, n):
    """c-TF-IDF（与 BERTopic 相同）：tf × log(1 + 每类平均词数 / 该词在所有类中的总频次)。
    tf 按“包含该词的样本数”计，一条长样本反复出现同一个词不会把它顶上来。"""
    total = Counter()
    for c in counters: total.update(c)
    avg = sum(total.values()) / max(len(counters), 1)
    return [[w for w, _ in sorted(((w, tf * math.log(1 + avg / total[w])) for w, tf in c.items()), key=lambda x: -x[1])[:n]] for c in counters]


def stage_keywords(args, files):
    import jieba, logging
    jieba.setLogLevel(logging.WARNING)
    meta = load_meta(files)
    data = args.data or meta['data']
    # 行号要和向量一一对应，数据文件换过（大小不同）就不能再用
    if os.path.getsize(data) != meta['data_bytes']: raise SystemExit(f'{data} 的大小与向量化时不同，不是同一个数据文件')
    labels, scores = np.load(files['labels']), np.load(files['scores']).astype(np.float32)
    groups = np.load(files['groups'])
    k, n = len(groups), len(labels)
    counts = np.bincount(labels, minlength=k)
    order = np.lexsort((-scores, labels))  # 按主题分组，组内按与中心的相似度降序
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(args.seed)
    tops, samples, want = [], [], set()
    for t in range(k):
        members = order[starts[t]:starts[t] + counts[t]]
        tops.append(members[:args.n_examples].tolist())
        samples.append(rng.choice(members, min(len(members), args.kw_samples), replace=False).tolist())
        want.update(tops[-1], samples[-1])

    # 再读一遍原始数据，只取被选中的样本
    print(f'[关键词] 从 {data} 读取 {len(want)} 条样本（每个主题最相似的 {args.n_examples} 条 + 随机 {args.kw_samples} 条）')
    texts, seen = {}, 0
    for i, line in iter_lines(data, meta['rows']):
        seen = i + 1
        if i in want: texts[i] = extract_text(json.loads(line), meta)
    if seen != n: raise SystemExit(f'{data} 只有 {seen} 条，与向量数 {n} 不一致，数据文件可能换过')

    st = time.time()
    topic_df = [Counter(w for r in s for w in tokenize(texts[r])) for s in samples]
    group_df = [sum((topic_df[t] for t in range(k) if groups[t] == g), Counter()) for g in range(groups.max() + 1)]
    topic_kw, group_kw = ctfidf(topic_df, args.n_keywords), ctfidf(group_df, args.n_keywords)
    print(f'[关键词] 分词和 c-TF-IDF 耗时 {time.time() - st:.0f}s')

    mean_scores = np.bincount(labels, weights=scores, minlength=k) / np.maximum(counts, 1)
    clip = lambda r: {'row': int(r), 'score': round(float(scores[r]), 4), 'text': texts[r][:300]}
    topics = [{
        'topic': t, 'group': int(groups[t]), 'size': int(counts[t]), 'ratio': round(counts[t] / n, 6), 'score': round(float(mean_scores[t]), 4),
        'keywords': topic_kw[t], 'examples': [clip(r) for r in tops[t]],
        'random_examples': [clip(r) for r in samples[t] if r not in tops[t]][:args.n_examples],
    } for t in range(k)]
    group_rows = []
    for g in range(len(group_df)):
        members = [t for t in range(k) if groups[t] == g]
        size = int(counts[members].sum())
        group_rows.append({'group': g, 'size': size, 'ratio': round(size / n, 6), 'topics': members, 'keywords': group_kw[g]})
    write_jsonl(files['topics'], topics)
    write_jsonl(files['group_list'], group_rows)
    print(f'[关键词] -> {files["topics"]}，{files["group_list"]}')


def check_label(data):
    if not isinstance(data.get('name'), str) or not data['name'].strip(): raise ValueError('没有 name')
    return {'name': data['name'].strip(), 'description': str(data.get('description', '')).strip()}


def stage_label(args, files):
    """调 DeepSeek 给每个主题起名，再根据小主题的名字给大类起名。"""
    if not os.path.exists(files['topics']): raise SystemExit(f'找不到 {files["topics"]}，请先运行 --stage keywords')
    load_env_file(args.env_file)
    api = argparse.Namespace(**{**vars(args), 'model': args.llm_model})
    topics, group_rows = read_jsonl(files['topics']), read_jsonl(files['group_list'])

    def topic_worker(client, api, t):
        examples = '\n'.join(f'{j}. {e["text"][:200]}'.replace('\n', ' ') for j, e in enumerate(t['examples'] + t['random_examples'], 1))
        label, usage = chat_json(client, api, TOPIC_PROMPT.format(keywords='、'.join(t['keywords']), examples=examples), args.json_max_tokens, check_label)
        return {'topic': t['topic'], **label}, usage
    run_stage(api, topics, 'topic', topic_worker, files['topic_labels'], '主题命名', lambda r: r['name'])

    names = {r['topic']: r['name'] for r in read_jsonl(files['topic_labels'])} if os.path.exists(files['topic_labels']) else {}
    by_id = {t['topic']: t for t in topics}

    def group_worker(client, api, g):
        lines = [f'- {names.get(t) or " / ".join(by_id[t]["keywords"][:4])}（{by_id[t]["size"] / max(g["size"], 1):.0%}）' for t in g['topics'][:40]]
        label, usage = chat_json(client, api, GROUP_PROMPT.format(topics='\n'.join(lines)), args.json_max_tokens, check_label)
        return {'group': g['group'], **label}, usage
    run_stage(api, group_rows, 'group', group_worker, files['group_labels'], '大类命名', lambda r: r['name'])


def md_cell(text):
    return str(text).replace('\n', ' ').replace('|', '\\|')


def stage_report(args, files):
    meta = load_meta(files)
    topics, group_rows = read_jsonl(files['topics']), read_jsonl(files['group_list'])
    load_names = lambda path, key: {r[key]: r for r in read_jsonl(path)} if os.path.exists(path) else {}
    topic_names, group_names = load_names(files['topic_labels'], 'topic'), load_names(files['group_labels'], 'group')
    tname = lambda t: topic_names.get(t['topic'], {}).get('name') or ' / '.join(t['keywords'][:4])
    gname = lambda g: group_names.get(g['group'], {}).get('name') or ' / '.join(g['keywords'][:4])
    by_id = {t['topic']: t for t in topics}
    n = meta['rows']
    max_ratio = max(g['ratio'] for g in group_rows) or 1
    bar = lambda r: '█' * max(1, round(r / max_ratio * 20))
    pct = lambda r: f'{r:.2%}'

    lines = [
        f'# 主题分布：{os.path.basename(meta["data"])}', '',
        f'- 数据：`{meta["data"]}`，共 {n:,} 条；文本：{meta["field"] or "自动识别"}（对话取 {meta["text_mode"]}），截断到 {meta["max_chars"]} 字',
        f'- 向量模型：`{meta["model"]}`；{len(topics)} 个主题，{len(group_rows)} 个大类；名称来源：{"DeepSeek 命名" if topic_names else "c-TF-IDF 关键词（运行 --stage label 可生成中文名称）"}',
        f'- 生成时间：{time.strftime("%Y-%m-%d %H:%M")}', '',
        '## 大类分布', '',
        '| 大类 | 名称 | 占比 | | 条数 | 主题数 |', '|---:|---|---:|---|---:|---:|',
    ]
    lines += [f'| {g["group"]} | {md_cell(gname(g))} | {pct(g["ratio"])} | {bar(g["ratio"])} | {g["size"]:,} | {len(g["topics"])} |' for g in group_rows]
    lines += ['', '## 各大类下的主题', '', '占比是占全部样本的比例；相似度是样本与主题中心的平均余弦相似度，越低说明这个主题越松散。', '']
    for g in group_rows:
        desc = group_names.get(g['group'], {}).get('description')
        lines += [f'### {g["group"]}. {md_cell(gname(g))}（{pct(g["ratio"])}）', '']
        if desc: lines += [desc, '']
        lines += [f'关键词：{"、".join(g["keywords"])}', '', '| 主题 | 名称 | 占比 | 条数 | 相似度 | 代表样本 |', '|---:|---|---:|---:|---:|---|']
        for t in (by_id[i] for i in g['topics']):
            example = t['examples'][0]['text'][:60] if t['examples'] else ''
            lines.append(f'| {t["topic"]} | {md_cell(tname(t))} | {pct(t["ratio"])} | {t["size"]:,} | {t["score"]:.3f} | {md_cell(example)} |')
        lines.append('')
    with open(files['report'], 'w', encoding='utf-8') as f: f.write('\n'.join(lines))

    # utf-8-sig：Excel 直接打开不乱码
    with open(files['csv'], 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['topic', 'group', 'group_name', 'name', 'description', 'size', 'ratio', 'score', 'keywords', 'example'])
        groups_by_id = {g['group']: g for g in group_rows}
        for t in topics:
            w.writerow([t['topic'], t['group'], gname(groups_by_id[t['group']]), tname(t), topic_names.get(t['topic'], {}).get('description', ''),
                        t['size'], t['ratio'], t['score'], ' '.join(t['keywords']), t['examples'][0]['text'][:200].replace('\n', ' ') if t['examples'] else ''])
    print(f'[报告] -> {files["report"]}，{files["csv"]}')
    for g in group_rows[:10]: print(f'  {pct(g["ratio"]):>7} {bar(g["ratio"])} {gname(g)}')


def main():
    parser = argparse.ArgumentParser(description="分析 jsonl 数据集的主题分布：向量化 -> K-Means 聚类 -> 关键词 -> （可选）DeepSeek 命名 -> 报告")
    parser.add_argument('--stage', required=True, choices=STAGES + ['all'], help="embed=分片向量化（GPU，最耗时）；cluster=K-Means 聚成主题和大类；keywords=抽样提关键词和代表样本；label=调 DeepSeek 起中文名（收费，不在 all 里）；report=生成报告；all=embed+cluster+keywords+report")
    parser.add_argument('--data', default='', type=str, help="要分析的 jsonl；embed 阶段必填，之后的阶段默认用 meta.json 里记录的路径")
    parser.add_argument('--work_dir', default='', type=str, help="输出目录，默认 dataset/topic_map/{数据文件名}/（已被 .gitignore 忽略）；Colab 上建议指向 Drive")
    parser.add_argument('--field', default='', type=str, help="取哪个字段作为文本，默认自动识别（conversations > chosen > text > question）")
    parser.add_argument('--text_mode', default='user', type=str, choices=['user', 'all'], help="对话数据取哪些发言：user=只取用户提问（看知识分布推荐），all=用户和助手都取")
    parser.add_argument('--max_chars', default=1000, type=int, help="每条文本截断到多少字（再由模型按 --max_seq_length 截断）")
    parser.add_argument('--model', default='intfloat/multilingual-e5-small', type=str, help="sentence-transformers 向量模型；中英混合推荐默认值，纯中文可用 BAAI/bge-small-zh-v1.5，更快可用 minishlab/potion-multilingual-128M")
    parser.add_argument('--prefix', default='query: ', type=str, help="文本前缀，e5 系列需要 'query: '，换其他模型时设为空字符串 ''")
    parser.add_argument('--max_seq_length', default=256, type=int, help="向量模型的最大 token 数，越大越慢")
    parser.add_argument('--batch_size', default=256, type=int, help="向量化 batch 大小")
    parser.add_argument('--shard_size', default=100000, type=int, help="每多少条存一个向量分片；续跑以分片为单位")
    parser.add_argument('--k', default=200, type=int, help="主题数（K-Means 簇数）")
    parser.add_argument('--k_top', default=20, type=int, help="大类数（对主题中心再聚类）")
    parser.add_argument('--train_size', default=500000, type=int, help="训练 K-Means 用的抽样条数，训练完再给全部样本分配主题")
    parser.add_argument('--iters', default=30, type=int, help="K-Means 迭代次数")
    parser.add_argument('--n_keywords', default=10, type=int, help="每个主题 / 大类保留的关键词数")
    parser.add_argument('--kw_samples', default=200, type=int, help="每个主题随机抽多少条样本统计关键词")
    parser.add_argument('--n_examples', default=5, type=int, help="每个主题保存几条代表样本（与中心最相似）和几条随机样本")
    parser.add_argument('--device', default='', type=str, help="运行设备，默认有 GPU 用 cuda")
    parser.add_argument('--seed', default=42, type=int, help="随机种子（抽样、K-Means 初始化）")
    parser.add_argument('--limit', default=0, type=int, help="冒烟测试：embed 只处理前 N 条；label 只命名前 N 个主题 / 大类（0=全部）")
    parser.add_argument('--resume', default=0, type=int, choices=[0, 1], help="续跑：embed 跳过已写好的分片，label 跳过已命名的主题（0=覆盖旧输出）")
    parser.add_argument('--llm_model', default='deepseek-flash', type=str, help="label 阶段用的 DeepSeek 模型")
    parser.add_argument('--base_url', default='https://api.deepseek.com', type=str, help="API 地址，API key 从环境变量 DEEPSEEK_API_KEY 读取")
    parser.add_argument('--env_file', default=os.path.join(ROOT, '.env'), type=str, help="启动时读取的环境变量文件，默认仓库根目录的 .env")
    parser.add_argument('--workers', default=8, type=int, help="label 阶段的并发请求数")
    parser.add_argument('--temperature', default=0.3, type=float, help="label 阶段的采样温度")
    parser.add_argument('--max_retries', default=5, type=int, help="429/5xx/超时时 SDK 自动退避重试的次数")
    parser.add_argument('--timeout', default=120, type=float, help="单次请求超时（秒）")
    parser.add_argument('--max_attempts', default=3, type=int, help="返回的 JSON 无效时重新请求的次数")
    parser.add_argument('--json_max_tokens', default=256, type=int, help="label 阶段单次输出上限（DeepSeek tokens）")
    args = parser.parse_args()

    stages = ALL_STAGES if args.stage == 'all' else [args.stage]
    if 'embed' in stages and not args.data: raise SystemExit('embed 阶段需要用 --data 指定数据文件')
    if not args.work_dir:
        if not args.data: raise SystemExit('请用 --work_dir 指定输出目录（或用 --data 推出默认目录）')
        args.work_dir = os.path.join(ROOT, 'dataset', 'topic_map', os.path.splitext(os.path.basename(args.data))[0])
    os.makedirs(args.work_dir, exist_ok=True)
    print(f'工作目录：{args.work_dir}')
    j = lambda name: os.path.join(args.work_dir, name)
    files = {
        'meta': j('meta.json'), 'emb_dir': j('1_embeddings'),
        'labels': j('2_labels.npy'), 'scores': j('2_scores.npy'), 'centroids': j('2_centroids.npy'), 'groups': j('2_groups.npy'),
        'topics': j('3_topics.jsonl'), 'group_list': j('3_groups.jsonl'),
        'topic_labels': j('4_topic_labels.jsonl'), 'group_labels': j('4_group_labels.jsonl'),
        'report': j('report.md'), 'csv': j('topics.csv'),
    }
    runners = {'embed': stage_embed, 'cluster': stage_cluster, 'keywords': stage_keywords, 'label': stage_label, 'report': stage_report}
    for stage in stages:
        runners[stage](args, files)


if __name__ == "__main__":
    main()
