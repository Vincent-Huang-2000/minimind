"""查看 topic_map.py 聚类结果里的完整样本（只读，不写任何文件）。

在 Colab notebook 里直接显示（工作目录为仓库根目录）：
    import sys; sys.path.insert(0, 'scripts')
    from view_topic_samples import view
    W = '/content/drive/MyDrive/minimind/topic_map/pretrain_t2t_mini'
    view(W, topics=[129, 494, 463])          # 指定主题
    view(W, groups=[19, 26], n=3)            # 指定大类，展开成其下所有主题
    view(W, pattern='通义|qwen|moss', n=20)   # 正则（不区分大小写），先统计匹配落在哪些主题
    view(W, groups=[14], pattern='天气')      # 只在这些主题里找匹配

每个主题（只给正则时是全部匹配）显示两组样本：随机 n 条，看主题的真实面貌；离主题中心最远的 n 条，看边缘混进了什么。
命令行（输出 markdown 文本）：python scripts/view_topic_samples.py --work_dir ... --topics 129,494
"""
import os
import re
import json
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from topic_map import iter_lines, read_json, md_cell, read_jsonl


def full_text(row, meta):
    """完整样本：不截断；对话数据所有发言都取，每条前面标上 role。"""
    value = row.get(meta['field']) if meta['field'] else next((row[k] for k in ('conversations', 'chosen', 'text', 'question') if k in row), None)
    if isinstance(value, list):
        return '\n\n'.join(f'[{m.get("role")}] {m.get("content") or ""}' if isinstance(m, dict) else str(m) for m in value)
    return str(value)


def load(work_dir, data=''):
    j = lambda name: os.path.join(work_dir, name)
    for name in ('meta.json', '2_labels.npy', '2_scores.npy', '2_groups.npy'):
        if not os.path.exists(j(name)): raise SystemExit(f'找不到 {j(name)}，请先用 topic_map.py 跑完 embed 和 cluster')
    meta = read_json(j('meta.json'))
    data = data or meta['data']
    # 行号要和向量一一对应，数据文件换过（大小不同）就不能再用
    if os.path.getsize(data) != meta['data_bytes']: raise SystemExit(f'{data} 的大小与向量化时不同，不是同一个数据文件')
    names = {t['topic']: ' / '.join(t['keywords'][:4]) for t in read_jsonl(j('3_topics.jsonl'))} if os.path.exists(j('3_topics.jsonl')) else {}
    if os.path.exists(j('4_topic_labels.jsonl')): names.update({r['topic']: r['name'] for r in read_jsonl(j('4_topic_labels.jsonl'))})
    return meta, data, np.load(j('2_labels.npy')), np.load(j('2_scores.npy')).astype(np.float32), np.load(j('2_groups.npy')), names


def build_markdown(work_dir, topics=(), groups=(), pattern='', n=5, data='', seed=42):
    if not (topics or groups or pattern): raise SystemExit('至少要给 topics、groups、pattern 中的一个')
    meta, data, labels, scores, group_of, names = load(work_dir, data)
    k, total = len(group_of), len(labels)
    bad = [t for t in topics if not 0 <= t < k] + [f'大类 {g}' for g in groups if not 0 <= g <= group_of.max()]
    if bad: raise SystemExit(f'编号超出范围（主题 0~{k - 1}，大类 0~{group_of.max()}）：{bad}')
    topics = sorted(set(topics) | {t for g in groups for t in np.flatnonzero(group_of == g).tolist()})
    scope = np.isin(labels, topics) if topics else np.ones(total, dtype=bool)
    regex = re.compile(pattern, re.IGNORECASE) if pattern else None

    # 有正则时先扫一遍全文，在主题范围内找匹配的样本
    cand = scope
    if regex:
        print(f'在 {data} 中查找匹配 {pattern!r} 的样本……')
        cand = np.zeros(total, dtype=bool)
        for i, line in iter_lines(data, meta['rows']):
            if scope[i] and regex.search(full_text(json.loads(line), meta)): cand[i] = True

    # 每组抽两种：随机 n 条 + 离中心最远（相似度最低）的 n 条，后者去掉已在随机里的
    rng = np.random.default_rng(seed)

    def pick(rows):
        rand = sorted(rng.choice(rows, min(len(rows), n), replace=False).tolist())
        far = [int(r) for r in rows[np.argsort(scores[rows], kind='stable')] if r not in rand][:n]
        return rand, far
    buckets = {t: pick(np.flatnonzero(cand & (labels == t))) for t in topics} if topics else {None: pick(np.flatnonzero(cand))}
    want = {r for rand, far in buckets.values() for r in rand + far}
    rows = {}
    for i, line in iter_lines(data, meta['rows']):
        if i in want: rows[i] = json.loads(line)

    sizes, hits = np.bincount(labels, minlength=k), np.bincount(labels[cand], minlength=k)

    def block(r):
        text = full_text(rows[r], meta)
        fence = '`' * max(4, max((len(m) for m in re.findall('`+', text)), default=0) + 1)
        return [f'##### 行 {r} · 主题 {labels[r]}（{md_cell(names.get(int(labels[r]), ""))}）· 相似度 {scores[r]:.3f}', '', fence + 'text', text, fence, '']

    lines = [f'- 数据：`{data}`，共 {total:,} 条；行号只计非空行；相似度是样本与所属主题中心的余弦相似度']
    if regex: lines.append(f'- 正则：`{pattern}`（不区分大小写），匹配 {int(cand.sum()):,} 条，占范围内的 {cand.sum() / max(scope.sum(), 1):.2%}')
    lines.append('')
    if regex and cand.any():
        lines += ['### 匹配分布', '', '| 主题 | 名称 | 匹配条数 | 主题条数 | 主题内占比 |', '|---:|---|---:|---:|---:|']
        lines += [f'| {t} | {md_cell(names.get(int(t), ""))} | {hits[t]:,} | {sizes[t]:,} | {hits[t] / sizes[t]:.1%} |' for t in np.argsort(-hits, kind='stable')[:30] if hits[t]]
        lines.append('')
    for t, (rand, far) in buckets.items():
        if regex and t is not None and not hits[t]: continue  # 给了正则时，没有匹配的主题不显示
        lines += ['### 匹配的样本' if t is None else f'### 主题 {t}：{md_cell(names.get(t, ""))}（{sizes[t]:,} 条{f"，匹配 {hits[t]:,} 条" if regex else ""}）', '']
        lines += [f'#### 随机 {len(rand)} 条', '']
        for r in rand: lines += block(r)
        lines += [f'#### 离中心最远 {len(far)} 条', '']
        for r in far: lines += block(r)
    return '\n'.join(lines)


def view(work_dir, topics=(), groups=(), pattern='', n=5, data='', seed=42):
    """在 notebook 里直接显示。"""
    from IPython.display import Markdown, display
    display(Markdown(build_markdown(work_dir, topics, groups, pattern, n, data, seed)))


def main():
    parser = argparse.ArgumentParser(description="查看 topic_map.py 聚类结果里的完整样本，输出 markdown（只读）")
    parser.add_argument('--work_dir', required=True, type=str, help="topic_map.py 的输出目录")
    parser.add_argument('--data', default='', type=str, help="数据文件，默认用 meta.json 里记录的路径")
    parser.add_argument('--topics', default='', type=str, help="主题编号，逗号分隔")
    parser.add_argument('--groups', default='', type=str, help="大类编号，逗号分隔，展开成其下所有主题")
    parser.add_argument('--pattern', default='', type=str, help="只看完整文本匹配这个正则的样本（不区分大小写）")
    parser.add_argument('--n', default=5, type=int, help="每组随机抽几条、离中心最远取几条")
    parser.add_argument('--seed', default=42, type=int, help="随机种子，换一个能看到另一批随机样本")
    args = parser.parse_args()
    parse = lambda s: [int(x) for x in s.split(',') if x.strip()]
    print(build_markdown(args.work_dir, parse(args.topics), parse(args.groups), args.pattern, args.n, args.data, args.seed))


if __name__ == "__main__":
    main()
