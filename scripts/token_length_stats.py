import os
import json
import random
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
EMPTY_THINK = '<think>\n\n</think>\n\n'
DEFAULT_CANDIDATES = [256, 340, 384, 512, 640, 768, 1024, 1536, 2048]


def read_lines(path, limit=0, sample=0, seed=42):
    """limit>0 只读前 limit 行（冒烟用）；sample>0 对全文件做蓄水池抽样，只扫描一遍，不把整个文件读进内存"""
    rng, picked = random.Random(seed), []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            if not line.strip(): continue
            if not sample or len(picked) < sample:
                picked.append(line)
            else:
                j = rng.randint(0, i)
                if j < sample: picked[j] = line
    return [json.loads(line) for line in picked]


def detect_type(sample):
    if 'text' in sample: return 'pretrain'
    if 'chosen' in sample: return 'dpo'
    if 'conversations' in sample: return 'sft'
    raise ValueError(f"无法识别数据格式，字段: {list(sample.keys())}，请用 --type 指定")


def chat_text(tokenizer, conversations, prompt_only=False):
    """和 lm_dataset.py 的拼接方式一致：解析 tools / tool_calls 字符串，套 chat template；
    不做随机加 system（训练时 20% 概率），空思考标签按训练时 80% 的情况直接去掉"""
    messages, tools = [], None
    for message in conversations:
        message = dict(message)
        if message.get('role') == 'system' and message.get('tools'):
            tools = json.loads(message['tools']) if isinstance(message['tools'], str) else message['tools']
        if message.get('tool_calls') and isinstance(message['tool_calls'], str):
            message['tool_calls'] = json.loads(message['tool_calls'])
        messages.append(message)
    if prompt_only:  # RL 阶段：去掉最后一条回答，只统计喂给模型的 prompt
        return tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, tools=tools)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)
    return text.replace(EMPTY_THINK, '')


def to_texts(tokenizer, samples, data_type):
    """返回每条样本要统计的文本；DPO 一条样本有 chosen / rejected 两段，分别统计后取较长的"""
    if data_type == 'pretrain': return [[str(s['text'])] for s in samples]
    if data_type == 'sft': return [[chat_text(tokenizer, s['conversations'])] for s in samples]
    if data_type == 'prompt': return [[chat_text(tokenizer, s['conversations'], prompt_only=True)] for s in samples]
    if data_type == 'dpo': return [[chat_text(tokenizer, s['chosen']), chat_text(tokenizer, s['rejected'])] for s in samples]
    raise ValueError(data_type)


def count_tokens(tokenizer, groups, data_type, batch_size=1000):
    flat = [t for g in groups for t in g]
    lengths = []
    for i in range(0, len(flat), batch_size):
        # 预训练不加特殊 token，再手动补 bos/eos 两个；其余和训练一样直接 tokenizer(prompt)
        ids = tokenizer(flat[i:i + batch_size], add_special_tokens=data_type != 'pretrain').input_ids
        lengths += [len(x) + (2 if data_type == 'pretrain' else 0) for x in ids]
    tokens, chars, k = [], [], 0
    for g in groups:
        tokens.append(max(lengths[k:k + len(g)]))
        chars.append(max(len(t) for t in g))
        k += len(g)
    return tokens, chars


def percentile(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p / 100))]


def report(name, data_type, tokens, chars, candidates, target_pct, current):
    vals, n, total = sorted(tokens), len(tokens), sum(tokens)
    print(f"\n== {name}（类型 {data_type}，统计 {n} 条）==")
    print(f"平均 {total / n:.0f} | " + " | ".join(f"p{p} {percentile(vals, p)}" for p in [50, 75, 90, 95, 99]) + f" | max {vals[-1]}")
    print(f"中位数字符/token: {sorted(c / max(t, 1) for c, t in zip(chars, tokens))[n // 2]:.2f}")
    # 推荐值：覆盖目标分位数的最小 64 的倍数
    rec = -(-percentile(vals, target_pct) // 64) * 64
    rows = sorted(set(candidates) | {rec} | ({current} if current else set()))
    print("\n| max_seq_len | 截断样本 | 被截掉的 token | padding 占比 | 备注 |")
    print("|---:|---:|---:|---:|---|")
    for L in rows:
        cut = sum(t > L for t in tokens)
        lost = sum(t - L for t in tokens if t > L)
        pad = sum(L - t for t in tokens if t < L)
        note = '、'.join(x for x, ok in [('当前默认', L == current), (f'覆盖 p{target_pct:g}', L == rec)] if ok)
        print(f"| {L} | {cut / n:.1%} | {lost / total:.1%} | {pad / (n * L):.1%} | {note} |")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="统计数据集的 token 长度分布，用于选择训练的 --max_seq_len")
    parser.add_argument('--data', nargs='+', required=True, help="jsonl 路径，可以一次传多个")
    parser.add_argument('--type', default='auto', choices=['auto', 'pretrain', 'sft', 'dpo', 'prompt'],
                        help="auto 按字段判断（text→pretrain，chosen→dpo，conversations→sft）；PPO/GRPO/Agent RL 的数据用 prompt，只统计去掉最后回答后的 prompt")
    parser.add_argument('--sample', default=20000, type=int, help="随机抽样条数（蓄水池抽样，会扫一遍全文件），0 表示全部统计")
    parser.add_argument('--limit', default=0, type=int, help="只读前 N 行，本地冒烟用；和 --sample 同时给时先截前 N 行再抽样")
    parser.add_argument('--candidates', nargs='+', type=int, default=DEFAULT_CANDIDATES, help="要对比的 max_seq_len 候选值")
    parser.add_argument('--target_pct', default=95, type=float, help="推荐值覆盖的分位数")
    parser.add_argument('--current', default=0, type=int, help="当前训练用的 max_seq_len，会在表里标出来")
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    for path in args.data:
        samples = read_lines(path, args.limit, args.sample, args.seed)
        if not samples:
            print(f"{path} 没有数据，跳过")
            continue
        data_type = detect_type(samples[0]) if args.type == 'auto' else args.type
        tokens, chars = count_tokens(tokenizer, to_texts(tokenizer, samples, data_type), data_type)
        report(os.path.basename(path), data_type, tokens, chars, args.candidates, args.target_pct, args.current)
