import os
import re
import json
import math
import time
import random
import hashlib
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STAGES = ['concepts', 'merge', 'questions', 'answers', 'build']

# 每个概念从这里抽题型；另按 --false_premise_ratio 把最后一题换成“错误前提”，训练模型识破错误说法
QUESTION_TYPES = {
    '定义': '问这个概念是什么、指什么',
    '举例': '要求举出这个概念的具体实例',
    '对比': '问它与某个相关概念的区别或联系',
    '原因': '问为什么、背后的原理或成因',
    '应用': '问它在生活、学习或实际中的用途',
    '误区': '问关于它的一个常见误解，并要求纠正',
}
FALSE_PREMISE = ('错误前提', '问题本身包含一个与该概念相关的错误说法（如张冠李戴、因果颠倒），回答者需要先指出错误')

CONCEPTS_PROMPT = """你在为一个小型中文语言模型准备训练数据。下面是一道知识题（领域：{domain}）：
{question}

请找出回答这道题需要掌握的知识概念，输出 JSON：
{{"concepts": [{{"name": "概念名", "kind": "core", "explanation": "一句话解释"}}]}}

要求：
1. kind 为 core 的是题目直接涉及的核心概念，2–4 个；kind 为 related 的是与核心概念紧密相关、有助于理解它们的相邻概念，2–3 个。
2. name 是简短的名词或术语，不超过 12 个字，不要写成句子或问题。
3. explanation 用一句话准确解释该概念，不超过 60 个字。
4. 只提取真实存在的知识。如果题目包含虚构或错误的前提，提取识破这个前提所需的真实知识（如相关人物的年代、国籍），不要把虚构的内容当作概念。"""

QUESTIONS_PROMPT = """你在为一个小型中文语言模型准备训练数据。请围绕下面这个概念出题（领域：{domain}）：
概念：{name}
解释：{explanation}

请生成 {n} 个关于这个概念的问题，每种题型一个：
{types}

输出 JSON：
{{"questions": [{{"type": "题型", "question": "问题"}}]}}

要求：
1. 每个问题都能独立看懂、不依赖上下文，不超过 50 个字。
2. 问题之间不要重复，也不要只是同一个问题换了说法。
3. 每个问题都应能在 {answer_chars} 字以内讲清楚，不要出需要长篇推导的题。
4. 不要生成与下列问题意思相同或相近的问题；如果某个题型避不开它们，就换一个角度提问：
{avoid}"""

ANSWER_SYSTEM = """你是一个知识丰富、表达清楚的中文助手。回答要求：
1. 直接回答问题，内容准确，不编造；不确定的地方要说明不确定。
2. 如果问题包含错误前提，先明确指出错误，再给出正确信息。
3. 简洁：全文不超过 {answer_chars} 个字，通常两到四段或几个要点即可。
4. 不用 Markdown 标题和表格；公式用纯文本写，不用 LaTeX。"""


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')


def require(path, stage):
    if not os.path.exists(path): raise SystemExit(f'找不到 {path}，请先运行 --stage {stage}')
    return read_jsonl(path)


def load_done(path, key):
    """读出已完成的记录，并把文件重写成只含完整行（断连时最后一行可能写了一半）。"""
    done = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try: done.append(json.loads(line))
                except json.JSONDecodeError: pass
        write_jsonl(path, done)
    return {r[key] for r in done}


def normalize(text):
    # 去掉空白和标点（中文字符算 \w 会保留），用于去重和相似度比较
    return re.sub(r'[\W_]+', '', text).lower()


def concept_id(name):
    # 由概念名决定，重跑 merge 或删改概念表后 id 不变，续跑不会错位
    return 'c' + hashlib.md5(normalize(name).encode('utf-8')).hexdigest()[:10]


def char_ngrams(text):
    return Counter(list(text) + [text[i:i + 2] for i in range(len(text) - 1)])


class ProbeMatcher:
    """找出与生成问题最相似的原题：单字 + 双字的 TF-IDF 余弦相似度。
    IDF 在原题和全部生成问题上统计，“是什么意思”“有什么区别”这类套话出现得多、权重低，比较的主要是概念本身；
    较短的一句被另一句包含时直接视为重复（如“什么是质数”与“什么是质数？请用自己的话解释”）。"""

    def __init__(self, probes, questions):
        texts = [p for _, p in probes] + questions
        df = Counter(g for t in texts for g in set(char_ngrams(t)))
        self.idf = {g: math.log((1 + len(texts)) / (1 + n)) + 1 for g, n in df.items()}
        self.probes = [(pid, p, self.vector(p)) for pid, p in probes]

    def vector(self, text):
        v = {g: c * self.idf.get(g, 1.0) for g, c in char_ngrams(text).items()}
        return v, math.sqrt(sum(x * x for x in v.values())) or 1.0

    def best(self, text):
        (v, norm), best = self.vector(text), (-1.0, '')
        for pid, p, (pv, pnorm) in self.probes:
            if min(len(text), len(p)) >= 4 and (text in p or p in text): return 1.0, pid
            best = max(best, (sum(x * pv.get(g, 0) for g, x in v.items()) / (norm * pnorm), pid))
        return best


def load_env_file(path):
    """读取 KEY=VALUE 格式的环境变量文件（模板见仓库根目录 .env.example），系统里已设置的同名变量优先。"""
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line: continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))


def make_client(args):
    from openai import OpenAI
    api_key = os.environ.get('DEEPSEEK_API_KEY')
    if not api_key: raise SystemExit(f'没有找到 DEEPSEEK_API_KEY：把 .env.example 复制为 {args.env_file} 并填入 key，或直接设置同名环境变量')
    return OpenAI(api_key=api_key, base_url=args.base_url, max_retries=args.max_retries, timeout=args.timeout)


def chat(client, args, messages, max_tokens, json_mode=False):
    # 关闭思考模式：便宜，且推理过程太长，MiniMind 的训练长度装不下
    kwargs = {'response_format': {'type': 'json_object'}} if json_mode else {}
    resp = client.chat.completions.create(
        model=args.model, messages=messages, max_tokens=max_tokens, temperature=args.temperature,
        extra_body={'thinking': {'type': 'disabled'}}, **kwargs
    )
    u = resp.usage
    usage = Counter(calls=1, prompt=u.prompt_tokens, cache_hit=getattr(u, 'prompt_cache_hit_tokens', 0) or 0, completion=u.completion_tokens)
    choice = resp.choices[0]
    return (choice.message.content or '').strip(), choice.finish_reason, usage


def chat_json(client, args, prompt, max_tokens, check):
    usage, err = Counter(), None
    for _ in range(args.max_attempts):
        content, finish, u = chat(client, args, [{'role': 'user', 'content': prompt}], max_tokens, json_mode=True)
        usage += u
        try:
            if finish != 'stop': raise ValueError(f'finish_reason={finish}')
            return check(json.loads(content)), usage
        except (ValueError, AttributeError, TypeError) as e:
            err = e
    raise ValueError(f'连续 {args.max_attempts} 次没拿到有效 JSON：{err}')


def check_concepts(data):
    items = [c for c in data['concepts'] if isinstance(c, dict) and isinstance(c.get('name'), str) and isinstance(c.get('explanation'), str)]
    items = [c for c in items if 0 < len(c['name'].strip()) <= 20]
    if not items: raise ValueError('没有有效的概念')
    return [{'name': c['name'].strip(), 'kind': 'core' if c.get('kind') == 'core' else 'related', 'explanation': c['explanation'].strip()} for c in items]


def check_questions(data):
    items = [q for q in data['questions'] if isinstance(q, dict) and isinstance(q.get('question'), str) and q['question'].strip()]
    if not items: raise ValueError('没有有效的问题')
    return [{'type': str(q.get('type', '')).strip(), 'question': q['question'].strip()} for q in items]


def run_stage(args, items, key, worker, out_path, desc, show):
    """并发调用 API，按完成顺序逐条追加写入；失败的条目不写，--resume 1 重跑时会再试。"""
    if args.limit: items = items[:args.limit]
    done = load_done(out_path, key) if args.resume else set()
    todo = [it for it in items if it[key] not in done]
    print(f'[{desc}] 共 {len(items)} 条，已完成 {len(done)}，本次待跑 {len(todo)} -> {out_path}')
    if not todo: return
    client = make_client(args)
    usage, failed, st = Counter(), [], time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with open(out_path, 'a' if args.resume else 'w', encoding='utf-8') as f:
            futures = {pool.submit(worker, client, args, it): it for it in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                it = futures[fut]
                try:
                    rec, u = fut.result()
                except Exception as e:
                    failed.append(it[key])
                    print(f'[{n}/{len(todo)}] {it[key]} 失败：{e}')
                    continue
                usage += u
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
                print(f'[{n}/{len(todo)}] {it[key]} {show(rec)}')
    finally:
        # 中途 Ctrl+C 时不再等排队中的请求
        pool.shutdown(wait=False, cancel_futures=True)
    print(f'[{desc}] 完成 {len(todo) - len(failed)} 条，失败 {len(failed)} 条，耗时 {time.time() - st:.0f}s；'
          f'API 调用 {usage["calls"]} 次，输入 {usage["prompt"]} tokens（缓存命中 {usage["cache_hit"]}），输出 {usage["completion"]} tokens')
    if failed: print(f'[{desc}] 失败的条目可加 --resume 1 重跑：{failed[:20]}{" ..." if len(failed) > 20 else ""}')


def stage_concepts(args, files):
    def worker(client, args, row):
        prompt = CONCEPTS_PROMPT.format(domain=row.get('domain') or '通用', question=row['question'])
        concepts, usage = chat_json(client, args, prompt, args.json_max_tokens, check_concepts)
        return {'id': row['id'], 'domain': row.get('domain', ''), 'question': row['question'], 'concepts': concepts}, usage
    run_stage(args, read_jsonl(args.data), 'id', worker, files['raw'], '提取概念', lambda r: '、'.join(c['name'] for c in r['concepts']))


def stage_merge(args, files):
    merged = {}
    for r in sorted(require(files['raw'], 'concepts'), key=lambda r: r['id']):
        for c in r['concepts']:
            cid = concept_id(c['name'])
            m = merged.setdefault(cid, {'cid': cid, 'name': c['name'], 'domain': r['domain'], 'kind': c['kind'], 'explanation': c['explanation'], 'source_ids': [], 'source_questions': []})
            if c['kind'] == 'core': m['kind'] = 'core'
            if r['id'] not in m['source_ids']:
                m['source_ids'].append(r['id'])
                m['source_questions'].append(r['question'])
    write_jsonl(files['concepts'], merged.values())
    n_raw = sum(len(r['concepts']) for r in read_jsonl(files['raw']))
    print(f'[合并概念] {n_raw} 个原始概念去重后剩 {len(merged)} 个（core {sum(m["kind"] == "core" for m in merged.values())}）-> {files["concepts"]}')
    print('[合并概念] 可以先检查、删改这个文件再跑 questions；注意重跑 merge 会覆盖你的修改')


def pick_types(args, cid):
    # 按 cid 固定随机种子，续跑时同一概念的题型不变
    rng = random.Random(f'{args.seed}-{cid}')
    types = [(t, QUESTION_TYPES[t]) for t in rng.sample(list(QUESTION_TYPES), args.questions_per_concept)]
    if rng.random() < args.false_premise_ratio: types[-1] = FALSE_PREMISE
    return types


def stage_questions(args, files):
    def worker(client, args, c):
        types = pick_types(args, c['cid'])
        prompt = QUESTIONS_PROMPT.format(
            domain=c.get('domain') or '通用', name=c['name'], explanation=c['explanation'], n=len(types), answer_chars=args.answer_chars,
            types='\n'.join(f'- {t}：{d}' for t, d in types), avoid='\n'.join(f'- {q}' for q in c['source_questions'])
        )
        questions, usage = chat_json(client, args, prompt, args.json_max_tokens, check_questions)
        return {'cid': c['cid'], 'name': c['name'], 'questions': [{'qid': f'{c["cid"]}-{j}', **q} for j, q in enumerate(questions)]}, usage
    concepts = require(files['concepts'], 'merge')
    run_stage(args, concepts, 'cid', worker, files['questions'], '生成问题', lambda r: f'{r["name"]}：{len(r["questions"])} 题')


def stage_answers(args, files):
    # 只回答概念表里还在的概念，删掉的概念即使已经出过题也跳过
    cids = {c['cid'] for c in require(files['concepts'], 'merge')}
    items, seen = [], set()
    for r in require(files['questions'], 'questions'):
        if r['cid'] not in cids: continue
        for q in r['questions']:
            if q['qid'] in seen: continue
            seen.add(q['qid'])
            items.append({'qid': q['qid'], 'cid': r['cid'], 'concept': r['name'], 'type': q['type'], 'question': q['question']})
    system = ANSWER_SYSTEM.format(answer_chars=args.answer_chars)

    def worker(client, args, q):
        answer, finish, usage = chat(client, args, [{'role': 'system', 'content': system}, {'role': 'user', 'content': q['question']}], args.answer_max_tokens)
        return {**q, 'answer': answer, 'finish_reason': finish}, usage
    run_stage(args, items, 'qid', worker, files['answers'], '生成回答', lambda r: f'{len(r["answer"])} 字 {r["question"][:30]}')


def stage_build(args, files):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    concepts = {c['cid']: c for c in require(files['concepts'], 'merge')}
    answers = [r for r in require(files['answers'], 'answers') if r['cid'] in concepts]
    probes = [(r['id'], normalize(r['question'])) for r in read_jsonl(args.data)]
    matcher = ProbeMatcher(probes, [normalize(r['question']) for r in answers])
    budget = args.max_seq_len - args.reserve_tokens
    kept, meta, dropped, seen, lengths = [], [], [], set(), []

    for r in answers:
        nq = normalize(r['question'])
        if r['finish_reason'] != 'stop' or not r['answer']:
            dropped.append({**r, 'reason': 'truncated'}); continue
        if nq in seen:
            dropped.append({**r, 'reason': 'duplicate'}); continue
        sim, pid = matcher.best(nq)
        if sim >= args.sim_threshold:
            dropped.append({**r, 'reason': 'similar_to_probe', 'probe_id': pid, 'similarity': round(sim, 3)}); continue
        conversations = [{'role': 'user', 'content': r['question']}, {'role': 'assistant', 'content': r['answer']}]
        # 与 SFTDataset 的算法一致：渲染模板后整体分词（含空思考标签，偏保守）
        n_tokens = len(tokenizer(tokenizer.apply_chat_template(conversations, tokenize=False)).input_ids)
        if n_tokens > budget:
            dropped.append({**r, 'reason': 'too_long', 'tokens': n_tokens}); continue
        seen.add(nq)
        lengths.append(n_tokens)
        kept.append({'conversations': conversations})
        meta.append({'qid': r['qid'], 'cid': r['cid'], 'concept': r['concept'], 'type': r['type'], 'tokens': n_tokens, 'source_ids': concepts[r['cid']]['source_ids']})

    # 训练文件只放 conversations：SFTDataset 用固定 Features 加载，多余的顶层字段会导致加载失败
    write_jsonl(args.output, kept)
    write_jsonl(files['meta'], meta)
    write_jsonl(files['dropped'], dropped)
    print(f'[组装 SFT] 保留 {len(kept)} 条 -> {args.output}')
    print(f'[组装 SFT] 丢弃 {len(dropped)} 条 {dict(Counter(d["reason"] for d in dropped))} -> {files["dropped"]}')
    if lengths:
        lengths.sort()
        pct = lambda p: lengths[min(len(lengths) - 1, int(len(lengths) * p))]
        print(f'[组装 SFT] token 长度 p50={pct(0.5)} p90={pct(0.9)} max={lengths[-1]}（上限 {budget} = max_seq_len {args.max_seq_len} - 预留 {args.reserve_tokens}）')


def main():
    parser = argparse.ArgumentParser(description="调用 DeepSeek API，把知识探针题库扩增成 SFT 数据：提取概念 -> 合并去重 -> 围绕概念出题 -> 生成回答 -> 组装")
    parser.add_argument('--stage', required=True, choices=STAGES + ['all'], help="concepts=从每道题提取概念；merge=合并去重成概念表（不调 API）；questions=围绕每个概念出题；answers=生成回答；build=过滤并组装 SFT jsonl（不调 API）；all=依次全部执行")
    parser.add_argument('--data', default=os.path.join(ROOT, 'eval_data', 'small_llm_knowledge_probe_142.jsonl'), type=str, help="题库 jsonl（每行至少含 id 和 question）")
    parser.add_argument('--work_dir', default=os.path.join(ROOT, 'dataset'), type=str, help="中间文件目录（probe_aug_*.jsonl），默认 dataset/，已被 .gitignore 忽略")
    parser.add_argument('--output', default='', type=str, help="最终 SFT 数据，默认 {work_dir}/sft_probe_aug.jsonl，可直接作为 train_full_sft.py 的 --data_path")
    parser.add_argument('--model', default='deepseek-flash', type=str, help="DeepSeek 模型名（deepseek-flash / deepseek-v4-pro）")
    parser.add_argument('--base_url', default='https://api.deepseek.com', type=str, help="API 地址，API key 从环境变量 DEEPSEEK_API_KEY 读取")
    parser.add_argument('--env_file', default=os.path.join(ROOT, '.env'), type=str, help="启动时读取的环境变量文件，默认仓库根目录的 .env（模板见 .env.example）")
    parser.add_argument('--workers', default=8, type=int, help="并发请求数")
    parser.add_argument('--temperature', default=1.0, type=float, help="采样温度（0-2）")
    parser.add_argument('--max_retries', default=5, type=int, help="429/5xx/超时时 SDK 自动退避重试的次数")
    parser.add_argument('--timeout', default=300, type=float, help="单次请求超时（秒）")
    parser.add_argument('--max_attempts', default=3, type=int, help="返回的 JSON 无效时重新请求的次数")
    parser.add_argument('--json_max_tokens', default=1024, type=int, help="concepts / questions 阶段单次输出上限（DeepSeek tokens）")
    parser.add_argument('--questions_per_concept', default=5, type=int, choices=range(1, len(QUESTION_TYPES) + 1), help="每个概念出几道题（每题一种题型）")
    parser.add_argument('--false_premise_ratio', default=0.2, type=float, help="多大比例的概念把最后一题换成“错误前提”题")
    parser.add_argument('--answer_chars', default=300, type=int, help="提示词里要求的回答字数上限（软限制）")
    parser.add_argument('--answer_max_tokens', default=512, type=int, help="answers 阶段单次输出上限（DeepSeek tokens，硬限制），撞上限的回答在 build 时丢弃")
    parser.add_argument('--max_seq_len', default=768, type=int, help="与 train_full_sft.py 的 --max_seq_len 保持一致，超长样本会被截断成没有结尾的回答，build 时直接丢弃")
    parser.add_argument('--reserve_tokens', default=64, type=int, help="给 SFTDataset 随机添加的 system prompt 预留的 token 数")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="生成的问题与任一原题的相似度（字级 TF-IDF 余弦）达到该值即丢弃（0-1，越小越严格），被丢弃的问题和匹配到的原题写在 *_dropped.jsonl 里供检查")
    parser.add_argument('--seed', default=42, type=int, help="随机种子，决定每个概念抽到的题型")
    parser.add_argument('--limit', default=0, type=int, help="每个调 API 的阶段只跑前 N 条（0=全部），用于冒烟测试")
    parser.add_argument('--resume', default=0, type=int, choices=[0, 1], help="续跑：跳过输出文件里已有的条目并在其后追加（0=覆盖旧输出）")
    args = parser.parse_args()

    load_env_file(args.env_file)
    os.makedirs(args.work_dir, exist_ok=True)
    if not args.output: args.output = os.path.join(args.work_dir, 'sft_probe_aug.jsonl')
    stem = os.path.splitext(args.output)[0]
    files = {
        'raw': os.path.join(args.work_dir, 'probe_aug_concepts_raw.jsonl'),
        'concepts': os.path.join(args.work_dir, 'probe_aug_concepts.jsonl'),
        'questions': os.path.join(args.work_dir, 'probe_aug_questions.jsonl'),
        'answers': os.path.join(args.work_dir, 'probe_aug_answers.jsonl'),
        'meta': f'{stem}_meta.jsonl',
        'dropped': f'{stem}_dropped.jsonl',
    }
    runners = {'concepts': stage_concepts, 'merge': stage_merge, 'questions': stage_questions, 'answers': stage_answers, 'build': stage_build}
    for stage in (STAGES if args.stage == 'all' else [args.stage]):
        runners[stage](args, files)


if __name__ == "__main__":
    main()
