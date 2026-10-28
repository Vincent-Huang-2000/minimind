import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import math
import argparse
import warnings
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_model_params
warnings.filterwarnings('ignore')

# 在两次训练都没见过的留出集上，用完全相同的方式给每个权重算 loss / perplexity，作为打包对比的质量结论。
# 每篇文档写成 [bos] 文本 [eos]，按 max_seq_len 切成窗口（和打包切段一致），每个窗口单独前向，不和别的文档拼在一起。
# loss 按 token 平均，并拆成两部分：
#   head：每篇的第一个窗口，也就是不打包训练时保留下来的部分；
#   tail：超长文档后续的窗口，不打包训练时被截断丢掉，只有打包训练过（窗口单独计算，看不到前文）。
# 需要 torch，在 Colab 上运行；结果合并写入 {work_dir}/heldout_loss.json，重跑某个权重只覆盖它自己的那条。
# 用法（仓库根目录）:
#   python experiment/pack_compare/eval_heldout_loss.py --weights pretrain_nopack,pretrain_pack --save_dir <Drive>/out
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DTYPES = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}


def build_windows(path, tokenizer, max_seq_len, limit=0):
    """返回 [(token 列表, 是否为文档第一个窗口)]，按长度降序排好，同一 batch 内少补 pad"""
    windows, docs = [], 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            if limit and docs >= limit: break
            ids = [tokenizer.bos_token_id] + tokenizer(str(json.loads(line)['text']), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
            for s in range(0, len(ids), max_seq_len):
                w = ids[s:s + max_seq_len]
                if len(w) >= 2: windows.append((w, s == 0))  # 只有 1 个 token 的窗口没有可预测的位置
            docs += 1
    windows.sort(key=lambda w: -len(w[0]))
    return windows, docs


def load_model(args, weight):
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = os.path.join(args.save_dir, f'{weight}_{args.hidden_size}{moe_suffix}.pth')
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    return model.to(DTYPES[args.dtype]).eval().to(args.device)


@torch.no_grad()
def evaluate(model, windows, pad_id, batch_size, device):
    sums = {'head': [0.0, 0], 'tail': [0.0, 0]}  # [loss 之和, 预测的 token 数]
    for b in range(0, len(windows), batch_size):
        batch = windows[b:b + batch_size]
        T = len(batch[0][0])
        input_ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        for i, (w, _) in enumerate(batch): input_ids[i, :len(w)] = torch.tensor(w)
        labels = input_ids.clone()
        for i, (w, _) in enumerate(batch): labels[i, len(w):] = -100
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids).logits[:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100, reduction='none').view(len(batch), -1)
        per_window, counts = loss.sum(1).tolist(), (labels[:, 1:] != -100).sum(1).tolist()
        for (_, is_head), l, n in zip(batch, per_window, counts):
            part = sums['head' if is_head else 'tail']
            part[0] += l
            part[1] += n
    out = {}
    for name, (l, n) in (('all', [sums['head'][0] + sums['tail'][0], sums['head'][1] + sums['tail'][1]]), *sums.items()):
        out[name] = {'loss': l / n, 'ppl': math.exp(l / n), 'tokens': n} if n else {'loss': None, 'ppl': None, 'tokens': 0}
    return out


def report(results, names):
    print(f"{'权重':<24}" + ''.join(f'{p + " loss":>12}{p + " ppl":>11}' for p in ('all', 'head', 'tail')))
    for name in names:
        r = results[name]
        print(f'{name:<24}' + ''.join(f'{r[p]["loss"]:>12.4f}{r[p]["ppl"]:>11.2f}' if r[p]['tokens'] else f'{"-":>12}{"-":>11}' for p in ('all', 'head', 'tail')))
    if len(names) == 2:
        a, b = (results[n] for n in names)
        diffs = [f'{p} {b[p]["loss"] - a[p]["loss"]:+.4f}' for p in ('all', 'head', 'tail') if a[p]['tokens'] and b[p]['tokens']]
        print(f'loss 差值（{names[1]} - {names[0]}，负数表示后者更好）: ' + '，'.join(diffs))


def main():
    parser = argparse.ArgumentParser(description='在留出集上比较多个预训练权重的 loss / perplexity')
    parser.add_argument('--data', default=os.path.join(ROOT, 'dataset', 'pretrain_t2t_mini_heldout.jsonl'), type=str, help='留出集 jsonl（split_heldout.py 生成）')
    parser.add_argument('--weights', default='pretrain_nopack,pretrain_pack', type=str, help='逗号分隔的权重前缀，与训练的 --save_weight 一致')
    parser.add_argument('--save_dir', default=os.path.join(ROOT, 'out'), type=str, help='模型权重目录')
    parser.add_argument('--work_dir', default=os.path.join(ROOT, 'out', 'pack_compare'), type=str, help='对比实验的输出目录，Colab 上指向 Drive')
    parser.add_argument('--max_seq_len', default=340, type=int, help='评测窗口长度，与训练的 --max_seq_len 一致')
    parser.add_argument('--batch_size', default=32, type=int, help='评测 batch size')
    parser.add_argument('--hidden_size', default=768, type=int, help='隐藏层维度')
    parser.add_argument('--num_hidden_layers', default=8, type=int, help='隐藏层数量')
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help='是否使用MoE架构（0=否，1=是）')
    parser.add_argument('--dtype', default='float16' if torch.cuda.is_available() else 'float32', choices=list(DTYPES), help='推理精度，loss 统一用 float32 计算')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='运行设备')
    parser.add_argument('--limit', default=0, type=int, help='只用前 N 篇（0=全部），用于冒烟测试')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    windows, docs = build_windows(args.data, tokenizer, args.max_seq_len, args.limit)
    print(f'留出集 {docs} 篇，切成 {len(windows)} 个窗口（其中 tail 窗口 {sum(not h for _, h in windows)} 个）')

    path = os.path.join(args.work_dir, 'heldout_loss.json')
    saved = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f: saved = json.load(f)
    setting = {'data': os.path.abspath(args.data), 'docs': docs, 'max_seq_len': args.max_seq_len}
    if saved and {k: saved.get(k) for k in setting} != setting:
        print(f'[提示] {path} 里是另一份留出集或窗口长度的结果，本次将覆盖')
        saved = {}
    results = saved.get('results', {})

    names = [w.strip() for w in args.weights.split(',') if w.strip()]
    for name in names:
        model = load_model(args, name)
        results[name] = evaluate(model, windows, tokenizer.pad_token_id, args.batch_size, args.device)
        print(f'{name}: loss {results[name]["all"]["loss"]:.4f}')
        del model
        if args.device.startswith('cuda'): torch.cuda.empty_cache()

    report(results, names)
    os.makedirs(args.work_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({**setting, 'results': results}, f, ensure_ascii=False, indent=2)
    print(f'已保存: {path}')


if __name__ == '__main__':
    main()
