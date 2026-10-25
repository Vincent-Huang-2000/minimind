import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import argparse
import warnings
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_questions(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), f'数据里有重复的 id: {path}'
    return rows


def load_done(path):
    """读出已完成的记录，并把文件重写成只含完整行（断连时最后一行可能写了一半）。"""
    done = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try: done.append(json.loads(line))
                except json.JSONDecodeError: pass
        with open(path, 'w', encoding='utf-8') as f:
            for r in done: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return {r['id'] for r in done}


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        inference_rope_scaling=args.inference_rope_scaling
    ))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def main():
    parser = argparse.ArgumentParser(description="用训练好的权重批量回答 jsonl 题库，并把答案写成 jsonl")
    parser.add_argument('--data', default=os.path.join(ROOT, 'eval_data', 'small_llm_knowledge_probe_142.jsonl'), type=str, help="题库 jsonl（每行至少含 id 和 question），默认是仓库里的 eval_data/small_llm_knowledge_probe_142.jsonl")
    parser.add_argument('--output', default='', type=str, help="答案输出 jsonl，默认 {save_dir}/{weight}_{题库文件名}_answers.jsonl，跟权重放一起，save_dir 指向 Drive 时结果也在 Drive")
    parser.add_argument('--save_dir', default=os.path.join(ROOT, 'out'), type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="每题最大生成长度，撞上限的题会在结果里标 truncated=true")
    parser.add_argument('--do_sample', default=1, type=int, choices=[0, 1], help="是否采样（0=贪心解码，结果可复现）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    parser.add_argument('--seed', default=42, type=int, help="随机种子，第 i 题用 seed+i，续跑与整体重跑的采样结果一致")
    parser.add_argument('--limit', default=0, type=int, help="只跑前 N 题（0=全部），用于冒烟测试")
    parser.add_argument('--resume', default=0, type=int, choices=[0, 1], help="续跑：跳过输出文件里已有的 id 并在其后追加（0=覆盖旧输出）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()

    if not args.output:
        stem = os.path.splitext(os.path.basename(args.data))[0]
        args.output = os.path.join(args.save_dir, f'{args.weight}_{stem}_answers.jsonl')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    rows = load_questions(args.data)
    if args.limit: rows = rows[:args.limit]
    done = load_done(args.output) if args.resume else set()
    todo = [(i, r) for i, r in enumerate(rows) if r['id'] not in done]
    print(f'题库 {len(rows)} 题，已完成 {len(done)}，本次待跑 {len(todo)} -> {args.output}')

    model, tokenizer = init_model(args)
    st_all, total_tokens = time.time(), 0
    with open(args.output, 'a' if args.resume else 'w', encoding='utf-8') as f:
        for n, (i, row) in enumerate(todo, 1):
            setup_seed(args.seed + i)
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": row['question']}], tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

            st = time.time()
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=bool(args.do_sample),
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
            )
            new_ids = generated_ids[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            total_tokens += len(new_ids)

            f.write(json.dumps({**row, 'answer': answer, 'gen_tokens': len(new_ids), 'truncated': new_ids[-1].item() != tokenizer.eos_token_id}, ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{n}/{len(todo)}] {row["id"]} ({len(new_ids) / (time.time() - st):.1f} tokens/s) {row["question"]}\n    {answer[:80]!r}')

    elapsed = time.time() - st_all
    print(f'完成：{len(todo)} 题，{total_tokens} tokens，耗时 {elapsed:.0f}s，已保存 {args.output}')


if __name__ == "__main__":
    main()
