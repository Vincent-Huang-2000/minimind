import os
import argparse

# MiniMind 训练数据集文件列表（文件名: 大小, 用途）
DATASET_FILES = {
    "pretrain_t2t_mini.jsonl": ("1.2GB", "轻量预训练数据，快速复现 Zero 模型"),
    "sft_t2t_mini.jsonl": ("1.6GB", "轻量 SFT 数据，快速复现 Zero 模型"),
    "pretrain_t2t.jsonl": ("10GB", "主线预训练数据，完整复现"),
    "sft_t2t.jsonl": ("14GB", "主线 SFT 数据，完整复现"),
    "rlaif.jsonl": ("24MB", "PPO/GRPO/CISPO 等强化学习数据"),
    "dpo.jsonl": ("53MB", "DPO 偏好训练数据"),
    "agent_rl.jsonl": ("86MB", "Agentic RL 主线数据（train_agent.py）"),
    "agent_rl_math.jsonl": ("18MB", "Agentic RL 数学补充数据"),
}
MINI_FILES = ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"]
MODELSCOPE_REPO = "gongjy/minimind_dataset"
HF_REPO = "jingyaogong/minimind_dataset"


def resolve_files(names):
    if "all" in names: return list(DATASET_FILES)
    files = []
    for name in names:
        name = name if name.endswith(".jsonl") else f"{name}.jsonl"
        if name not in DATASET_FILES: raise SystemExit(f"未知数据集文件: {name}，可用 --list 查看可选文件")
        if name not in files: files.append(name)
    return files


def download_from_modelscope(files, local_dir):
    from modelscope.hub.snapshot_download import dataset_snapshot_download
    dataset_snapshot_download(MODELSCOPE_REPO, local_dir=local_dir, allow_patterns=files)


def download_from_hf(files, local_dir):
    # 国内网络可先设置环境变量 HF_ENDPOINT=https://hf-mirror.com 走镜像
    from huggingface_hub import hf_hub_download
    for file in files:
        hf_hub_download(HF_REPO, file, repo_type="dataset", local_dir=local_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind 数据集下载")
    parser.add_argument('--files', nargs='+', default=MINI_FILES, help="要下载的文件名（可省略.jsonl后缀），all=全部，默认下载 mini 组合")
    parser.add_argument('--source', default='modelscope', type=str, choices=['modelscope', 'hf'], help="下载源（modelscope=国内推荐，hf=HuggingFace）")
    parser.add_argument('--local_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'), type=str, help="保存目录")
    parser.add_argument('--list', action='store_true', help="列出可下载的文件")
    args = parser.parse_args()

    if args.list:
        for name, (size, desc) in DATASET_FILES.items():
            print(f"{'*' if name in MINI_FILES else ' '} {name:<26}{size:>7}  {desc}")
        raise SystemExit(0)

    files = resolve_files(args.files)
    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    print(f"下载源: {args.source}，保存到: {local_dir}")
    for name in files: print(f"  - {name} ({DATASET_FILES[name][0]})")

    if args.source == 'modelscope': download_from_modelscope(files, local_dir)
    else: download_from_hf(files, local_dir)

    missing = [name for name in files if not os.path.exists(os.path.join(local_dir, name))]
    if missing: raise SystemExit(f"以下文件未下载成功: {', '.join(missing)}")
    print("下载完成")
