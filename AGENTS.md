# AGENTS.md

## 项目用途

- 本仓库是 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 的个人 fork（`Vincent-Huang-2000/minimind`），用于学习和实践从零训练小型语言模型 MiniMind（主线约 64M 参数）。
- 覆盖完整训练链路：Pretrain → SFT → LoRA / DPO / PPO / GRPO / 蒸馏 / Agentic RL，核心代码均为 PyTorch 原生实现。
- 个人改动在 `custom` 分支上，`master` 跟随上游。

## 训练环境：只在 Google Colab 中训练

**所有训练都只在 Colab 上进行，不在本地机器上训练。** 本地（Windows 笔记本）只用来阅读、修改和提交代码。

对 agent 的要求：

- 不要在本地启动训练脚本，也不要为了"验证"在本地下载完整数据集；本地最多做语法 / import 检查或极小样例的冒烟测试。
- 给训练或性能优化建议时，以 Colab 环境为准，不要针对本地 GPU 或 Windows 做适配。
- Colab 为单卡环境，DDP / 多卡相关的优化优先级很低。
- Colab 运行时会断开、`/content` 下的文件会被清空：训练产物应保存到 Google Drive，长时间训练依赖 `--from_resume 1` 续训，改动保存 / 续训逻辑时要格外小心。
- Colab 分配的 GPU 型号不固定：T4 不原生支持 bf16，应使用 `--dtype float16`（脚本会自动启用 GradScaler）；L4 / A100 可使用默认的 `bfloat16`。

## 目录与运行约定

- 训练脚本在 `trainer/`，需在 `trainer/` 目录下运行，脚本中的 `../dataset`、`../out`、`../checkpoints` 都是相对路径。
- 推理权重输出到 `out/`（可用 `--save_dir` 修改）；续训文件写入 `checkpoints/`，该路径在脚本中写死，要放到 Drive 需建软链接或改代码。
- 数据集放在 `dataset/`，用 `python scripts/download_dataset.py` 下载（默认 mini 组合，`--list` 查看全部）；`dataset/*.jsonl` 已加入 `.gitignore`，不要提交数据文件。
- 评测：`python eval_llm.py --weight full_sft`。
