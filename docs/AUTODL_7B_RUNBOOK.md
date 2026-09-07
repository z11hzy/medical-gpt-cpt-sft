# AutoDL Qwen2.5-7B 运行手册

## 目标与对照约束

7B 实验复用 0.5B 正式实验的同一份 CPT/SFT 数据和划分，并保持 seed 42、1 epoch、LoRA `rank=8 / alpha=16 / dropout=0.05`、CPT 有效 batch 8、SFT 有效 batch 4。7B 的两条 SFT 分支必须使用相同精度方案；不能一条用 BF16 LoRA、另一条用 QLoRA。

固定版本：

- MedicalGPT commit：`ccc05f4b46442ecdefcc95d53aeedc9834d09dd6`
- Qwen2.5-7B revision：`d149729398750b98c0af14eb82c78cfe92750796`

## 租卡建议

先选择单张 24GB、支持 BF16 的 NVIDIA GPU 和 Python 3.11+ 镜像。若显存不足，使用 QLoRA 配置重跑两条分支，并保持两条分支的精度方案一致。

## 上传与初始化

先把代码推到 GitHub，再在 AutoDL 中 clone。数据包不进公开 GitHub，应单独上传到 clone 后仓库的 `transfer/` 目录。

```bash
git clone https://github.com/z11hzy/medical-gpt-cpt-sft.git
cd medical-gpt-cpt-sft

sha256sum -c transfer/autodl-data-v2.sha256
tar -xzf transfer/autodl-data-v2.tar.gz
bash scripts/setup_autodl.sh
bash scripts/check_autodl_env.sh
```

环境检查必须显示 `status: PASS`，并核对 GPU、MedicalGPT commit 和五个 CPT/SFT split 的 SHA-256。

## 运行顺序

```bash
# 正式 CPT：先记录 Base 验证指标，再训练 CPT
bash scripts/run_7b_cpt.sh

# 两条 SFT 对照分支
bash scripts/run_7b_sft_base.sh
bash scripts/run_7b_sft_cpt.sh

# 独立测试集和 C-Eval
bash scripts/run_7b_eval.sh
```

## RLHF 后训练与正式评测

在完成 SFT 并准备偏好数据后，可按下面顺序运行 DPO、奖励模型和 GRPO：

```bash
# 7B SFT → DPO
bash scripts/run_7b_dpo.sh

# 0.5B Reward Model（供 PPO/GRPO 使用）
bash scripts/run_05b_rm.sh

# 7B GRPO 训练和独立评测
bash scripts/run_formal_eval_grpo.sh
```

PPO 的可复现实验入口是 `tools/run_true_ppo.py`，使用与报告一致的 0.5B 工程验证配置。RLHF 结果摘要位于 `reports/results/rlhf/`。

每个训练目录都有 `run_config.json`、`progress.json`、`dataset_identity.json` 和 Trainer 指标。正式输出默认位于 `/root/autodl-tmp/medical-gpt-outputs`；代码仓库被删除或实例被关机前，应下载这些目录以及 `reports/results/7b/`。

## OOM 时切换 QLoRA

如果 BF16 正式训练 OOM，执行：

```bash
source scripts/set_env.sh
"$PYTHON_BIN" -m pip install -r requirements-qlora.txt
export USE_QLORA=1
bash scripts/check_autodl_env.sh
```

环境检查通过后，在运行每一个正式脚本的同一终端中保留 `USE_QLORA=1`。如果已经产生同名正式输出目录，运行器会拒绝覆盖；先保存失败日志，再明确改名或移动旧目录，不要直接删除实验记录。

## 数据传输包

本地重新生成命令：

```powershell
E:\medical-gpt-portfolio\.venv\Scripts\python.exe tools\package_autodl_data.py
```

生成物位于 `transfer/`，包含压缩包、SHA-256 文件和逐文件 manifest。`transfer/` 已被 Git 忽略。
