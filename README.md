# MedicalGPT CPT & SFT

基于 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 和 [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 的中文医疗领域适配实验。本项目在 RTX 4060 Laptop 8GB 上完成数据审计、继续预训练（CPT）、有监督微调（SFT）和对照评测。

> 医疗免责声明：模型和语料未经临床级验证，不可用于诊断、治疗或替代专业医生。loss、PPL 和 C-Eval 分数不代表临床安全性。

## 项目亮点

- 审计约 36 万条医疗百科文本和 195 万条中文医疗问答。
- 构建确定性的 CPT/SFT 子集，记录数据版本、SHA-256、排除原因和样本来源。
- 严格解析训练参数，避免上游未知参数被静默忽略。
- 严格控制 `Base → SFT` 与 `CPT → SFT` 的数据、顺序、随机种子和超参数。
- 使用独立测试集、C-Eval 医学三科和 McNemar 配对检验。
- 在单张 8GB 消费级显卡上完成两阶段 LoRA 训练。

## 实验流程

```text
原始医疗数据 → 数据审计/去重/污染检查 → 20k CPT
                                      ├→ Base → SFT
                                      └→ CPT  → SFT
                                             ↓
                                  独立测试集 + C-Eval
```

## 数据处理

CPT 从 361,420 条医疗百科文本中抽取 20,000 条候选，经清洗后保留 19,901 篇文档，得到 28,946 个 256-token block，共 7,410,176 个训练 token。

SFT 对 1,949,972 条中文医疗问答做流式审计，固定生成 20,000 条训练数据，并保留官方 validation/test 各 500 条。Base 与 CPT 分支使用完全相同的 token 化数据。

详细规则和限制见 [数据卡](data/DATA_CARD.md)、[CPT 审计](reports/CPT_AUDIT.md)和 [SFT 审计](reports/SFT_AUDIT.md)。原始数据和处理后数据不进入 Git。

## 正式结果

### CPT

| 指标 | Base | CPT | 相对变化 |
|---|---:|---:|---:|
| 验证 loss | 3.0320 | 2.8777 | -5.09% |
| 验证 PPL | 20.7393 | 17.7730 | -14.30% |
| 独立测试 loss | 3.0249 | 2.8747 | -4.96% |
| 独立测试 PPL | 20.5911 | 17.7201 | -13.94% |
| C-Eval 医学 | 51/90 | 52/90 | +1 题，不显著 |

CPT 明显改善同分布医疗文本的下一 token 预测能力，但没有充分证据证明其提升医学选择题能力。

### SFT

| 分支 | 验证 loss / PPL | 独立测试 loss / PPL | C-Eval 医学 |
|---|---:|---:|---:|
| Base → SFT | 2.6061 / 13.5463 | 2.6443 / 14.0729 | 52/90（57.78%） |
| CPT → SFT | 2.5992 / 13.4524 | 2.6351 / 13.9444 | 50/90（55.56%） |

CPT 前置使独立 SFT 测试集的回答 token loss 下降 0.347%、PPL 下降 0.913%，但 C-Eval 少答对 2 题，差异不显著（McNemar exact `p=0.7266`）。本轮只支持“CPT 带来小幅回答概率建模收益”，不支持“CPT 提升医学选择题正确率”。

完整说明见 [SFT 正式实验报告](reports/SFT_RESULTS_SHORT.md)。

## 训练配置

| 配置 | CPT | SFT |
|---|---:|---:|
| 方法 | LoRA | LoRA |
| rank / alpha / dropout | 8 / 16 / 0.05 | 8 / 16 / 0.05 |
| 最大长度 | 256 | 512 |
| 有效 batch | 8 | 4 |
| 学习率 | 2e-4 | 2e-5 |
| warmup steps | 181 | 250 |
| epoch / 更新次数 | 1 / 3,619 | 1 / 5,000 |
| 单分支耗时 | 约 53.5 分钟 | 约 37.5 分钟 |

SFT loss/PPL 只统计助手回答 token，不能与 CPT 全文 next-token PPL 直接比较。

## 已验证环境

```text
Windows 11
Python 3.13.14
PyTorch 2.14.0+cu126
Transformers 5.16.1
PEFT 0.20.0
Datasets 5.0.1
NVIDIA RTX 4060 Laptop GPU 8GB
```

固定版本：

- MedicalGPT commit：`ccc05f4b46442ecdefcc95d53aeedc9834d09dd6`
- Qwen2.5-0.5B revision：`060db6499f32faf8b98477b0a26969ef7d8b9987`

## 目录结构

```text
configs/          PowerShell 环境与正式运行入口
data_pipeline/    CPT/SFT 数据审计和确定性子集生成
eval/             语言模型测试与 C-Eval 评测
tools/            严格训练运行器和结果展示
data/             数据格式与数据卡，不含原始数据
reports/          审计报告和机器可读结果
```

本地专用目录 `.venv/`、`.cache/`、`models/`、`outputs/`、`data/raw/`、`data/processed/`、`data/eval/` 和 `vendor/` 均被 Git 忽略。

## 复现入口

```powershell
git clone https://github.com/shibing624/MedicalGPT.git vendor/MedicalGPT
git -C vendor/MedicalGPT checkout ccc05f4b46442ecdefcc95d53aeedc9834d09dd6
. .\configs\set_env.ps1

# CPT
.\.venv\Scripts\python.exe data_pipeline\clean_cpt_20k.py
.\.venv\Scripts\python.exe tools\run_cpt_stage.py base
.\.venv\Scripts\python.exe tools\run_cpt_stage.py cpt

# SFT
.\.venv\Scripts\python.exe data_pipeline\prepare_sft_subset.py
.\.venv\Scripts\python.exe tools\run_sft_stage.py base --mode formal
.\.venv\Scripts\python.exe tools\run_sft_stage.py cpt --mode formal

# 终端结果摘要
.\.venv\Scripts\python.exe tools\show_final_eval_results.py
.\.venv\Scripts\python.exe tools\show_sft_results.py
```

## AutoDL 上的 7B 对照实验

7B 分支固定使用 `Qwen/Qwen2.5-7B` revision `d149729398750b98c0af14eb82c78cfe92750796`，并复用与 0.5B 完全相同的 CPT/SFT 数据、随机种子、1 个 epoch、LoRA 配置和有效 batch。默认先尝试 BF16 LoRA；若 24GB 显卡在烟雾测试中 OOM，再安装 `requirements-qlora.txt` 并设置 `USE_QLORA=1`，不要在两条对照分支中混用精度方案。

```bash
# 仓库和私有数据包上传到 AutoDL 后，在仓库根目录执行
sha256sum -c transfer/autodl-data-v2.sha256
tar -xzf transfer/autodl-data-v2.tar.gz
bash scripts/setup_autodl.sh
bash scripts/check_autodl_env.sh
bash scripts/run_7b_smoke.sh

# 烟雾测试通过后再启动正式实验
bash scripts/run_7b_cpt.sh
bash scripts/run_7b_sft_base.sh
bash scripts/run_7b_sft_cpt.sh
bash scripts/run_7b_eval.sh
```

`transfer/autodl-data-v2.tar.gz` 由 `tools/package_autodl_data.py` 生成，包含处理后的 CPT/SFT 数据及评测数据，但被 `.gitignore` 排除，不能上传到公开仓库。模型、输出和上游源码同样不进入 Git。请选择 Python 3.11+、CUDA 可用且支持 BF16 的 AutoDL 镜像；脚本默认把缓存、模型和训练输出放在 `/root/autodl-tmp`，避免实例系统盘重置后丢失。

## 局限性

- 工程清洗不等于逐条临床事实认证。
- 结果来自 0.5B 模型，不能直接外推到 7B。
- C-Eval 仅使用公开 validation 的 90 道医学题，不是隐藏测试成绩。
- 无法完全排除基模预训练阶段见过公开评测内容。
- PPL 下降不等于医学事实性、安全性或诊疗能力提升。
- 尚未完成强化学习、RAG、系统化生成质量评测或临床专家评审。

## 致谢与许可

- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)，Apache-2.0。
- [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)，Apache-2.0。
- [shibing624/medical](https://huggingface.co/datasets/shibing624/medical)，仓库声明 Apache-2.0；具体原始内容来源仍需独立审查。
- [C-Eval](https://huggingface.co/datasets/ceval/ceval-exam)，CC BY-NC-SA 4.0。

本项目自身代码的根目录许可证将在正式发布前补充。上游源码、模型权重和原始数据不包含在本仓库中。
