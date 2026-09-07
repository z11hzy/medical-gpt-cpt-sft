# MedicalGPT：中文医疗模型训练与 RLHF 全链路实验

本项目基于 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 和 Qwen2.5，完成中文医疗领域的 CPT/PT、SFT、DPO、PPO、GRPO 训练与评测。项目重点是记录可复现的工程链路、数据版本和诚实的实验结论，而不是宣称模型具备临床诊疗能力。

> 医疗免责声明：本项目仅用于研究和工程实践。模型未经临床验证，不能替代医生进行诊断、治疗、处方或急救决策。

## 实验主线

```text
医学文本清洗
    └──> CPT/PT
医疗问答清洗
    └──> SFT
          ├──> DPO
          ├──> Reward Model
          └──> PPO / GRPO
                    └──> 独立测试集 + C-Eval
```

## 已完成内容

- CPT/PT：使用 20k 医学文本子集，完成数据清洗、去重、训练和 PPL/loss 评测。
- SFT：完成 `Base → SFT` 与 `CPT → SFT` 两条 7B 对照分支。
- DPO：完成 7B SFT 后偏好优化，保留偏好准确率和 reward margin。
- PPO：在 0.5B 上完成 policy、reference、Reward Model、value model 和 KL 约束的真实 PPO 工程验证。
- GRPO：在 7B SFT 底座上完成 4-bit QLoRA GRPO，并用 500 条独立开放式测试集和 90 道 C-Eval 医学题进行正式评测。

## 主要结果

| 分支 | 测试 PPL | C-Eval 医学题 |
|---|---:|---:|
| Base | 12.2032 | 76/90 |
| CPT | 10.8903 | 77/90 |
| Base → SFT | 8.2940 | 75/90 |
| CPT → SFT | 8.3748 | 78/90 |

GRPO 正式评测中，SFT 与 GRPO 的 C-Eval 都是 `74/90 = 82.22%`；500 条开放式测试集上的自动奖励均分由 `0.1596` 降至 `0.0800`。因此当前结论是：RLHF 工程链路已跑通，但这版奖励模型和 GRPO 配置没有证明整体优于 SFT。

完整中文总结见：[最终实验报告](reports/FINAL_REPORT_ZH.md)。

## 仓库结构

```text
configs/          训练和评测配置
data/             数据格式、数据卡和使用说明
data_pipeline/    CPT/SFT 数据清洗与子集构造
eval/             语言模型和 C-Eval 评测
reports/          一份最终报告与正式机器结果
scripts/          正式训练/评测入口
tools/            训练、数据准备和结果处理工具
```

原始数据、模型权重、checkpoint、缓存和训练输出均不进入 Git。烟雾测试、临时 V1 实验和过程性报告也不作为公开项目成果保留。

## 复现主入口

```powershell
. .\configs\set_env.ps1

# 数据准备
.\.venv\Scripts\python.exe data_pipeline\clean_cpt_20k.py
.\.venv\Scripts\python.exe data_pipeline\prepare_sft_subset.py

# 7B CPT/PT 与 SFT
bash scripts/run_7b_cpt.sh
bash scripts/run_7b_sft_base.sh
bash scripts/run_7b_sft_cpt.sh
bash scripts/run_7b_eval.sh
```

AutoDL 环境说明见 [7B AutoDL 运行手册](docs/AUTODL_7B_RUNBOOK.md)。公开仓库只保存代码、配置、数据说明和脱敏后的结果摘要；运行前需要自行准备模型和处理后的数据。

## 数据与评测说明

- SFT 测试集和 C-Eval 只用于评测，不能反向用于训练偏好数据。
- C-Eval 使用公开医学 validation split 的 90 道题，不等同于临床验证。
- 自动 Reward Model 分数用于相对比较，不是独立医学裁判。
- 后续若继续优化，建议先重构医学偏好数据，再做 CoT SFT 冷启动和 GRPO。

## 致谢与许可证

- [MedicalGPT](https://github.com/shibing624/MedicalGPT)
- [Qwen2.5](https://huggingface.co/Qwen)
- [C-Eval](https://github.com/hailsong/ceval)

上游项目和数据集的许可证、归属与使用限制请以其官方说明为准。本仓库中的脚本和文档仅代表本项目实验代码。
