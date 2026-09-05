# SFT 正式实验短报告

## 实验设置

- 基座：Qwen2.5-0.5B。
- 对照：`Base → SFT` 与 `CPT → SFT`。后一分支从已完成的 20k 医疗 CPT LoRA 继续训练。
- 两分支严格共享同一份清洗后的 20,000 条训练集、500 条验证集、500 条测试集，token 化数据指纹已核验一致。
- 训练：LoRA rank 8 / alpha 16 / dropout 0.05，最大长度 512，有效 batch 4，学习率 2e-5，5% warmup，1 epoch（5,000 次参数更新）。SFT loss 只统计助手回答 token。

## 结果

| 分支 | 验证 loss | 验证 PPL | 独立测试 loss | 独立测试 PPL | C-Eval 医学 |
|---|---:|---:|---:|---:|---:|
| Base → SFT | 2.6061 | 13.5463 | 2.6443 | 14.0729 | 52/90（57.78%） |
| CPT → SFT | 2.5992 | 13.4524 | 2.6351 | 13.9444 | 50/90（55.56%） |

CPT 前置相较 Base→SFT，在独立 SFT 测试集上使回答 loss 下降 0.347%、PPL 下降 0.913%，属于稳定但幅度较小的语言建模收益。两条分支训练均完整达到 5,000 步和 1 epoch，耗时分别约 37.47 和 37.53 分钟。

C-Eval 上 CPT→SFT 少答对 2 题；两模型有 11 题预测变化，其中 Base→SFT 独有 5 题答对、CPT→SFT 独有 3 题答对，McNemar 精确检验 p=0.7266，差异不显著。下降集中在 `clinical_medicine`（13/22 → 11/22），不能据此认定 CPT 分支更差，也不能声称 CPT 提升了医学选择题能力。

## 结论与限制

本轮结果支持：CPT 为后续 SFT 带来小幅的回答 token 概率建模收益；但没有证据显示其提升 C-Eval 医学问答正确率。C-Eval 此处是公开 validation 的 90 道医学题，不是隐藏测试集；训练语料没有逐条完成临床事实核验，因此模型与这些指标都不能用于证明临床安全性或诊疗能力。

终端复现摘要：

```powershell
. .\configs\set_env.ps1
.\.venv\Scripts\python.exe tools\show_sft_results.py
```
