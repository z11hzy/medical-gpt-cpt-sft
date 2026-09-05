# GitHub 发布筛选清单

当前阶段不执行 `git add`、commit 或 push。本清单供第一次公开发布前人工检查。

## 建议进入仓库

- 根目录：`.gitignore`、`README.md`、`requirements.txt`、`requirements-autodl.txt`、`requirements-qlora.txt`，以及待创建的 `LICENSE`。
- 正式配置：`set_env.ps1`、`run_pt_full20k.ps1`、`run_final_eval.ps1`、`run_ceval_medical.ps1`。
- 数据流水线：`clean_cpt_20k.py`、`prepare_sft_subset.py`、`validate_jsonl.py`。
- 训练工具：`audit_cpt.py`、`run_cpt_stage.py`、`run_sft_stage.py`、`merge_lora.py` 和三个 `show_*_results.py`。
- AutoDL：`scripts/` 下的环境检查、7B 烟雾测试、正式 CPT/SFT 和评测入口，以及 `tools/check_autodl_env.py`、`tools/package_autodl_data.py`。
- 评测：`test_lm.py`、`ceval_medical.py`。
- 文档：数据卡、schemas、CPT/SFT 审计、SFT 最终短报告、实验模板。
- 结果：正式 CPT/SFT 的 txt 与 JSON、四份最终 C-Eval JSON、CPT 独立测试 JSON。

## 不进入第一版仓库

- `.venv/`、`.cache/`、`.tmp/`、`__pycache__/`。
- `models/`、`outputs/`、原始数据、处理后数据和 C-Eval 原始文件。
- `transfer/` 私有数据传输包及其校验文件；只上传到 AutoDL，不进入公开 GitHub。
- `vendor/MedicalGPT/`；由 README 中固定 commit 的下载方式替代。
- smoke、CPT v1、旧合并模型、optimizer、checkpoint 和 TensorBoard 日志。
- 未正式运行的 RM/RLOO 草稿和 tokenizer 演示。
- 过时的 `NEXT_STEPS.md`，以及尚未完成正式评分的 safety 生成脚本和案例。

## 发布前待办

- [ ] 更新 `data/DATA_CARD.md` 的正式 v2 与测试集状态。
- [ ] 更新或重写 `docs/NEXT_STEPS.md`。
- [x] 新增 7B 正式 SFT 与一键评测脚本。
- [ ] 将 SFT 测试原始指标整理为 `reports/results/` 下的 JSON。
- [x] 固定当前核心依赖版本，并将 QLoRA 依赖拆为可选项。
- [x] 增加固定 revision 的 7B 模型下载脚本和可校验的私有数据打包工具。
- [ ] 在实际 AutoDL 实例上完成从 clone 到烟雾测试的端到端验证。
- [ ] 增加根目录 `LICENSE` 和第三方归属说明。
- [ ] 运行密钥扫描与待提交文件大小检查。
- [ ] 人工检查 README 后再决定是否暂存和提交。
