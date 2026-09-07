"""Audit downloaded AutoDL metrics and build a Chinese report without GPU dependencies."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / 'transfer/7b-results'
OUT = BACKUP / 'medical-gpt-outputs'
EVAL = BACKUP / 'medical-gpt-cpt-sft/reports/results/7b'
DEST = ROOT / 'reports/results/7b'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def paired(a, b):
    def index(rows):
        result = {(r['subject'], r['id']): r for r in rows}
        assert len(result) == len(rows), 'Duplicate question IDs'
        return result
    x, y = index(a['predictions']), index(b['predictions'])
    assert x.keys() == y.keys()
    left = right = 0
    for key in x:
        assert x[key]['gold'] == y[key]['gold']
        ca = x[key]['prediction'] == x[key]['gold']
        cb = y[key]['prediction'] == y[key]['gold']
        left += ca and not cb
        right += cb and not ca
    n = left + right
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(left, right) + 1)) / 2**n)
    return dict(left_only_correct=left, right_only_correct=right, exact_p=p)


def main():
    stages = ['pt-base-full-eval-7b-v2', 'pt-full20k-7b-v2', 'sft-base-20k-7b-v2', 'sft-cpt-20k-7b-v2']
    names = ['Base', 'CPT', 'Base -> SFT', 'CPT -> SFT']
    ce = [read(EVAL / f'ceval-{s}.json') for s in ['base', 'cpt', 'sft-base', 'sft-cpt']]
    valid = [read(OUT / s / 'eval_results.json') for s in stages]
    test = [read(EVAL / 'test-lm-base.json'), read(EVAL / 'test-lm-cpt.json'),
            read(OUT / 'eval-sft-base-test-7b-v2/eval_results.json'),
            read(OUT / 'eval-sft-cpt-test-7b-v2/eval_results.json')]
    for i, stage in enumerate(stages):
        progress = read(OUT / stage / 'progress.json')
        assert progress['status'] == 'complete'
        if i:
            state = read(OUT / stage / 'trainer_state.json')
            assert state['global_step'] == (3619 if i == 1 else 5000)
            assert abs(state['epoch'] - 1) < 1e-6
        for value in (valid[i], test[i]):
            assert math.isfinite(value['eval_loss'])
            assert math.isclose(math.exp(value['eval_loss']), value['perplexity'], rel_tol=1e-6)
        assert len(ce[i]['predictions']) == ce[i]['total'] == 90
        assert sum(r['prediction'] == r['gold'] for r in ce[i]['predictions']) == ce[i]['correct']
    a, b = [read(OUT / s / 'dataset_identity.json') for s in stages[2:]]
    for key in ['train_tokenized_sha256', 'valid_tokenized_sha256']:
        assert a[key] == b[key]
    assert test[0]['packed_test_sha256'] == test[1]['packed_test_sha256']
    cpt_a, cpt_b = [read(OUT / s / 'dataset_identity.json') for s in stages[:2]]
    assert cpt_a['eval_sha256'] == cpt_b['eval_sha256']
    log = (OUT / 'overnight.log').read_text(encoding='utf-8')
    assert 'ALL_EXPERIMENTS_COMPLETE' in log and 'PIPELINE_EXIT=0' in log
    stats = {'CPT_vs_Base': paired(ce[0], ce[1]), 'CPT_SFT_vs_Base_SFT': paired(ce[2], ce[3])}
    runtime = {names[i]: read(OUT / stages[i] / 'train_results.json')['train_runtime'] / 60 for i in [1, 2, 3]}
    rows = [dict(branch=names[i], validation=valid[i], test=test[i], ceval_correct=ce[i]['correct'], ceval_total=90) for i in range(4)]
    DEST.mkdir(parents=True, exist_ok=True)
    # Preserve raw JSON metrics and provenance; never copy weights or data into reports.
    for path in EVAL.glob('*.json'):
        (DEST / path.name).write_bytes(path.read_bytes())
    for stage in stages + ['eval-sft-base-test-7b-v2', 'eval-sft-cpt-test-7b-v2']:
        for filename in ['eval_results.json', 'train_results.json', 'dataset_identity.json', 'run_config.json', 'progress.json']:
            path = OUT / stage / filename
            if path.exists():
                target = DEST / stage / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    summary = dict(rows=rows, paired_tests=stats, training_minutes=runtime, data_identity_verified=True)
    (DEST / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['=' * 88, 'Qwen2.5-7B | 20k CPT / SFT | BF16 LoRA | RTX 4090',
             'Branch          ValidLoss  ValidPPL   TestLoss   TestPPL   C-Eval', '-' * 88]
    for i, name in enumerate(names):
        lines.append(f"{name:<15} {valid[i]['eval_loss']:9.4f} {valid[i]['perplexity']:9.4f} {test[i]['eval_loss']:10.4f} {test[i]['perplexity']:9.4f}  {ce[i]['correct']}/90 ({ce[i]['micro_accuracy']*100:.2f}%)")
    lines += ['-' * 88, f"CPT vs Base: McNemar exact p={stats['CPT_vs_Base']['exact_p']:.4f}",
              f"CPT->SFT vs Base->SFT: McNemar exact p={stats['CPT_SFT_vs_Base_SFT']['exact_p']:.4f}",
              'CPT / SFT PPL use different label masks; compare only within each pair.',
              'Verified: full epochs, paired data identity, 90 matched C-Eval questions.', '=' * 88]
    terminal = '\n'.join(lines) + '\n'
    (DEST / 'comparison.txt').write_text(terminal, encoding='utf-8')
    cpt_pct = (test[1]['perplexity'] / test[0]['perplexity'] - 1) * 100
    sft_pct = (test[3]['perplexity'] / test[2]['perplexity'] - 1) * 100
    report = ['# Qwen2.5-7B 正式实验结果', '',
              '全部实验于 2026-09-06 08:47:06（北京时间）成功完成，日志退出码为 0，随后触发 AutoDL 自动关机。流程约 6 小时 18 分钟，不含模型下载。', '',
              '## 配置与核验', '',
              'RTX 4090；Python 3.12.3；PyTorch 2.5.1+cu124；Transformers 5.16.1；BF16 LoRA，rank=8、alpha=16、dropout=0.05，开启梯度检查点。', '',
              'CPT 使用清洗后的 19,901 篇文本（28,946 个 256-token 块），有效 batch 8，1 epoch / 3,619 步。两条 SFT 使用同样的 20,000 条问答，有效 batch 4，长度 512，1 epoch / 5,000 步。', '',
              '已核验完整训练步数、epoch、Base/CPT 验证块哈希、两条 SFT 的训练及验证 token 哈希、CPT 测试块哈希及 C-Eval 90 题 ID/答案配对。', '',
              '## 指标', '', '| 分支 | 验证 loss | 验证 PPL | 测试 loss | 测试 PPL | C-Eval 医学 |', '|---|---:|---:|---:|---:|---:|']
    for i, name in enumerate(names):
        report.append(f"| {name} | {valid[i]['eval_loss']:.4f} | {valid[i]['perplexity']:.4f} | {test[i]['eval_loss']:.4f} | {test[i]['perplexity']:.4f} | {ce[i]['correct']}/90 ({ce[i]['micro_accuracy']*100:.2f}%) |")
    report += ['', 'CPT 指标统计全文 next-token；SFT 指标仅统计回答 token，不能将两类 PPL 直接作优劣比较。', '', '## 结论', '',
               f'- CPT 将医疗文本独立测试 PPL 改变 {cpt_pct:+.2f}%，说明同分布文本建模改善。C-Eval 从 76/90 到 77/90，配对检验 p={stats["CPT_vs_Base"]["exact_p"]:.4f}。',
               f'- CPT→SFT 的回答测试 PPL 相对 Base→SFT 改变 {sft_pct:+.2f}%（略差），但 C-Eval 从 75/90 到 78/90，增加 3 题；配对检验 p={stats["CPT_SFT_vs_Base_SFT"]["exact_p"]:.4f}。',
               '- 本轮不能声称 CPT 稳定提升 SFT 效果：回答概率指标略退化，选择题差异需结合配对检验及更大评测集解读。', '', '## 实测训练耗时', '']
    report += [f'- {name}：{minutes:.2f} 分钟。' for name, minutes in runtime.items()]
    report += ['', '## 与 0.5B 实验的关系', '',
               '使用相同清洗数据的 0.5B 实验，四组 C-Eval 分别为 51、52、52、50 / 90；7B 为 76、77、75、78 / 90。7B 分数整体更高，但不能将规模、基模预训练差异和本次 CPT 的贡献混为一谈。', '',
               '0.5B 的 CPT→SFT 回答测试 PPL 为 13.9444，略优于直接 SFT 的 14.0729；7B 的变化方向相反。因此没有跨规模一致的 CPT→SFT 收益。两次运行硬件、PyTorch 版本、micro-batch 和梯度检查点不同，不属于逐位可复现实验。', '',
               '## 局限与产物', '',
               '- 单次 seed=42；C-Eval 是公开 validation 医学三科 90 题，不是隐藏测试集。不能证明临床安全性或统计上稳定的能力提升。',
               '- 指标和日志归档保存在本地 transfer/7b-results.tar.gz；SHA-256 为 6ca008db58ac6e6b7cfb826eebce413277f99e9df5c906158e19307754b2c775。',
               '- 本次备份不含 adapter 权重和中间 checkpoint，它们仍保留在 AutoDL。',
               '- 原始指标与对比 JSON 位于 [results/7b](results/7b/)，截图用文本位于 [comparison.txt](results/7b/comparison.txt)。', '']
    (ROOT / 'reports/QWEN25_7B_RESULTS_ZH.md').write_text('\n'.join(report), encoding='utf-8')
    print(terminal)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
