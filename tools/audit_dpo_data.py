"""Read-only audit of fixed upstream preference data; emit counts, not training data."""
import collections
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def norm(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s)).casefold()

def main():
    heldout = set()
    for split in ['valid', 'test']:
        path = ROOT / f'data/processed/sft-clean-20k-v2/{split}/{split}.jsonl'
        for line in path.read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            row = json.loads(line)
            heldout.update(norm(m['value']) for m in row['conversations'] if m['from'] == 'human')
    import pandas as pd
    ceval = set()
    for path in (ROOT / 'data/eval/ceval').glob('*/val-*.parquet'):
        ceval.update(norm(s) for s in pd.read_parquet(path)['question'])
    report = {'revision': 'd7a9a7061bf647aa14cd6592dda4e24785ad3b29', 'files': {},
              'checks_scope': 'All rows; normalized exact matches only, no semantic or medical label verification'}
    all_prompts = {}
    for lang in ['zh', 'en']:
        path = ROOT / f'data/raw/dpo-en-zh-20k/dpo_{lang}.jsonl'
        counts = collections.Counter()
        prompts, triples = collections.Counter(), collections.Counter()
        chosen_lengths, rejected_lengths, total_lengths = [], [], []
        for line in path.read_text(encoding='utf-8').split('\n'):
            if not line.strip():
                continue
            counts['rows'] += 1
            try:
                row = json.loads(line)
            except ValueError:
                counts['invalid_json'] += 1
                continue
            fields = ['question', 'response_chosen', 'response_rejected']
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
                counts['invalid_required_fields'] += 1
                continue
            history = row.get('history', [])
            if not isinstance(history, list) or any(not isinstance(h, list) or len(h) != 2 or any(not isinstance(t, str) for t in h) for h in history):
                counts['invalid_history'] += 1
                continue
            if not isinstance(row.get('system', ''), str):
                counts['invalid_system'] += 1
                continue
            q,c,r = [norm(row[k]) for k in fields]
            prompt = json.dumps([norm(row.get('system', '')), [[norm(t) for t in h] for h in history], q], ensure_ascii=False)
            prompts[prompt] += 1
            triples[(prompt,c,r)] += 1
            counts['same_chosen_rejected'] += c == r
            counts['has_history'] += bool(history)
            counts['has_system'] += bool(row.get('system', ''))
            counts['replacement_character'] += '\ufffd' in json.dumps(row,ensure_ascii=False)
            counts['sft_heldout_question_exact_overlap'] += q in heldout
            counts['ceval_question_exact_overlap'] += q in ceval
            counts['chosen_longer_chars'] += len(row['response_chosen']) > len(row['response_rejected'])
            chosen_lengths.append(len(row['response_chosen']))
            rejected_lengths.append(len(row['response_rejected']))
            context = len(row.get('system','')) + sum(len(t) for h in history for t in h) + len(row['question'])
            total_lengths.append(context + max(len(row['response_chosen']),len(row['response_rejected'])))
        counts['duplicate_prompt_excess'] = sum(v-1 for v in prompts.values())
        counts['duplicate_pair_excess'] = sum(v-1 for v in triples.values())
        counts['reversed_preference_pairs'] = sum((p,r,c) in triples for p,c,r in triples if c!=r)//2
        ordered = sorted(total_lengths)
        report['files'][lang] = {'bytes': path.stat().st_size, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'counts':dict(counts), 'mean_chosen_chars':statistics.mean(chosen_lengths),
            'mean_rejected_chars':statistics.mean(rejected_lengths),
            'longer_sequence_chars_p50':statistics.median(ordered),
            'longer_sequence_chars_p90':ordered[int((len(ordered)-1)*.9)],
            'longer_sequence_chars_max':max(ordered)}
        all_prompts[lang] = set(prompts)
    report['cross_language_exact_prompt_overlap'] = len(all_prompts['zh'] & all_prompts['en'])
    target = ROOT / 'reports/results/dpo-data-audit.json'
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
