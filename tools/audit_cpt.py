"""Read-only audit of source data and configuration; writes results only."""
from __future__ import annotations
import collections
import hashlib
import importlib.util
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, HfArgumentParser
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC', s).lower() if c.isalnum())
def read(path):
    texts, errors = [], []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            t = json.loads(line)['text']
            if not isinstance(t, str) or not t.strip(): raise ValueError('empty/non-string text')
            texts.append(t)
        except Exception as e: errors.append({'line': i, 'error': str(e)})
    return texts, errors
def emit(stage): print(stage, flush=True)
def main():
    result = {}
    spec = importlib.util.spec_from_file_location('medical_pt_audit', ROOT/'vendor/MedicalGPT/training/pretraining.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    script = (ROOT/'configs/run_pt_0_5b_v1.ps1').read_text(encoding='utf-8')
    cli = []
    for name, value in re.findall(r'(--[\w_]+)(?:\s+("[^"]*"|[^\s`]+))?', script):
        cli.append(name)
        if value and not value.startswith('--'):
            cli.append(value.strip('"').replace('$root', str(ROOT)))
    # Reconstruct using tokenization so consecutive boolean flags remain intact.
    command = script[script.index('  --model_name_or_path'):script.index('if ($LASTEXITCODE')]
    cli = [x.strip('"').replace('$root', str(ROOT)) for x in re.findall(r'"[^"]*"|[^\s`]+', command)]
    parser = HfArgumentParser((mod.ModelArguments, mod.DataArguments, mod.Seq2SeqTrainingArguments, mod.ScriptArguments))
    parsed = parser.parse_args_into_dataclasses(cli, return_remaining_strings=True)
    args = parsed[2]
    result['parameters'] = {'ignored_arguments': parsed[-1], 'warmup_steps': args.warmup_steps,
        'effective_warmup_for_500_steps': args.get_warmup_steps(500),
        'effective_batch': args.per_device_train_batch_size * args.gradient_accumulation_steps,
        'optimizer': str(args.optim), 'bf16': args.bf16, 'gradient_checkpointing': args.gradient_checkpointing,
        'max_train_samples': parsed[1].max_train_samples, 'max_eval_samples': parsed[1].max_eval_samples}
    result['parameters']['proposed_fractional_warmup_check'] = math.ceil(500 * 0.05)
    emit('parameter audit complete')
    paths = {'train': ROOT/'data/processed/pretrain/train/train.jsonl',
        'valid': ROOT/'data/processed/pretrain/valid/valid.jsonl'}
    texts, normalized = {}, {}
    result['data'] = {}
    tok = AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B', use_fast=False, local_files_only=True)
    result['tokenizer'] = {'class': type(tok).__name__, 'is_fast': tok.is_fast, 'eos': tok.eos_token_id}
    for split, path in paths.items():
        texts[split], errors = read(path)
        ts = texts[split]
        ns = [norm(t) for t in ts]
        normalized[split] = ns
        counts = collections.Counter(ns)
        lens, packed, dropped, covered, remaining_cap = [], 0, 0, 0, (4000 if split=='train' else 500)*256
        for start in range(0,len(ts),1000):
            ids = tok(ts[start:start+1000])['input_ids']
            sizes = [len(x)+(not x or x[-1]!=tok.eos_token_id) for x in ids]
            lens.extend(map(len, ids))
            total = sum(sizes)
            kept = total//256*256 if total>=256 else total
            packed += math.ceil(kept/256)
            dropped += total-kept
            budget = min(kept, remaining_cap)
            pos = 0
            for n in sizes:
                if pos < budget: covered += 1
                pos += n
            remaining_cap -= budget
        result['data'][split] = {'documents': len(ts), 'json_errors': errors,
            'raw_exact_duplicate_excess': len(ts)-len(set(ts)),
            'normalized_duplicate_excess': len(ns)-len(counts),
            'duplicate_groups': sum(n>1 for n in counts.values()),
            'token_count_without_eos': sum(lens), 'packed_blocks': packed, 'packed_input_tokens':packed*256,
            'dropped_tail_tokens':dropped, 'documents_touched_by_current_cap':covered,
            'token_length_percentiles':dict(zip(['min','p50','p90','p95','p99','max'],map(float,np.percentile(lens,[0,50,90,95,99,100])))),
            'replacement_character_documents':sum('\ufffd' in t for t in ts),
            'short_under_20_chars':sum(len(t)<20 for t in ts),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        emit(f'{split} count/tokenization complete: {result["data"][split]}')
    trainset = set(normalized['train'])
    result['overlap'] = {'valid_exact_normalized_matches': [i+1 for i,t in enumerate(normalized['valid']) if t in trainset]}
    subjects = ['basic_medicine','clinical_medicine','physician']
    ceval = []
    for subject in subjects:
        frame = pd.read_parquet(ROOT/f'data/eval/ceval/{subject}/val-00000-of-00001.parquet')
        for row in frame.to_dict('records'):
            q = norm(row['question'])
            hits = [i+1 for i,t in enumerate(normalized['train']) if q and q in t]
            if hits: ceval.append({'subject':subject,'id':int(row['id']),'question_chars':len(q),'train_lines':hits})
    result['overlap']['ceval_question_substring_matches'] = ceval
    emit('exact and C-Eval stem overlap complete')
    # Candidate screening, not a semantic/factual quality verdict.
    rules = {'absolute_claim':r'根治|包治|百分之百|百分百|无副作用|保证治愈|一定能治',
        'folk_remedy':r'偏方|秘方|祖传', 'contact_or_promotion':r'加微信|加QQ|联系电话|咨询热线|点击购买',
        'phone_like':r'(?<!\d)1[3-9]\d{9}(?!\d)', 'url':r'https?://|www\.',
        'hypertension_infertility_cooccurrence':r'高血压[\s\S]{0,100}不孕|不孕[\s\S]{0,100}高血压'}
    result['review_candidates'] = {}
    for key, pattern in rules.items():
        lines = [i+1 for i,t in enumerate(texts['train']) if re.search(pattern,t)]
        result['review_candidates'][key] = {'count':len(lines),'first_lines':lines[:15]}
    emit('lexical review flags complete')
    vec = HashingVectorizer(analyzer='char',ngram_range=(3,4),n_features=2**20,alternate_sign=False,norm=None,dtype=np.float32)
    idf = TfidfTransformer()
    matrix = idf.fit_transform(vec.transform(normalized['train']))
    nn = NearestNeighbors(n_neighbors=2,metric='cosine',algorithm='brute',n_jobs=2).fit(matrix)
    pairs = set()
    for start in range(0,matrix.shape[0],250):
        distances, indices = nn.kneighbors(matrix[start:start+250])
        for local,(ds,ixs) in enumerate(zip(distances,indices)):
            i = start+local
            for d,j in zip(ds,ixs):
                if i!=j and d<=0.05 and normalized['train'][i]!=normalized['train'][j]:
                    pairs.add((min(i+1,int(j)+1),max(i+1,int(j)+1)))
        if start%5000==0: emit(f'near-duplicate screening {start}/{matrix.shape[0]}')
    distances, indices = nn.kneighbors(idf.transform(vec.transform(normalized['valid'])), n_neighbors=1)
    result['near_duplicates'] = {'method':'char 3/4-gram hashed TF-IDF, 2**20 buckets, cosine>=0.95, top-2 neighbors; candidate screening only',
        'train_nonexact_candidate_pairs':len(pairs),'train_pair_examples':sorted(pairs)[:25],
        'valid_train_candidates':[{'valid_line':i+1,'train_line':int(indices[i,0])+1,'similarity':float(1-d[0])} for i,d in enumerate(distances) if d[0]<=0.05]}
    blocks = result['data']['train']['packed_blocks']
    result['full_run_estimate'] = {'steps_at_effective_batch_8':math.ceil(blocks/8),
        'warmup_steps_at_5_percent':math.ceil(math.ceil(blocks/8)*0.05),
        'minutes_linear_extrapolation':433.0604*(blocks/4000)/60,
        'note':'Measured v1 scaled by block count; not a guarantee; eval/saving/load/thermals differ.'}
    output = ROOT/'reports/results/cpt-audit.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
