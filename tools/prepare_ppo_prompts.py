"""Create deterministic prompt-only PPO train/eval splits from SFT data."""
import argparse, json, random, re, unicodedata
from pathlib import Path
from transformers import AutoTokenizer

def norm(s): return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).casefold()
def convert(path, tokenizer):
    out=[]; seen=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row=json.loads(line); messages=[]
        system=row.get("system_prompt", "")
        if system: messages.append({"role":"system","content":system})
        for turn in row.get("conversations", []):
            if turn.get("from") == "human":
                text=turn.get("value", "").strip(); key=norm(text)
                if text and key not in seen:
                    prompt=tokenizer.apply_chat_template(messages + [{"role":"user","content":text}],
                                                         tokenize=False, add_generation_prompt=True)
                    out.append({"prompt":prompt}); seen.add(key)
                break
    return out

p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,required=True); p.add_argument("--validation",type=Path,required=True)
p.add_argument("--tokenizer",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--train-size",type=int,default=1000); p.add_argument("--eval-size",type=int,default=100)
a=p.parse_args(); tok=AutoTokenizer.from_pretrained(a.tokenizer)
train=convert(a.train,tok); valid=convert(a.validation,tok); random.Random(20260907).shuffle(train)
a.output.mkdir(parents=True,exist_ok=True)
for name,rows in (("train",train[:a.train_size]),("validation",valid[:a.eval_size])):
    with (a.output/f"{name}.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
print(json.dumps({"train":min(len(train),a.train_size),"validation":min(len(valid),a.eval_size)}))
