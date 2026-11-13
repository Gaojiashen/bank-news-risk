"""对原始日度新闻进行蓄水池抽样，并展开目标银行的上下文。"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from code.common.paths import ROOT, NEWS
from code.common.banks import load_aliases, mentioned_banks, split_sentences

def draw_articles(source,count,used,seed):
    rng=random.Random(seed);chosen=[];seen=0
    for path in sorted(Path(source).glob("[12][0-9][0-9][0-9]/[01][0-9]/????????.csv")):
        relative=path.relative_to(source).as_posix()
        with path.open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle)
            if not {"Title","NewsContent"}.issubset(reader.fieldnames or []):raise ValueError(f"Invalid daily news: {path}")
            for number,row in enumerate(reader,2):
                key=(relative,str(number))
                if key in used:continue
                text=(row["Title"].strip()+"\n"+row["NewsContent"]).strip()
                if not text:continue
                item=dict(source_file=relative,source_row=str(number),content=text)
                seen+=1
                if len(chosen)<count:chosen.append(item)
                else:
                    j=rng.randrange(seen)
                    if j<count:chosen[j]=item
    if len(chosen)<count:raise ValueError(f"Only {seen} unsampled articles; requested {count}")
    rng.shuffle(chosen);return chosen

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=NEWS)
    parser.add_argument("--count",type=int,default=1000)
    parser.add_argument("--batch",type=int,required=True)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.count<1 or args.batch<1:parser.error("Count and batch must be positive")
    csv.field_size_limit(2**31-1)
    folder=ROOT/"data/annotation/pilot_batches";folder.mkdir(parents=True,exist_ok=True)
    output=folder/f"batch_{args.batch:03d}.csv"
    if output.exists():raise FileExistsError(output)
    history_path=ROOT/"data/provenance/sampled_articles_daily.json"
    history=json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    used={(r["source_file"],str(r["source_row"])) for r in history}
    chosen=draw_articles(args.source,args.count,used,args.seed)
    aliases=load_aliases()
    with output.open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["NewsContent","BankName","SourceFile","SourceRow","SentenceIndex","分析依据","相关/不相关","积极中性消极"])
        for row in chosen:
            sentences=split_sentences(row["content"])
            for i,sentence in enumerate(sentences):
                for bank in mentioned_banks(sentence,aliases):
                    context="".join(sentences[max(0,i-1):i+2])
                    writer.writerow([context,bank,row["source_file"],row["source_row"],i+1,"","",""])
    history.extend(dict(source_file=r["source_file"],source_row=r["source_row"],batch=args.batch) for r in chosen)
    from code.news_features.extract_news_embeddings import write_json
    write_json(history_path,history)
    print(f"Saved {len(chosen)} articles for AI annotation: {output}")

if __name__=="__main__":main()
