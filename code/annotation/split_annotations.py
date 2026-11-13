"""划分完成AI标注的文章，确保同篇文章的所有银行片段属于同一集合。"""
import argparse
from collections import defaultdict
import csv
import random
from pathlib import Path
from code.common.paths import ROOT
from code.common.text_training import COLUMNS

def partition(rows,seed=20261101):
    groups=defaultdict(list);seen=set()
    for row in rows:
        key=tuple(row[k] for k in ("SourceFile","SourceRow","SentenceIndex","BankName"))
        if key in seen:raise ValueError(f"Duplicate labeled sample: {key}")
        seen.add(key)
        rel,senti=row["相关/不相关"].strip(),row["积极中性消极"].strip()
        if not row["分析依据"].strip() or (rel,senti) not in {("0",""),("1","-1"),("1","1")}:
            raise ValueError(f"Sample lacks valid final annotation: {key}")
        groups[(row["SourceFile"],row["SourceRow"])].append(row)
    keys=sorted(groups);random.Random(seed).shuffle(keys)
    if len(keys)<7:raise ValueError("Too few original articles to split")
    a,b=int(len(keys)*.7),int(len(keys)*.85)
    return {name:[r for k in selected for r in groups[k]] for name,selected in
            (("train",keys[:a]),("validation",keys[a:b]),("test",keys[b:]))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path,required=True,help="新的数据集输出目录，不覆盖已有正式数据集")
    parser.add_argument("--seed",type=int,default=20261101)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    csv.field_size_limit(2**31-1)
    rows=[]
    for path in args.inputs:
        with path.open(encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle)
            if reader.fieldnames!=COLUMNS:raise ValueError(f"Unexpected annotation columns: {path}")
            rows.extend(reader)
    splits=partition(rows,args.seed)
    # 先暂存全部输出；失败时不发布不完整的数据集目录。
    import tempfile
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".split_",dir=args.output.parent) as directory:
        stage=Path(directory)/"dataset";stage.mkdir()
        for task in ("relevance","sentiment"):
            folder=stage/task;folder.mkdir()
            for name,records in splits.items():
                with (folder/f"{name}.csv").open("w",encoding="utf-8-sig",newline="") as handle:
                    writer=csv.DictWriter(handle,fieldnames=COLUMNS);writer.writeheader()
                    for original in records:
                        row=dict(original)
                        if task=="sentiment" and row["相关/不相关"]!="1":continue
                        if task=="relevance":row["积极中性消极"]=""
                        writer.writerow(row)
        from .validate_annotations import validate_splits
        for task in ("relevance","sentiment"):validate_splits(stage/task,task)
        stage.rename(args.output)
    print(f"Saved article-isolated relevance/sentiment splits: {args.output}")

if __name__=="__main__":main()
