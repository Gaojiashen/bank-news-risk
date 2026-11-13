"""检查最终标注数据集及文章级集合隔离。"""
import argparse
from pathlib import Path
from code.common.paths import ROOT
from code.common.text_training import read_split, DataError

def validate_splits(folder, task, include_test=True):
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    sets, result = {}, {}
    for name in names:
        rows = read_split(Path(folder) / f"{name}.csv", name)
        identities = [(r["source_file"],r["source_row"],r["sentence_index"],r["bank"]) for r in rows]
        if len(set(identities)) != len(identities):
            raise DataError(f"Duplicate sample identities in {folder}/{name}")
        if task == "sentiment" and any(r["rel"] != 1 or r["senti_raw"] not in (-1,1) for r in rows):
            raise DataError("Final sentiment splits must contain related positive/negative examples only")
        sets[name] = {(r["source_file"],r["source_row"]) for r in rows}
        result[name] = rows
    for a in names:
        for b in names:
            if a < b and sets[a] & sets[b]:
                raise DataError(f"Articles cross {a}/{b} in {folder}")
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",choices=("relevance","sentiment","both"),default="both")
    args=parser.parse_args()
    for task in ("relevance","sentiment") if args.task=="both" else (args.task,):
        rows=validate_splits(ROOT/"data/annotation"/task,task)
        print(task,{name:len(values) for name,values in rows.items()},"OK")

if __name__=="__main__":main()
