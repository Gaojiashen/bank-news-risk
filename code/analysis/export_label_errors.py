"""从已有预测CSV中导出标签不一致的文本样本。"""
import argparse
import csv
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    with args.predictions.open(encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    pair=next(((a,b) for a,b in (("true_senti","pred_senti"),("true_rel","pred_rel")) if a in fields and b in fields),None)
    if pair is None:raise ValueError("Prediction CSV lacks true/predicted text labels")
    errors=[r for r in rows if r[pair[0]]!=r[pair[1]]]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(errors)
    print(f"Saved {len(errors)} errors: {args.output}")

if __name__=="__main__":main()
