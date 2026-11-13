"""仅训练相关性模型，不覆盖已选定并保存的正式模型。"""
import argparse
import json
from transformers import AutoTokenizer
from code.common import text_training as base
from code.common.config import load_config, project_path
from code.common.paths import ROOT
from code.annotation.validate_annotations import validate_splits

def run(config, check=False):
    for key in ("seed","max_len","train_batch_size","eval_batch_size","learning_rate","num_epochs","weight_decay","warmup_ratio","max_grad_norm","device"):
        if key in config:setattr(base,key.upper(),config[key])
    folder=project_path(config["dataset"])
    data=validate_splits(folder,"relevance",include_test=False)
    print({name:len(rows) for name,rows in data.items()})
    if check:
        print("CHECK OK; no test data read, no model loaded, no training")
        return
    out=project_path(config["output"])
    if out.exists() and any(out.iterdir()):raise FileExistsError(f"Use a new experiment directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    base.seed_everything(base.SEED)
    tokenizer=AutoTokenizer.from_pretrained(base.MODEL_NAME,use_fast=True)
    train=base.build_features(data["train"],tokenizer,base.MAX_LEN)
    validation=base.build_features(data["validation"],tokenizer,base.MAX_LEN)
    result=base.train_task("相关性",train,[r["rel"] for r in data["train"]],validation,
                           [r["rel"] for r in data["validation"]],2,base.resolve_device(base.DEVICE),out,["不相关","相关"],1,"相关")
    tokenizer.save_pretrained(out/"relevance/tokenizer")
    base.dump_json(out/"config.json",config)
    base.dump_json(out/"relevance/config.json",dict(model_name=base.MODEL_NAME,max_len=base.MAX_LEN,num_labels=2,architecture="BertClassifier"))
    rows=base.base_pred_rows(validation)
    for row,truth,pred in zip(rows,result["true"],result["pred"]):row.update(true_rel=int(truth),pred_rel=int(pred))
    base.write_csv(out/"predictions.csv",rows)
    from code.analysis.plot_results import plot_text_metrics
    plot_text_metrics(result["metrics"],out/"figures")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/relevance.yaml"))
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args();run(load_config(args.config),args.check)

if __name__=="__main__":main()
