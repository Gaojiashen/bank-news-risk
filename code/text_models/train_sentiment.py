"""统一二分类情绪训练入口，支持CLS或提及池化、普通或温和平衡采样。"""
import argparse
import csv
import json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from code.common import text_training as base
from code.common.text_features import build_mention_features
from code.common.text_models import MentionBertClassifier
def mild_sampling_weights(labels):
    """计算逐样本抽样概率，使类别抽中份额正比于该类样本数的平方根。"""
    from collections import Counter
    counts = Counter(labels)
    if any(counts[c] == 0 for c in range(2)):
        raise base.DataError("训练集缺少某个情绪类别")
    raw = [counts[label] ** -0.5 for label in labels]
    total = sum(raw)
    return [weight / total for weight in raw]


def train_task_with_loss_history(task_name, train_feats, train_labels,
                                 val_feats, val_labels, num_labels, device,
                                 out_dir, label_names, positive_index=None,
                                 positive_name=None, pooling="mention", sampling="mild"):
    """执行既定训练流程，记录每一步及每一轮的损失。"""
    base.seed_everything(base.SEED)
    sample_weights = mild_sampling_weights(train_labels) if sampling == "mild" else None
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels),
                                     replacement=True,
                                     generator=torch.Generator().manual_seed(base.SEED)) if sample_weights is not None else None
    train_loader = DataLoader(base.FeatureDataset(train_feats, train_labels),
                              batch_size=base.TRAIN_BATCH_SIZE, sampler=sampler, shuffle=sampler is None,
                              num_workers=base.NUM_WORKERS, drop_last=False,
                              collate_fn=base.collate_batch)
    val_loader = DataLoader(base.FeatureDataset(val_feats, val_labels),
                            batch_size=base.EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=base.NUM_WORKERS,
                            collate_fn=base.collate_batch)
    model = (MentionBertClassifier(num_labels) if pooling == "mention" else base.BertClassifier(num_labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base.LEARNING_RATE,
                                  weight_decay=base.WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * base.NUM_EPOCHS)
    scheduler = base.get_linear_schedule_with_warmup(
        optimizer, int(total_steps * base.WARMUP_RATIO), total_steps)
    loss_fn = nn.CrossEntropyLoss()
    task_dir = out_dir / "sentiment"
    task_dir.mkdir(parents=True, exist_ok=True)
    best_path = task_dir / "best_model.pt"
    best_score, best_epoch, history = -1.0, -1, []
    step = 0
    with (out_dir / "loss_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "epoch", "batch", "global_step", "batch_size", "batch_loss",
            "running_epoch_loss", "learning_rate"])
        writer.writeheader()
        for epoch in range(1, base.NUM_EPOCHS + 1):
            model.train()
            running, seen = 0.0, 0
            for batch_index, batch in enumerate(train_loader, 1):
                labels = batch["labels"].to(device)
                batch = base.batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch["input_ids"], batch["attention_mask"],
                               batch["token_type_ids"])
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), base.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                step += 1
                batch_size = labels.size(0)
                loss_value = loss.item()
                running += loss_value * batch_size
                seen += batch_size
                writer.writerow({"epoch": epoch, "batch": batch_index,
                                 "global_step": step, "batch_size": batch_size,
                                 "batch_loss": loss_value,
                                 "running_epoch_loss": running / seen,
                                 "learning_rate": scheduler.get_last_lr()[0]})
            handle.flush()
            logits, probs, y_val = base.predict(model, val_loader, device)
            y_pred = probs.argmax(axis=1)
            rep = base.classification_report(y_val, y_pred, label_names,
                positive_index=positive_index, positive_name=positive_name)
            score = rep[base.SELECTION_METRIC]
            val_loss = loss_fn(torch.from_numpy(logits),
                               torch.from_numpy(y_val).long()).item()
            history.append({"epoch": epoch, "train_loss": running / seen,
                            "val_loss": val_loss, "val_macro_f1": score,
                            "val_acc": float((y_pred == y_val).mean())})
            print(f"  epoch {epoch}/{base.NUM_EPOCHS}  train_loss={running / seen:.4f}  "
                  f"val_loss={val_loss:.4f}  val_macro_f1={score:.4f}  "
                  f"val_acc={(y_pred == y_val).mean():.4f}")
            if score > best_score:
                best_score, best_epoch = score, epoch
                torch.save({"state_dict": model.state_dict(),
                            "num_labels": num_labels, "model_name": base.MODEL_NAME,
                            "max_len": base.MAX_LEN, "label_names": label_names,
                            "epoch": epoch, "val_macro_f1": score}, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, probs, y_val = base.predict(model, val_loader, device)
    y_pred = probs.argmax(axis=1)
    metrics = base.classification_report(y_val, y_pred, label_names,
        positive_index=positive_index, positive_name=positive_name)
    metrics.update({"task": task_name, "num_labels": num_labels,
                    "label_names": label_names, "best_epoch": best_epoch,
                    "best_val_macro_f1": best_score, "history": history,
                    "train_size": len(train_feats), "val_size": len(val_feats),
                    "selection_metric": base.SELECTION_METRIC})
    base.dump_json(task_dir / "metrics.json", metrics)
    base.dump_json(task_dir / "labels.json", {
        "label_names": label_names,
        "index_to_label": {i: n for i, n in enumerate(label_names)}})
    return {"model": model, "metrics": metrics, "probs": probs,
            "pred": y_pred, "true": y_val}


def main():
    from code.common.paths import ROOT
    from code.common.config import load_config, project_path
    from code.annotation.validate_annotations import validate_splits
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"configs/sentiment_mention_balanced.yaml"))
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args();config=load_config(args.config)
    pooling,sampling=config["pooling"],config["sampling"]
    if pooling not in ("cls","mention") or sampling not in ("none","mild"):
        raise ValueError("Unknown pooling or sampling configuration")
    for key in ("seed","max_len","train_batch_size","eval_batch_size","learning_rate","num_epochs","weight_decay","warmup_ratio","max_grad_norm","device"):
        if key in config:setattr(base,key.upper(),config[key])
    data=validate_splits(project_path(config["dataset"]),"sentiment",include_test=False)
    print({name:len(rows) for name,rows in data.items()})
    if args.check:
        print("CHECK OK; no test data read, no model loaded, no training");return
    out=project_path(config["output"])
    if out.exists() and any(out.iterdir()):raise FileExistsError(f"Use a new experiment directory: {out}")
    out.mkdir(parents=True,exist_ok=True)
    base.seed_everything(base.SEED)
    tokenizer=AutoTokenizer.from_pretrained(base.MODEL_NAME,use_fast=True)
    builder=build_mention_features if pooling=="mention" else base.build_features
    train=builder(data["train"],tokenizer,base.MAX_LEN)
    validation=builder(data["validation"],tokenizer,base.MAX_LEN)
    result=train_task_with_loss_history("情绪",train,[r["senti_idx"] for r in data["train"]],
        validation,[r["senti_idx"] for r in data["validation"]],2,base.resolve_device(base.DEVICE),out,
        ["消极","积极"],0,"消极",pooling=pooling,sampling=sampling)
    tokenizer.save_pretrained(out/"sentiment/tokenizer")
    base.dump_json(out/"config.json",config)
    base.dump_json(out/"sentiment/config.json",dict(model_name=base.MODEL_NAME,max_len=base.MAX_LEN,num_labels=2,
        architecture="MentionBertClassifier" if pooling=="mention" else "BertClassifier"))
    rows=base.base_pred_rows(validation)
    for row,truth,pred in zip(rows,result["true"],result["pred"]):
        row.update(true_senti=-1 if int(truth)==0 else 1,pred_senti=-1 if int(pred)==0 else 1)
    base.write_csv(out/"predictions.csv",rows)
    from code.analysis.plot_results import plot_text_metrics
    plot_text_metrics(result["metrics"],out/"figures")

if __name__=="__main__":main()
