"""共享文本训练流程，仅使用训练与验证数据，不包含已淘汰实验。"""
import csv
import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from .paths import ROOT as PROJECT_ROOT
from .text_features import DataError, aliases_for, build_features, build_mention_features
from .text_models import BertClassifier, MentionBertClassifier
SEED=42
MODEL_NAME="google-bert/bert-base-chinese"
MAX_LEN=512
PAD_TOKEN_ID=0
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=3
WEIGHT_DECAY=.01
WARMUP_RATIO=.1
MAX_GRAD_NORM=1.
DROPOUT=.1
DEVICE="auto"
NUM_WORKERS=0
SELECTION_METRIC="macro_f1"
SENTI_TO_INDEX={-1:0, 1:1}
REL_LABELS={0:"不相关",1:"相关"}
COLUMNS=["NewsContent","BankName","SourceFile","SourceRow","SentenceIndex","分析依据","相关/不相关","积极中性消极"]

def read_split(path, expected):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise DataError(f"Unexpected annotation columns: {path}")
        result=[]
        for number, row in enumerate(reader,2):
            rel=row["相关/不相关"].strip()
            senti=row["积极中性消极"].strip()
            if rel not in ("0","1") or senti not in ("","-1","1") or (rel=="0" and senti):
                raise DataError(f"Invalid final binary label: {path}:{number}")
            if not row["NewsContent"].strip() or not row["BankName"].strip():
                raise DataError(f"Empty article or bank: {path}:{number}")
            raw=int(senti) if senti else None
            result.append(dict(split=expected,row_no=number,source_file=row["SourceFile"],source_row=row["SourceRow"],
                               sentence_index=row["SentenceIndex"],bank=row["BankName"].strip(),content=row["NewsContent"],
                               rel=int(rel),senti_raw=raw,senti_idx=SENTI_TO_INDEX.get(raw)))
    if not result:
        raise DataError(f"Empty annotation split: {path}")
    return result

def describe_split(rows):
    return f"rows={len(rows)}, relevance={dict(Counter(r['rel'] for r in rows))}, sentiment={dict(Counter(r['senti_raw'] for r in rows if r['senti_raw'] is not None))}"
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dump_json(path: Path, obj) -> None: # 把obj以json形式保存在path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def token_length_report(name: str, feats: list, max_len: int) -> dict:
    lens = np.array([f["orig_tokens"] for f in feats])
    over = int((lens > max_len).sum())
    truncated = [f for f in feats if f["truncated"]]
    no_mention = [f for f in feats if not f["mention_found"]]
    fallback = [f for f in truncated if f["truncate_mode"] == "head_tail_fallback"]
    report = {
        "split": name,
        "samples": len(feats),
        "tokens_min": int(lens.min()),
        "tokens_mean": float(lens.mean()),
        "tokens_median": float(np.median(lens)),
        "tokens_p95": float(np.percentile(lens, 95)),
        "tokens_max": int(lens.max()),
        "over_max_len": over,
        "over_max_len_ratio": float(over / len(feats)) if feats else 0.0,
        "truncated": len(truncated),
        "truncated_mention_centered": len(truncated) - len(fallback),
        "truncated_head_tail_fallback": len(fallback),
        "mention_not_found": len(no_mention),
    }
    print(f"  [{name}] token 长度 min/mean/median/p95/max = "
          f"{report['tokens_min']}/{report['tokens_mean']:.1f}/{report['tokens_median']:.0f}/"
          f"{report['tokens_p95']:.0f}/{report['tokens_max']}；"
          f"超过 {max_len} 的 {over} 条（{report['over_max_len_ratio']:.1%}）；"
          f"实际截断 {len(truncated)} 条（其中按提及居中 {report['truncated_mention_centered']}、"
          f"无法定位改首尾拼接 {len(fallback)}）；全文找不到银行提及 {len(no_mention)} 条")
    return report


class FeatureDataset(Dataset):
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        f = self.feats[idx]
        return {
            "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(f["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(f["token_type_ids"], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_batch(batch):
    """批内动态 padding（各样本长度不同，不能直接 stack）。"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), PAD_TOKEN_ID, dtype=torch.long)
    # torch.full创建一个指定形状，所有位置填入同一个值的张量
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    token_type_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].size(0)
        input_ids[i, :n] = item["input_ids"] # 把input_ids复制到第i行的前n个位置
        attention_mask[i, :n] = item["attention_mask"]
        token_type_ids[i, :n] = item["token_type_ids"]
        labels[i] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "token_type_ids": token_type_ids, "labels": labels}


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad() # 不记录梯度
def predict(model, loader, device):
    model.eval() # 切换到评估模式：关闭dropout、不更新参数
    logits_all, labels_all = [], []
    for batch in loader:
        labels = batch["labels"]
        batch = batch_to_device(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
        # 得到每条样本的类别分数
        logits_all.append(logits.detach().cpu())
        # detach得到一个数值相同，但与当前自动求导计算图断开的张量；
        # 之后对该张量做计算不会沿模型计算过程反向传播梯度
        labels_all.append(labels)
    logits = torch.cat(logits_all, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    return logits.numpy(), probs, torch.cat(labels_all).numpy()


def classification_report(y_true, y_pred, label_names, positive_index=None,
                          positive_name=None, extra=None):
    """逐类 P/R/F1 + macro-F1 + 混淆矩阵。"""
    labels = list(range(len(label_names)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    rep = {
        "per_class": {
            label_names[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in labels
        },
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "confusion_matrix": cm,
        "confusion_labels": [label_names[i] for i in labels],
    }
    if positive_index is not None:
        rep[f"recall_{positive_name}"] = float(r[positive_index])
    if extra:
        rep.update(extra)
    return rep


def print_report(title, rep):
    print(f"\n--- {title} ---")
    print(f"  {'类别':<6}{'precision':>10}{'recall':>9}{'f1':>9}{'support':>9}")
    for name, m in rep["per_class"].items():
        print(f"  {name:<6}{m['precision']:>10.4f}{m['recall']:>9.4f}{m['f1']:>9.4f}{m['support']:>9}")
    print(f"  macro-F1 = {rep['macro_f1']:.4f}   macro-P = {rep['macro_precision']:.4f}   "
          f"macro-R = {rep['macro_recall']:.4f}")
    for k, v in rep.items():
        if k.startswith("recall_"):
            print(f"  {k} = {v:.4f}")
    print("  混淆矩阵（行=真实，列=预测）：")
    print("      " + "  ".join(f"{n:>6}" for n in rep["confusion_labels"]))
    for name, row in zip(rep["confusion_labels"], rep["confusion_matrix"]):
        print(f"  {name:<4}" + "  ".join(f"{v:>6}" for v in row))


def train_task(task_name, train_feats, train_labels, val_feats, val_labels,
               num_labels, device, out_dir, label_names,
               positive_index=None, positive_name=None):
    """独立的训练循环：自己一份 BERT + 分类层，自己优化器、自己最佳检查点。"""
    print(f"\n{'=' * 78}\n训练 {task_name}：训练 {len(train_feats)} 条，验证 {len(val_feats)} 条，"
          f"{num_labels} 类\n{'=' * 78}")
    seed_everything(SEED)                      # 每个任务重新设种子，保证可复现

    train_loader = DataLoader(FeatureDataset(train_feats, train_labels),
                              batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, drop_last=False,
                              collate_fn=collate_batch)
    # shuffle每次遍历train_loader时打乱样本顺序再分批
    # drop_last=False保留最后一个不足batch_size的批次
    val_loader = DataLoader(FeatureDataset(val_feats, val_labels),
                            batch_size=EVAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_batch)

    model = BertClassifier(num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) * NUM_EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    loss_fn = nn.CrossEntropyLoss()

    task_dir = out_dir / ("relevance" if task_name == "相关性" else "sentiment")
    task_dir.mkdir(parents=True, exist_ok=True)
    best_path = task_dir / "best_model.pt"
    best_score, best_epoch, history = -1.0, -1, []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train() # 训练模式
        running, seen = 0.0, 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True) # 清除梯度
            logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            loss = loss_fn(logits, labels) # 前向计算 # 直接传 logits，不先 softmax
            loss.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM) # 梯度裁剪
            optimizer.step() # 更新参数
            scheduler.step() # 更新学习率
            running += loss.item() * labels.size(0) # 总计损失
            seen += labels.size(0) # 批大小

        _, probs, y_val = predict(model, val_loader, device)
        y_pred = probs.argmax(axis=1)
        rep = classification_report(y_val, y_pred, label_names,
                                    positive_index=positive_index,
                                    positive_name=positive_name)
        score = rep[SELECTION_METRIC]
        train_loss = running / max(1, seen)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_macro_f1": score,
                        "val_acc": float((y_pred == y_val).mean())})
        print(f"  epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_macro_f1={score:.4f}  val_acc={(y_pred == y_val).mean():.4f}")

        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"state_dict": model.state_dict(),
                        "num_labels": num_labels,
                        "model_name": MODEL_NAME,
                        "max_len": MAX_LEN,
                        "label_names": label_names,
                        "epoch": epoch,
                        "val_macro_f1": score}, best_path)
            print(f"    新的最佳检查点：epoch {epoch}，macro-F1={score:.4f}")

    # 用最佳检查点做最终验证与预测
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    logits, probs, y_val = predict(model, val_loader, device)
    y_pred = probs.argmax(axis=1)
    metrics = classification_report(y_val, y_pred, label_names,
                                    positive_index=positive_index,
                                    positive_name=positive_name)
    metrics.update({
        "task": task_name,
        "num_labels": num_labels,
        "label_names": label_names,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "history": history,
        "train_size": len(train_feats),
        "val_size": len(val_feats),
        "selection_metric": SELECTION_METRIC,
    })
    dump_json(task_dir / "metrics.json", metrics)
    dump_json(task_dir / "labels.json", {
        "label_names": label_names,
        "index_to_label": {i: n for i, n in enumerate(label_names)},
    })
    print(f"  最佳 epoch={best_epoch}，macro-F1={best_score:.4f}")
    return {"model": model, "metrics": metrics, "probs": probs, "pred": y_pred,
            "true": y_val}


def base_pred_rows(feats):
    rows = []
    for f in feats:
        m = f["meta"]
        rows.append({
            "SourceFile": m["source_file"],
            "SourceRow": m["source_row"],
            "SentenceIndex": m["sentence_index"],
            "BankName": m["bank"],
            "truncated": int(f["truncated"]),
            "truncate_mode": f["truncate_mode"],
            "mention_found": int(f["mention_found"]),
            "orig_tokens": f["orig_tokens"],
        })
    return rows


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
