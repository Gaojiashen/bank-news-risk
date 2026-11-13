"""为每个原始文章与目标银行组合提取一个冻结BERT向量。

已有标签仅用于筛选组合，不重新判断相关性或情绪。每个组合读取一次原始标题和正文，
沿用以目标提及为中心的512词元输入规则，对提及词元末层表示取平均。
每天的NPZ保存文章向量及银行的文章等权均值。此处不拟合PCA，不训练预测模型。
--check不加载BERT权重、不写输出；不修改已有标注或原始新闻文件。
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "models/sentiment_bert_mention_balanced"
RAW_FIELDS = ["NewsID", "DeclareDate", "Title", "NewsContent", "NewsSource"]
NAME_ALIASES = {"浦发银行": "上海浦东发展银行"}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def article_bank_inputs(raw, labels, relative, banks):
    """按文章与银行组合去重标注记录，再回查原始文本。"""
    bank_index = {str(bank): i for i, bank in enumerate(banks)}
    selected = set()
    with Path(labels).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"SourceFile", "SourceRow", "BankName", "积极消极"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing annotation columns: {labels}")
        for row in reader:
            source_file = row["SourceFile"].strip().replace("\\", "/")
            if source_file != relative:
                raise ValueError(f"Annotation refers to a different daily source: {source_file!r} in {labels}")
            bank = row["BankName"].strip()
            bank = NAME_ALIASES.get(bank, bank)
            if bank not in bank_index:
                raise ValueError(f"Unknown bank {bank!r} in {labels}")
            if row["积极消极"].strip() not in ("积极", "消极"):
                raise ValueError(f"Unexpected existing sentiment in {labels}")
            source_row = int(row["SourceRow"])
            if source_row < 2:
                raise ValueError(f"Invalid original CSV record number: {source_row}")
            selected.add((source_row, bank_index[bank]))
    needed = {row for row, _ in selected}
    originals = {}
    day = date.fromisoformat(f"{Path(relative).stem[:4]}-{Path(relative).stem[4:6]}-{Path(relative).stem[6:8]}")
    with Path(raw).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if next(reader, None) != RAW_FIELDS:
            raise ValueError(f"Unexpected raw-news header: {raw}")
        for record_number, values in enumerate(reader, start=2):
            if len(values) != len(RAW_FIELDS):
                raise ValueError(f"Malformed raw-news record {record_number}: {raw}")
            if record_number in needed:
                original = dict(zip(RAW_FIELDS, values))
                if original["DeclareDate"].strip()[:10] != day.isoformat():
                    raise ValueError(f"Original date differs from daily file: {raw}, record {record_number}")
                title, body = original["Title"].strip(), original["NewsContent"]
                content = title + "\n" + body if title else body
                if not content.strip():
                    raise ValueError(f"Empty selected article: {raw}, record {record_number}")
                originals[record_number] = (content, original["NewsID"])
    if needed != set(originals):
        raise ValueError(f"Annotation references nonexistent article rows: {sorted(needed - set(originals))} in {raw}")
    return [dict(source_file=relative, source_row=row, bank=str(banks[i]), bank_index=i,
                 content=originals[row][0], news_id=originals[row][1]) for row, i in sorted(selected)]


@torch.inference_mode()
def encode_features(encoder, features, device, batch_size=16, pad_token_id=0):
    """返回目标银行提及的平均表示，位置在随机失活层和分类头之前。"""
    encoder.eval()
    outputs = []
    for offset in range(0, len(features), batch_size):
        group = features[offset:offset + batch_size]
        width = max(len(feature["input_ids"]) for feature in group)
        ids = torch.full((len(group), width), pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        segments = torch.zeros_like(ids)
        for i, feature in enumerate(group):
            length = len(feature["input_ids"])
            ids[i, :length] = torch.as_tensor(feature["input_ids"], device=device)
            attention[i, :length] = torch.as_tensor(feature["attention_mask"], device=device)
            segments[i, :length] = torch.as_tensor(feature["token_type_ids"], device=device)
        bert_segments = ((segments == 1) | (segments == 2)).long()
        hidden = encoder(input_ids=ids, attention_mask=attention, token_type_ids=bert_segments).last_hidden_state
        target = ((segments >= 2) & attention.bool()).unsqueeze(-1)
        if torch.any(target.sum(dim=1) == 0):
            raise ValueError("No visible mention or prefix fallback token in input")
        vectors = (hidden * target).sum(1) / target.sum(1)
        if not torch.isfinite(vectors).all():
            raise RuntimeError("Nonfinite BERT embeddings")
        outputs.append(vectors.float().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, encoder.config.hidden_size), dtype=np.float32)


def daily_means(vectors, bank_indices, bank_count):
    means = np.zeros((bank_count, vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(bank_count, dtype=np.int64)
    np.add.at(means, bank_indices, vectors)
    np.add.at(counts, bank_indices, 1)
    np.divide(means, counts[:, None], out=means, where=counts[:, None] > 0)
    return means, counts


def save_day(output, records, features, vectors, banks, meta):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing embeddings: {output}")
    if len(records) != len(features) or vectors.shape[0] != len(records) or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding dimensions or contents do not match selected pairs")
    indices = np.array([r["bank_index"] for r in records], dtype=np.int64)
    means, counts = daily_means(vectors, indices, len(banks))
    report = dict(article_bank_pairs=len(records), articles=len({(r["source_file"], r["source_row"]) for r in records}),
                  embedding_dim=vectors.shape[1], truncated_pairs=sum(bool(f["truncated"]) for f in features),
                  fallback_pairs=sum(bool(f["pooling_fallback"]) for f in features))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}_", dir=output.parent) as directory:
        stage = Path(directory) / output.name
        np.savez_compressed(stage, embeddings=vectors.astype(np.float32),
                            source_files=np.array([r["source_file"] for r in records], dtype=str),
                            source_rows=np.array([r["source_row"] for r in records], dtype=np.int64),
                            bank_indices=indices, bank_names=np.asarray(banks, dtype=str),
                            news_ids=np.array([r["news_id"] if "news_id" in r else "" for r in records], dtype=str),
                            truncated=np.array([f["truncated"] for f in features], dtype=bool),
                            pooling_fallback=np.array([f["pooling_fallback"] for f in features], dtype=bool),
                            original_token_counts=np.array([f["orig_tokens"] for f in features], dtype=np.int64),
                            visible_mention_tokens=np.array([f["visible_mention_tokens"] for f in features], dtype=np.int64),
                            daily_bank_mean=means, article_counts=counts, has_news=counts > 0,
                            metadata_json=np.array(json.dumps(dict(identity=meta, report=report), ensure_ascii=False)))
        # Windows下重命名不会覆盖已有目标，并发任务也遵守该约束。
        stage.rename(output)
    return report


def verify_day(output, records, banks, meta):
    with np.load(output, allow_pickle=False) as d:
        saved = json.loads(str(d["metadata_json"]))
        if saved["identity"] != meta:
            raise ValueError(f"Source, labels or extraction profile changed: {output}")
        rows = np.array([r["source_row"] for r in records], dtype=np.int64)
        indices = np.array([r["bank_index"] for r in records], dtype=np.int64)
        files = np.array([r["source_file"] for r in records], dtype=str)
        vectors = d["embeddings"]
        if (not np.array_equal(d["source_rows"], rows) or not np.array_equal(d["bank_indices"], indices)
                or not np.array_equal(d["source_files"], files) or not np.array_equal(d["bank_names"], banks)
                or vectors.shape != (len(records), saved["report"]["embedding_dim"])
                or saved["report"]["article_bank_pairs"] != len(records) or not np.isfinite(vectors).all()):
            raise ValueError(f"Embedding identities/dimensions differ from current selected pairs: {output}")
        means, counts = daily_means(vectors, indices, len(banks))
        if not (np.array_equal(counts, d["article_counts"]) and np.array_equal(counts > 0, d["has_news"])
                and np.allclose(means, d["daily_bank_mean"], atol=1e-6)):
            raise ValueError(f"Daily embedding means/counts do not match articles: {output}")
    return saved["report"]


def load_encoder(model_dir, device):
    """从本地加载微调后的编码器权重，不执行分类。"""
    from transformers import AutoTokenizer, BertConfig, BertModel
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architecture") != "MentionBertClassifier":
        raise ValueError("Expected the target-mention sentiment classifier checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer", use_fast=True, local_files_only=True)
    # 模型结构配置已保存在此前训练使用的本地缓存中。
    # 此处无需下载预训练权重或访问网络。
    architecture = BertConfig.from_pretrained(config["model_name"], local_files_only=True)
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if checkpoint["num_labels"] != 2 or checkpoint["label_names"] != ["消极", "积极"]:
        raise ValueError("Unexpected sentiment checkpoint labels")
    state = {key.removeprefix("bert."): value for key, value in checkpoint["state_dict"].items() if key.startswith("bert.")}
    encoder = BertModel(architecture)
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).to(device).eval()
    if encoder.config.hidden_size != 768:
        raise ValueError("Expected 768-dimensional BERT output")
    return tokenizer, encoder


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".json_", dir=path.parent) as directory:
        stage = Path(directory) / path.name
        stage.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/news")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/labeled_news")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--bank-reference", type=Path, default=ROOT / "data/risk/daily_delta_covar/daily_delta_covar_16_banks.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2012, 4, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2022, 7, 31))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="仅检查来源和文章银行组合，不加载BERT权重、不写输出")
    args = parser.parse_args(argv)
    if args.start_date > args.end_date or min(args.batch_size, args.threads) < 1 or not 8 <= args.max_len <= 512:
        parser.error("Invalid date range, batch/thread count, or max-len (8..512)")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    source, labels, output, model_dir = (p.resolve() for p in (args.source, args.labels, args.output, args.model_dir))
    for path in (source, labels, model_dir):
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("Embedding output must be independent of sources, labels and model directories")
    with np.load(args.bank_reference, allow_pickle=False) as reference:
        banks = reference["bank_names"].astype(str).tolist()
        codes = reference["bank_codes"].astype(str).tolist()
    if len(banks) != 16 or len(set(banks)) != 16:
        raise ValueError("Expected the 16 target banks")
    dependencies = [model_dir / "best_model.pt", model_dir / "config.json", Path(__file__),
                    ROOT / "code/common/text_features.py",
                    ROOT / "code/common/text_models.py"]
    if not (model_dir / "tokenizer/tokenizer.json").is_file():
        raise FileNotFoundError("Missing saved local tokenizer")
    dependencies.extend(sorted(p for p in (model_dir / "tokenizer").rglob("*") if p.is_file()))
    print("Checking saved model and input provenance...", flush=True)
    fingerprints = {str(p.resolve()): digest(p) for p in dependencies}
    annotation_manifest = labels / "annotation_manifest.json"
    if annotation_manifest.is_file():
        previous = json.loads(annotation_manifest.read_text(encoding="utf-8"))
        previous_hash = previous.get("fingerprints", {}).get(str((model_dir / "best_model.pt").resolve()))
        if previous_hash is not None and previous_hash != fingerprints[str((model_dir / "best_model.pt").resolve())]:
            raise ValueError("Sentiment checkpoint differs from the one used for existing labels")
    profile = dict(version=1, source=str(source), labels=str(labels), model_dir=str(model_dir),
                   start_date=str(args.start_date), end_date=str(args.end_date), max_len=args.max_len,
                   banks=banks, bank_codes=codes, fingerprints=fingerprints, embedding_dim=768,
                   input_rule="original title+body; one input per SourceFile+SourceRow+bank; target-mention-centered truncation",
                   pooling="mean final-layer visible target-mention states; prefix fallback; before dropout/classifier",
                   daily_rule="calendar-day equal article mean per bank; empty banks zero; no PCA",
                   date_caveat="Early dates through 2018-10-30 include monthly uniformly allocated news dates",
                   encoder_caveat="Reuses the prior sentiment checkpoint; not a walk-forward retraining of its text encoder")
    profile_hash = hashlib.sha256(json.dumps(profile, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "extraction_manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Extraction output exists; use --resume or another --output")
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved != profile:
            from code.common.provenance import migrated_profile_allowed
            if not migrated_profile_allowed(manifest_path, saved, profile):
                raise ValueError("Extraction profile changed; use a new output directory")
            profile = saved
            profile_hash = hashlib.sha256(json.dumps(saved, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory contains files without an extraction manifest")
    # 加载权重或发布结果前，先检查全部请求日期。
    work, complete, pairs, article_count = [], [], 0, 0
    current = args.start_date
    while current <= args.end_date:
        relative = f"{current:%Y/%m/%Y%m%d}.csv"
        raw, annotated = source / relative, labels / relative
        if not raw.is_file() or not annotated.is_file():
            raise FileNotFoundError(f"Missing raw news or labels for {current}: {raw}, {annotated}")
        meta = dict(day=str(current), source_sha256=digest(raw), labels_sha256=digest(annotated), profile_sha256=profile_hash)
        progress_path = labels / "_progress" / Path(relative).with_suffix(".json")
        if progress_path.is_file():
            original = json.loads(progress_path.read_text(encoding="utf-8"))
            if original["source_sha256"] != meta["source_sha256"] or original["output_sha256"] != meta["labels_sha256"]:
                raise ValueError(f"Raw news/labels differ from completed annotation provenance: {current}")
        records = article_bank_inputs(raw, annotated, relative, banks)
        pairs += len(records)
        article_count += len({r["source_row"] for r in records})
        target = output / Path(relative).with_suffix(".npz")
        if target.exists():
            if not args.resume:
                raise FileExistsError(f"Existing output: {target}")
            report = verify_day(target, records, banks, meta)
            if report["embedding_dim"] != 768:
                raise ValueError(f"Unexpected saved dimension: {target}")
            complete.append(report)
        else:
            work.append((current, raw, annotated, relative, target, meta))
        current += timedelta(days=1)
    print(f"Days={len(work)+len(complete)}, unique article-bank pairs={pairs}, articles={article_count}, completed={len(complete)}, pending={len(work)}", flush=True)
    if args.check:
        print("CHECK OK; no BERT weights loaded, no outputs written.")
        return
    if not manifest_path.exists():
        write_json(manifest_path, profile)
    torch.set_num_threads(args.threads)
    tokenizer, encoder = None, None
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    device = torch.device(device_name)
    feature_builder = None
    for number, (day, raw, annotated, relative, target, meta) in enumerate(work, start=1):
        if digest(raw) != meta["source_sha256"] or digest(annotated) != meta["labels_sha256"]:
            raise ValueError(f"Input changed during extraction: {day}")
        records = article_bank_inputs(raw, annotated, relative, banks)
        if records:
            if encoder is None:
                print(f"Loading saved frozen sentiment encoder on {device}...", flush=True)
                tokenizer, encoder = load_encoder(model_dir, device)
                from code.common.text_features import build_mention_features
                feature_builder = build_mention_features
            features = feature_builder(records, tokenizer, args.max_len)
            vectors = encode_features(encoder, features, device, args.batch_size, tokenizer.pad_token_id)
        else:
            features, vectors = [], np.empty((0, 768), dtype=np.float32)
        if digest(raw) != meta["source_sha256"] or digest(annotated) != meta["labels_sha256"]:
            raise ValueError(f"Input changed during extraction: {day}")
        report = save_day(target, records, features, vectors, banks, meta)
        complete.append(report)
        print(f"[{number}/{len(work)}] {day}: pairs={report['article_bank_pairs']}, truncated={report['truncated_pairs']}, fallback={report['fallback_pairs']}", flush=True)
    totals = {key: sum(r[key] for r in complete) for key in ("article_bank_pairs", "articles", "truncated_pairs", "fallback_pairs")}
    summary = dict(days=len(complete), embedding_dim=768, totals=totals, profile_sha256=profile_hash,
                   output=str(output), note="Uncompressed semantics; calendar-day bank means require article-count weighting when combining weekends")
    write_json(output / "extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
