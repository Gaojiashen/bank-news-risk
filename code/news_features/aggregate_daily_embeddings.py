"""核验日度语义，并将文章等权均值对齐到交易日。

此处仅生成原始768维特征，PCA由后续训练入口使用训练期输入拟合。
周末与节假日的均值按文章数加权，而非按天数平均。
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def aggregate_calendar_embeddings(source, dates, banks, news_start, profile_hash, width=768):
    days = dates.astype("datetime64[D]")
    if not len(days) or np.any(np.diff(days).astype(int) <= 0):
        raise ValueError("Trading dates must be nonempty and increasing")
    sums = np.zeros((len(days), len(banks), width), dtype=np.float64)
    counts = np.zeros((len(days), len(banks)), dtype=np.int64)
    current = date.fromisoformat(news_start)
    if np.datetime64(current) > days[0]:
        raise ValueError("Semantic coverage begins after the first trading day")
    report = dict(calendar_days=0, article_bank_pairs=0, truncated_pairs=0, fallback_pairs=0)
    while np.datetime64(current) <= days[-1]:
        relative = f"{current:%Y/%m/%Y%m%d}.npz"
        path = Path(source) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing extracted daily embeddings: {path}")
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata_json"]))
            if meta["identity"]["day"] != str(current) or meta["identity"]["profile_sha256"] != profile_hash:
                raise ValueError(f"Mixed extraction profile or wrong date: {path}")
            if not np.array_equal(d["bank_names"].astype(str), banks):
                raise ValueError(f"Bank order differs in {path}")
            vectors, indices = d["embeddings"], d["bank_indices"]
            means, n = d["daily_bank_mean"], d["article_counts"]
            if (vectors.ndim != 2 or vectors.shape[1] != width or len(vectors) != len(indices)
                    or means.shape != (len(banks), width) or n.shape != (len(banks),)
                    or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0) or np.any(indices >= len(banks))
                    or not np.isfinite(vectors).all() or not np.isfinite(means).all()):
                raise ValueError(f"Invalid embedding values/dimensions: {path}")
            rows, files = d["source_rows"], d["source_files"].astype(str)
            source_file = relative.replace(".npz", ".csv")
            if (len(rows) != len(vectors) or len(files) != len(vectors) or np.any(rows < 2)
                    or np.any(files != source_file) or len(set(zip(rows.tolist(), indices.tolist()))) != len(vectors)):
                raise ValueError(f"Invalid or repeated article-bank identities: {path}")
            actual_counts = np.bincount(indices, minlength=len(banks))
            expected_sums = np.zeros((len(banks), width), dtype=np.float64)
            np.add.at(expected_sums, indices, vectors)
            expected_means = np.divide(expected_sums, actual_counts[:, None],
                                       out=np.zeros_like(expected_sums), where=actual_counts[:, None] > 0)
            if (not np.array_equal(n, actual_counts) or not np.array_equal(d["has_news"], n > 0)
                    or not np.allclose(means, expected_means, atol=2e-5, rtol=1e-5)
                    or meta["report"]["article_bank_pairs"] != len(vectors)
                    or meta["report"]["embedding_dim"] != width):
                raise ValueError(f"Saved counts/means differ from article vectors: {path}")
            t = int(np.searchsorted(days, np.datetime64(current)))
            sums[t] += expected_sums
            counts[t] += actual_counts
            report["article_bank_pairs"] += len(vectors)
            report["truncated_pairs"] += int(d["truncated"].sum())
            report["fallback_pairs"] += int(d["pooling_fallback"].sum())
        report["calendar_days"] += 1
        if report["calendar_days"] % 500 == 0:
            print(f"Verified semantics through {current}", flush=True)
        current += timedelta(days=1)
    features = np.divide(sums, counts[..., None], out=np.zeros_like(sums), where=counts[..., None] > 0).astype(np.float32)
    return features, counts, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--news", type=Path, default=ROOT / "data/features/daily_news_cooccurrence.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/features/daily_semantic_features.npz")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads((args.source / "extraction_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.source / "extraction_summary.json").read_text(encoding="utf-8"))
    profile_hash = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if summary["profile_sha256"] != profile_hash:
        raise ValueError("Extraction summary and manifest disagree")
    with np.load(args.news, allow_pickle=False) as d:
        dates, banks, codes = d["dates"].astype(str), d["bank_names"].astype(str), d["bank_codes"].astype(str)
        graph_meta = json.loads(str(d["metadata_json"]))
    if manifest["banks"] != banks.tolist() or manifest["bank_codes"] != codes.tolist():
        raise ValueError("Extracted semantic banks differ from news graph banks")
    start = graph_meta["news_start"]
    if manifest["start_date"] > start or manifest["end_date"] < dates[-1]:
        raise ValueError("Extraction date range does not cover daily news graphs")
    expected_days = (date.fromisoformat(manifest["end_date"]) - date.fromisoformat(manifest["start_date"])).days + 1
    files = list(args.source.glob("[12][0-9][0-9][0-9]/[01][0-9]/????????.npz"))
    if summary["days"] != expected_days or len(files) != expected_days:
        raise ValueError("Extraction is incomplete: missing daily files or summary count")
    features, counts, report = aggregate_calendar_embeddings(args.source, dates, banks, start, profile_hash)
    report.update(source=str(args.source.resolve()), extraction_profile_sha256=profile_hash,
                  extraction_summary=summary, news_start=start, news_end=str(dates[-1]),
                  alignment="calendar news in (previous trading day, current day]; equal weight per article",
                  compression="none; train-only PCA fitted by trainer", date_caveat=manifest["date_caveat"])
    print(json.dumps({"shape": list(features.shape), "counts": report}, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        print("CHECK OK; no files written.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".semantic_", dir=args.output.parent) as directory:
        stage = Path(directory) / args.output.name
        np.savez_compressed(stage, features=features, article_counts=counts, dates=dates,
                            bank_names=banks, bank_codes=codes, metadata_json=np.array(json.dumps(report, ensure_ascii=False)))
        stage.replace(args.output)
    print(f"Saved aligned daily semantics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
