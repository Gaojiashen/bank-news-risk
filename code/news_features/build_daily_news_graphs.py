"""构造交易日尺度的积极和消极新闻共现次数，不读取月度图。

在交易日t结束后，汇总上个交易日之后至t当天的自然日新闻，不使用未来新闻。
每篇文章对每家银行按片段情绪多数票确定方向，平票不计入对应情绪层。
由月度均分得到的日期仍是分配日期，不代表真实发布日期。源数据不改写。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from code.common.news_graph import NAME_ALIASES, label

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "data/risk/daily_delta_covar/daily_delta_covar_16_banks.npz"
OUT = ROOT / "data/features/daily_news_cooccurrence.npz"


def build_graphs(source: Path, dates: np.ndarray, banks: np.ndarray, news_start: str | None = None):
    dates = dates.astype("datetime64[D]")
    if not len(dates) or np.any(np.diff(dates).astype(int) <= 0):
        raise ValueError("Trading dates must be nonempty and strictly increasing")
    index = {str(name): i for i, name in enumerate(banks)}
    if len(index) != len(banks):
        raise ValueError("Bank names must be unique")
    positive = np.zeros((len(dates), len(banks), len(banks)), dtype=np.float32)
    negative = np.zeros_like(positive)
    current = date.fromisoformat(news_start or str(dates[0]))
    last = date.fromisoformat(str(dates[-1]))
    if current > date.fromisoformat(str(dates[0])):
        raise ValueError("News coverage must begin no later than first trading date")
    report = {"calendar_files": 0, "annotated_snippets": 0, "articles": 0,
              "bank_article_ties": 0, "allocated_date_files": 0}
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    while current <= last:
        path = source / f"{current:%Y/%m/%Y%m%d}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing annotated daily file (not treated as zero news): {path}")
        t = int(np.searchsorted(dates, np.datetime64(current)))
        articles = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"SourceFile", "SourceRow", "BankName", "积极消极"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing columns in {path}: {required - set(reader.fieldnames or [])}")
            for row in reader:
                bank = row["BankName"].strip()
                bank = NAME_ALIASES.get(bank, bank)
                if bank not in index:
                    raise ValueError(f"Unknown bank {bank!r} in {path}")
                polarity = label(row["积极消极"])
                if polarity is None:
                    raise ValueError(f"Unknown sentiment in {path}: {row['积极消极']!r}")
                key = (row["SourceFile"].strip(), row["SourceRow"].strip())
                if not all(key):
                    raise ValueError(f"Missing article identity in {path}")
                articles[key][index[bank]][0 if polarity == 1 else 1] += 1
                report["annotated_snippets"] += 1
        for counts in articles.values():
            report["bank_article_ties"] += sum(p == n for p, n in counts.values())
            for channel, adjacency in ((0, positive), (1, negative)):
                selected = sorted(i for i, votes in counts.items() if votes[channel] > votes[1 - channel])
                for offset, i in enumerate(selected):
                    for j in selected[offset + 1:]:
                        adjacency[t, i, j] += 1
                        adjacency[t, j, i] += 1
        report["articles"] += len(articles)
        report["calendar_files"] += 1
        report["allocated_date_files"] += current <= date(2018, 10, 30)
        current += timedelta(days=1)
    report.update({"axes": "trading_date, target_bank, source_bank", "news_start": news_start or str(dates[0]),
                   "news_end": str(dates[-1]), "alignment": "(previous trading date, current trading date]; end-of-day forecast",
                   "article_rule": "SourceFile+SourceRow; majority per bank, ties excluded; same polarity pair counts once",
                   "date_caveat": "Early news dates through 2018-10-30 include monthly uniform allocations; not real daily timing"})
    return positive, negative, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/labeled_news")
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--start", default="2012-04-01", help="新闻起始自然日，包含当天")
    parser.add_argument("--end", default=None, help="最后一个交易日，包含当天")
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    with np.load(args.target, allow_pickle=False) as target:
        dates = target["dates"].astype(str)
        select = dates >= start.isoformat()
        if args.end:
            select &= dates <= date.fromisoformat(args.end).isoformat()
        dates = dates[select]
        banks, codes = target["bank_names"].astype(str), target["bank_codes"].astype(str)
    positive, negative, report = build_graphs(args.source, dates, banks, start.isoformat())
    report.update({"source": str(args.source.resolve()), "risk_source": str(args.target.resolve())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, positive=positive, negative=negative, dates=dates,
                        bank_names=banks, bank_codes=codes, metadata_json=np.array(json.dumps(report, ensure_ascii=False)))
    args.output.with_suffix(".metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved daily graphs {positive.shape}: {args.output}", flush=True)
    print(f"Assigned-date files (includes monthly allocations): {report['allocated_date_files']}", flush=True)


if __name__ == "__main__":
    main()
