r"""用此前的 v1 相关性 + v5 温和平衡采样情绪模型标注日度新闻。

手动执行（项目根目录）：
    .\.venv\Scripts\python.exe -m code.news_features.annotate_daily_news --resume

输入：data/raw/news/YYYY/MM/YYYYMMDD.csv
输出：data/labeled_news/YYYY/MM/YYYYMMDD.csv
不会改写原始新闻，不训练模型，不修改已有人工标注或训练集。

沿用 公共文本推理模块 的模型、输入构造、银行识别、分句及
上下文规则。标题作为第一句（如有），正文随后；每个提及句对每家目标
银行生成一个“前一句+本句+后一句”样本。SentenceIndex 从 1 开始，
有标题时标题也参与句子编号。NewsContent 保存模型实际看到的片段；
全文可凭 SourceFile（相对于输入根目录）及 SourceRow（CSV 记录号，
表头计为第 1 条）回查。原文换行不会改变 SourceRow 的计数。

与此前批量标注一致：先判相关性，仅对相关样本输出积极/消极标签；
不相关样本不写入最终 CSV，数量记入 _progress 中的日度统计。
一篇新闻可展开多条目标银行样本，不能把输出行数当作新闻篇数。
所有标签均为模型预测；不生成或伪造人工分析依据，不还原人工标签。
空输入日、没有有效银行提及或没有相关样本的日期，输出只有表头。
原有均分日期直接沿用，不重新分配。

--resume 按已完成的日期继续，并核对源文件、输出文件和模型指纹。
每日 CSV 完整写好后才发布。若中断恰好发生在发布 CSV 与统计之间，
再次执行会提示具体文件；检查后移走该日输出 CSV 再运行即可。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 仅导入定义；原脚本的 main 有入口保护，不会触发训练或旧数据集生成。
from code.common import text_inference as previous

import torch


SOURCE_DIR = ROOT / "data" / "raw" / "news"
OUTPUT_DIR = ROOT / "data" / "labeled_news"
INPUT_FIELDS = ["NewsID", "DeclareDate", "Title", "NewsContent", "NewsSource"]
OUTPUT_FIELDS = ["DeclareDate", "NewsID", "Title", "NewsSource", "NewsContent",
                 "BankName", "SourceFile", "SourceRow", "SentenceIndex", "积极消极"]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def daily_files(source: Path, start: date, end: date) -> list[tuple[date, Path]]:
    if start > end:
        raise ValueError("起始日期不能晚于结束日期")
    result = []
    current = start
    while current <= end:
        path = source / f"{current.year:04d}" / f"{current.month:02d}" / f"{current:%Y%m%d}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入日度新闻：{path}")
        result.append((current, path))
        if current == end:
            break
        current += timedelta(days=1)
    return result


def annotate_day(day, source, relative, output, aliases, relevance, sentiment,
                 device, batch_size):
    stats = Counter(input_news=0, no_bank_samples=0, candidate_samples=0,
                    irrelevant_removed=0, positive=0, negative=0, output_samples=0)
    pending = []

    def flush(writer):
        if not pending:
            return
        decisions = previous.classify(pending, *relevance, "relevance", device)
        selected = [row for row, decision in zip(pending, decisions) if decision == 1]
        stats["irrelevant_removed"] += len(pending) - len(selected)
        if selected:
            emotions = previous.classify(selected, *sentiment, "sentiment", device)
            for row, decision in zip(selected, emotions):
                row["积极消极"] = "消极" if decision == 0 else "积极"
                writer.writerow(row)
                stats["negative" if decision == 0 else "positive"] += 1
                stats["output_samples"] += 1
        pending.clear()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{day:%Y%m%d}_", dir=output.parent) as temp:
        temporary = Path(temp) / output.name
        with source.open(encoding="utf-8-sig", newline="") as handle, \
                temporary.open("x", encoding="utf-8-sig", newline="") as target:
            reader = csv.reader(handle, strict=True)
            if next(reader, None) != INPUT_FIELDS:
                raise ValueError(f"输入字段不符合预期：{source}")
            writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for record_number, values in enumerate(reader, start=2):
                if len(values) != len(INPUT_FIELDS):
                    raise ValueError(f"输入记录列数异常：{source} 第 {record_number} 条")
                original = dict(zip(INPUT_FIELDS, values))
                if original["DeclareDate"].strip()[:10] != day.isoformat():
                    raise ValueError(f"输入日期与文件名不一致：{source} 第 {record_number} 条")
                stats["input_news"] += 1
                title = original["Title"].strip()
                body = original["NewsContent"]
                article = title + "\n" + body if title else body
                sentences = previous.split_sentences(article)
                candidates_before = stats["candidate_samples"]
                for index, sentence in enumerate(sentences, start=1):
                    for bank in previous.mentioned_banks(sentence, aliases):
                        context = "".join(sentences[max(0, index - 2):index + 1])
                        row = {name: original[name] for name in ("DeclareDate", "NewsID", "Title", "NewsSource")}
                        row.update(NewsContent=context, BankName=bank, SourceFile=relative,
                                   SourceRow=record_number, SentenceIndex=index)
                        pending.append(row)
                        stats["candidate_samples"] += 1
                        if len(pending) >= batch_size:
                            flush(writer)
                if candidates_before == stats["candidate_samples"]:
                    stats["no_bank_samples"] += 1
            flush(writer)
        if stats["candidate_samples"] != stats["irrelevant_removed"] + stats["output_samples"]:
            raise RuntimeError(f"样本统计不一致：{source}")
        if output.exists():
            raise FileExistsError(f"输出已存在，拒绝覆盖：{output}")
        # Windows 下 rename 不覆盖已存在的目标。
        temporary.rename(output)
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="用此前的模型批量标注日度银行新闻")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2012, 4, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2022, 7, 31))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true", help="核对指纹后跳过已完成日期")
    parser.add_argument("--check", action="store_true", help="检查已完成文件，不推理、不写入")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须为正整数")
    csv.field_size_limit(sys.maxsize)
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输入与输出目录必须独立，不能相同或互相包含")
    files = daily_files(source, args.start_date, args.end_date)
    alias_path = ROOT / "data" / "raw" / "bank_aliases.xlsx"
    fingerprint_paths = [
        previous.RELEVANCE_DIR / "best_model.pt",
        previous.SENTIMENT_DIR / "best_model.pt",
        alias_path, Path(__file__), Path(previous.__file__),
        ROOT / "code/common/banks.py",
        ROOT / "code/common/text_models.py",
        ROOT / "code/common/text_features.py",
    ]
    print("核对模型与代码文件……", flush=True)
    fingerprints = {}
    for path in fingerprint_paths:
        if not path.is_file():
            raise FileNotFoundError(f"缺少模型或依赖文件：{path}")
        fingerprints[str(path.resolve())] = digest(path)
    for folder in (previous.RELEVANCE_DIR / "tokenizer", previous.SENTIMENT_DIR / "tokenizer"):
        if not folder.is_dir():
            raise FileNotFoundError(f"缺少分词器目录：{folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                fingerprints[str(path.resolve())] = digest(path)
    settings = {"source": str(source), "start": str(args.start_date), "end": str(args.end_date),
                "fields": OUTPUT_FIELDS, "fingerprints": fingerprints}
    manifest = output / "annotation_manifest.json"
    if manifest.exists():
        if not args.resume:
            raise FileExistsError("已有标注任务，请加 --resume 继续，或指定新的 --output")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != settings:
            from code.common.provenance import migrated_profile_allowed
            if not migrated_profile_allowed(manifest, saved, settings):
                raise ValueError("Model, input range or unreviewed preprocessing changed; use a new output")
            settings = saved
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"输出目录有其他内容，拒绝混入：{output}")
        if not args.check:
            write_json(manifest, settings)

    work, totals = [], Counter()
    for day, path in files:
        relative = path.relative_to(source)
        target = output / relative
        report_path = output / "_progress" / relative.with_suffix(".json")
        if target.exists():
            if not args.resume or not report_path.is_file():
                raise FileExistsError(f"输出缺少可验证进度；检查后移走该日输出再继续：{target}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report["source_sha256"] != digest(path) or report["output_sha256"] != digest(target):
                raise ValueError(f"源文件或输出内容已变化，停止续跑：{path}")
            totals.update(report["stats"])
        else:
            work.append((day, path, relative, target, report_path))
    print(f"输入 {len(files)} 天，已完成 {len(files) - len(work)} 天，待处理 {len(work)} 天", flush=True)
    if args.check:
        print("CHECK OK; no inference or writes")
        return
    if work:
        device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device_name == "auto":
            device_name = "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请使用 --device cpu 或检查 GPU 环境")
        device = torch.device(device_name)
        print(f"设备：{device}；正在加载此前的相关性和情绪模型……", flush=True)
        relevance = previous.load_model(previous.RELEVANCE_DIR, "relevance", device)
        sentiment = previous.load_model(previous.SENTIMENT_DIR, "sentiment", device)
        aliases = previous.load_aliases()
        if len(aliases) != 16:
            raise ValueError("银行别名表未识别到 16 家银行")
        for number, (day, path, relative, target, report_path) in enumerate(work, start=1):
            print(f"[{number}/{len(work)}] 正在标注 {day}……", flush=True)
            before = digest(path)
            stats = annotate_day(day, path, relative.as_posix(), target, aliases,
                                 relevance, sentiment, device, args.batch_size)
            if digest(path) != before:
                raise RuntimeError(f"标注过程中源文件发生变化，请检查并移走该日输出：{target}")
            write_json(report_path, {"source_sha256": before, "output_sha256": digest(target), "stats": stats})
            totals.update(stats)
            print(f"  原文 {stats['input_news']} 篇；样本 {stats['candidate_samples']}；"
                  f"排除不相关 {stats['irrelevant_removed']}；"
                  f"积极 {stats['positive']}、消极 {stats['negative']}", flush=True)
    write_json(output / "annotation_summary.json", {"days": len(files), "totals": dict(totals)})
    print(f"完成：{output}\n汇总：{dict(totals)}", flush=True)


if __name__ == "__main__":
    main()
