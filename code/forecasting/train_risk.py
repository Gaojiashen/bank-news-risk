"""仅用风险和新闻训练日度基准、注意力及扩散模型。

默认使用60个交易日历史，预测下一交易日的Delta CoVaR点值。
先生成日度新闻图；--check仅检查输入，不启动训练。
小批次始终包含全部银行，避免将全部窗口一次性放入GPU。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .risk_models import DailyBenchmarkRiskGRU, DailyNewsNetworkRiskModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("benchmark", "attention_bias", "diffusion")
SEMANTIC_MODEL_NAMES = ("benchmark", "benchmark_semantic", "attention_bias", "diffusion")


def load_panel(target: Path, news: Path, start=None, end=None):
    with np.load(target, allow_pickle=False) as d:
        all_dates = d["dates"].astype(str)
        risk = d["delta_covar"].astype(np.float32)
        banks, codes = d["bank_names"].astype(str), d["bank_codes"].astype(str)
    if risk.shape != (len(all_dates), 16) or len(set(banks)) != 16 or len(set(codes)) != 16:
        raise ValueError("Expected daily targets [dates,16] with unique banks and codes")
    if np.any(all_dates[1:] <= all_dates[:-1]):
        raise ValueError("Risk dates must be strictly increasing")
    with np.load(news, allow_pickle=False) as d:
        dates = d["dates"].astype(str)
        pos, neg = d["positive"].astype(np.float32), d["negative"].astype(np.float32)
        if not np.array_equal(banks, d["bank_names"].astype(str)) or not np.array_equal(codes, d["bank_codes"].astype(str)):
            raise ValueError("Daily news and risk bank order differs")
        metadata = json.loads(str(d["metadata_json"])) if "metadata_json" in d else {}
    if not len(dates) or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("News dates must be nonempty and strictly increasing")
    idx = np.searchsorted(all_dates, dates)
    if np.any(idx >= len(all_dates)) or not np.array_equal(all_dates[idx], dates):
        raise ValueError("News contains dates absent from risk targets")
    if np.any(np.diff(idx) != 1):
        raise ValueError("News is missing trading dates within its coverage")
    for matrix in (pos, neg):
        if matrix.shape != (len(dates), 16, 16) or not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("Daily graphs must be finite nonnegative [dates,16,16]")
        if not np.array_equal(matrix, matrix.transpose(0, 2, 1)) or np.any(np.diagonal(matrix, axis1=1, axis2=2)):
            raise ValueError("Expected symmetric co-occurrence matrices with zero diagonal")
    selected = np.ones(len(dates), dtype=bool)
    if start:
        selected &= dates >= str(np.datetime64(start, "D"))
    if end:
        selected &= dates <= str(np.datetime64(end, "D"))
    if not selected.any():
        raise ValueError("No daily observations in requested date range")
    return dict(dates=dates[selected], risk=risk[idx[selected]], pos=pos[selected], neg=neg[selected],
                banks=banks, codes=codes, news_metadata=metadata)


def valid_windows(risk: np.ndarray, lookback: int, horizon: int):
    if lookback < 1 or horizon < 1:
        raise ValueError("Lookback and horizon must be positive")
    return np.array([t for t in range(lookback - 1, len(risk) - horizon)
                     if np.isfinite(risk[t - lookback + 1:t + 1]).all()
                     and np.isfinite(risk[t + horizon]).all()], dtype=int)


def split_windows(ends: np.ndarray, horizon: int):
    first, second = int(len(ends) * .70), int(len(ends) * .85)
    if first == 0 or second >= len(ends):
        raise ValueError("Insufficient complete daily windows")
    # 移除在下个集合首次预测时点尚不可见的标签对应窗口。
    # 允许目标日等于预测时点，因为预测时已知当天收盘信息。
    train = ends[:first]
    validation = ends[first:second]
    test = ends[second:]
    train = train[train + horizon <= validation[0]]
    validation = validation[validation + horizon <= test[0]]
    if min(len(train), len(validation), len(test)) < 8:
        raise ValueError("Need at least 8 windows per split after horizon-boundary purge")
    return dict(train=train, validation=validation, test=test)


class DailyWindows(Dataset):
    def __init__(self, risk, positive, negative, ends, lookback, horizon, semantic=None):
        self.risk, self.positive, self.negative = risk, positive, negative
        self.ends, self.lookback, self.horizon = ends, lookback, horizon
        self.semantic = semantic

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        t = int(self.ends[index])
        window = slice(t - self.lookback + 1, t + 1)
        inputs = (self.risk[window], self.positive[window], self.negative[window])
        if self.semantic is not None:
            inputs += (self.semantic[window],)
        return tuple(torch.from_numpy(array) for array in (*inputs, self.risk[t + self.horizon]))


def forward_batch(model, batch, return_details=False):
    """同时支持风险与图批次，以及可选的语义历史窗口。"""
    kwargs = {"return_details": return_details}
    if len(batch) == 5:
        kwargs["semantic"] = batch[3]
    return model(*batch[:3], **kwargs)


def predict(model, loader, device, attention=False):
    model.eval()
    estimates, actual, persistence, positive, negative = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(x.to(device) for x in batch)
            risk, target = batch[0], batch[-1]
            if attention:
                estimate, details = forward_batch(model, batch, return_details=True)
                positive.append(details["positive_attention"].cpu().numpy())
                negative.append(details["negative_attention"].cpu().numpy())
            else:
                estimate = forward_batch(model, batch)
            estimates.append(estimate.cpu().numpy())
            actual.append(target.cpu().numpy())
            persistence.append(risk[:, -1].cpu().numpy())
    result = tuple(np.concatenate(values) for values in (estimates, actual, persistence))
    return (*result, (np.concatenate(positive), np.concatenate(negative)) if attention else None)


def score(estimate, actual, persistence):
    return {"mae": float(np.abs(estimate - actual).mean()),
            "rmse": float(np.sqrt(np.mean((estimate - actual) ** 2))),
            "persistence_mae": float(np.abs(persistence - actual).mean()),
            "persistence_rmse": float(np.sqrt(np.mean((persistence - actual) ** 2)))}


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def plot_results(output, history, best_epoch, results, panel, test_ends, horizon, attention):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot([r["epoch"] for r in history], [r["train_mse"] for r in history], label="Train")
    ax.plot([r["epoch"] for r in history], [r["validation_mse"] for r in history], label="Validation")
    ax.axvline(best_epoch, linestyle="--", color="grey")
    ax.set(xlabel="Epoch", ylabel="Standardized MSE", title="Daily training loss")
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(figures / "loss.png", dpi=150); plt.close(fig)
    estimate, actual, persistence = (x * 100 for x in results["test"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, values, title in zip(axes, (estimate, persistence), ("Model", "Persistence")):
        ax.scatter(actual.ravel(), values.ravel(), s=5, alpha=.25)
        low, high = min(actual.min(), values.min()), max(actual.max(), values.max())
        ax.plot([low, high], [low, high], "--", color="#C44B4B")
        ax.set(xlabel="Actual (percentage points)", ylabel="Predicted (percentage points)", title=title)
    fig.savefig(figures / "test_predicted_vs_actual.png", dpi=150); plt.close(fig)
    dates = panel["dates"][test_ends + horizon].astype("datetime64[D]")
    fig, axes = plt.subplots(4, 4, figsize=(17, 12), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(dates, actual[:, i], label="Actual", linewidth=1)
        ax.plot(dates, estimate[:, i], label="Model", linewidth=.8)
        ax.plot(dates, persistence[:, i], label="Persistence", linewidth=.6, alpha=.5)
        ax.set_title(str(panel["codes"][i]), fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.supylabel("Delta CoVaR (percentage points)")
    fig.savefig(figures / "test_bank_timelines.png", dpi=150); plt.close(fig)
    # 展示两类模型最后一个测试输入日的原始图强度。
    for matrices, filename, caption in (([panel["pos"][test_ends[-1]], panel["neg"][test_ends[-1]]],
                                         "latest_input_graphs.png", "log(1 + news count)"),
                                        *([(list(attention), "latest_test_attention.png", "Mean attention")] if attention is not None else [])):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, matrix, polarity in zip(axes, matrices, ("Positive", "Negative")):
            values = np.log1p(matrix) if filename == "latest_input_graphs.png" else matrix.mean(axis=0)
            im = ax.imshow(values, cmap="Blues" if polarity == "Positive" else "Reds")
            ax.set_xticks(range(16), panel["codes"], rotation=90, fontsize=6)
            ax.set_yticks(range(16), panel["codes"], fontsize=6)
            ax.set(title=f"{polarity} | input {panel['dates'][test_ends[-1]]}", xlabel="Source bank", ylabel="Target bank")
            fig.colorbar(im, ax=ax, label=caption, shrink=.7)
        fig.savefig(figures / filename, dpi=150); plt.close(fig)


def fit_one(name, datasets, groups, panel, center, scale, output, args, device):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    semantic_dim = datasets["train"].semantic.shape[-1] if datasets["train"].semantic is not None else 0
    if name == "benchmark":
        model = DailyBenchmarkRiskGRU()
    elif name == "benchmark_semantic":
        model = DailyBenchmarkRiskGRU(semantic_dim=semantic_dim)
    else:
        model = DailyNewsNetworkRiskModel(name, semantic_dim=semantic_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0) for split, ds in datasets.items()}
    best, stale, best_state, best_epoch = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for batch in train:
            batch = tuple(x.to(device) for x in batch)
            risk, target = batch[0], batch[-1]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((forward_batch(model, batch) - target) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {name}, epoch {epoch}")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(risk); count += len(risk)
        estimate, actual, _, _ = predict(model, loaders["validation"], device)
        validation = float(np.mean((estimate - actual) ** 2))
        if not np.isfinite(validation):
            raise RuntimeError("Nonfinite validation loss")
        history.append(dict(epoch=epoch, train_mse=total / count, validation_mse=validation))
        print(f"{name}: epoch {epoch}, train MSE={total/count:.6f}, validation MSE={validation:.6f}", flush=True)
        if validation < best - 1e-8:
            best, stale, best_epoch = validation, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No finite checkpoint")
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=True)
    semantic_pca = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in panel.get("semantic_pca", {}).items()}
    torch.save(dict(model_state=best_state, model_name=name,
                    config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    risk_mean=center, risk_std=scale, bank_names=panel["banks"].tolist(),
                    bank_codes=panel["codes"].tolist(), semantic_input_dim=model.semantic_dim,
                    semantic_pca=semantic_pca if model.semantic_dim else {},
                    inputs=(["delta_covar"] if name == "benchmark" else
                            ["delta_covar", "semantics", "has_news"] if name == "benchmark_semantic" else
                            ["delta_covar", "positive_news", "negative_news"] + (["semantics", "has_news"] if model.semantic_dim else []))), output / "best_model.pt")
    write_csv(output / "loss_history.csv", history, ["epoch", "train_mse", "validation_mse"])
    metrics, results, rows, per_bank, last_attention = {}, {}, [], [], None
    for split, loader in loaders.items():
        estimate, actual, persistence, attention = predict(model, loader, device, attention=name == "attention_bias" and split == "test")
        estimate, actual, persistence = estimate * scale + center, actual * scale + center, persistence * scale + center
        results[split] = (estimate, actual, persistence)
        ends = groups[split]
        metrics[split] = {**score(estimate, actual, persistence), "windows": len(ends),
                          "first_target_date": str(panel["dates"][ends[0] + args.horizon]),
                          "last_target_date": str(panel["dates"][ends[-1] + args.horizon])}
        for i, bank in enumerate(panel["banks"]):
            per_bank.append(dict(model=name, split=split, bank=str(bank), bank_code=str(panel["codes"][i]),
                                 **score(estimate[:, i], actual[:, i], persistence[:, i])))
        for k, t in enumerate(ends):
            for i, bank in enumerate(panel["banks"]):
                rows.append(dict(split=split, input_end_date=str(panel["dates"][t]), target_date=str(panel["dates"][t + args.horizon]),
                                 bank=str(bank), bank_code=str(panel["codes"][i]), actual_delta_covar=float(actual[k, i]),
                                 predicted_delta_covar=float(estimate[k, i]), persistence_delta_covar=float(persistence[k, i])))
        if attention is not None:
            np.savez_compressed(output / "test_last_step_attention.npz", positive=attention[0], negative=attention[1],
                                input_end_dates=panel["dates"][ends], target_dates=panel["dates"][ends + args.horizon],
                                bank_names=panel["banks"], bank_codes=panel["codes"])
            last_attention = (attention[0][-1], attention[1][-1])
    write_csv(output / "predictions.csv", rows, list(rows[0]))
    write_csv(output / "bank_metrics.csv", per_bank, list(per_bank[0]))
    summary = dict(model=name, lookback_trading_days=args.lookback, horizon_trading_days=args.horizon,
                   best_epoch=best_epoch, parameter_count=sum(p.numel() for p in model.parameters()),
                   semantic_input_dim=model.semantic_dim,
                   splits=metrics, unit="fraction; multiply by 100 for percentage points")
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_results(output, history, best_epoch, results, panel, groups["test"], args.horizon, last_attention)
    print(f"{name}: test MAE={metrics['test']['mae']:.6f}, persistence MAE={metrics['test']['persistence_mae']:.6f}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *SEMANTIC_MODEL_NAMES), default="all")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1, help="向前预测的交易日数，目标为该日点值而非区间平均")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target", type=Path, default=ROOT / "data/risk/daily_delta_covar/daily_delta_covar_16_banks.npz")
    parser.add_argument("--news", type=Path, default=ROOT / "data/features/daily_news_cooccurrence.npz")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--semantic-inputs", type=Path, default=None, help="对齐后的日度768维语义文件，启用仅训练期拟合的PCA")
    parser.add_argument("--semantic-dim", type=int, default=16, help="PCA维数，另追加一维有无新闻标记")
    parser.add_argument("--start", default=None, help="限定输入和目标的日期范围，包含边界")
    parser.add_argument("--end", default=None)
    parser.add_argument("--check", action="store_true", help="读取并检查输入和划分，不训练、不写文件")
    parser.add_argument("--config", type=Path, default=None)
    import sys
    cli = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(cli)
    if args.config:
        from code.common.config import load_config, project_path
        config = load_config(args.config)
        for key,value in config.items():
            if not hasattr(args,key):
                parser.error(f"Unknown risk configuration key: {key}")
            if "--"+key.replace("_","-") not in cli:
                setattr(args,key,project_path(value) if key in ("semantic_inputs","output","news","target") and value is not None else value)
    if min(args.lookback, args.horizon, args.epochs, args.patience, args.batch_size, args.threads) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Lengths, counts and learning rate must be positive")
    if args.semantic_dim < 1 or (args.model == "benchmark_semantic" and args.semantic_inputs is None):
        parser.error("Semantic PCA dimension must be positive; benchmark_semantic requires --semantic-inputs")
    if args.output is None:
        args.output = ROOT / "experiments" / ("risk_graph_semantic" if args.semantic_inputs else "risk_graph")
    if args.semantic_inputs and args.output.resolve() == (ROOT / "experiments/risk_graph").resolve():
        parser.error("Use a separate output directory for semantic runs")
    torch.set_num_threads(args.threads)
    panel = load_panel(args.target, args.news, args.start, args.end)
    groups = split_windows(valid_windows(panel["risk"], args.lookback, args.horizon), args.horizon)
    # 仅使用训练输入中的不重复日期拟合统一风险标准化参数。
    visible = np.zeros(len(panel["dates"]), dtype=bool)
    for t in groups["train"]:
        visible[t - args.lookback + 1:t + 1] = True
    observed = panel["risk"][visible]
    center, scale = float(observed.mean()), max(float(observed.std()), 1e-8)
    semantic = None
    if args.semantic_inputs:
        from .semantic_features import load_semantics, fit_semantic_pca
        raw_semantic, counts, metadata = load_semantics(args.semantic_inputs, panel)
        semantic, pca = fit_semantic_pca(raw_semantic, counts, visible, args.semantic_dim)
        panel["semantic_pca"] = pca
        del raw_semantic
    report = dict(inputs=["delta_covar", "positive_news", "negative_news"], lookback_trading_days=args.lookback,
                  horizon_trading_days=args.horizon, risk_mean=center, risk_std=scale,
                  dropped_invalid_windows=(max(0, len(panel["dates"]) - args.lookback - args.horizon + 1)
                                           - len(valid_windows(panel["risk"], args.lookback, args.horizon))),
                  news_metadata=panel["news_metadata"], splits={})
    if args.semantic_inputs:
        report["inputs"].extend(["compressed_semantics", "has_news"])
        report["semantics"] = dict(source=str(args.semantic_inputs.resolve()), pca_dimensions=args.semantic_dim,
                                    model_input_dimensions=semantic.shape[-1], fit_bank_days=pca["fit_bank_days"],
                                    fit_first_date=str(panel["dates"][np.flatnonzero(visible)[0]]),
                                    fit_last_date=str(panel["dates"][np.flatnonzero(visible)[-1]]),
                                    explained_variance_ratio=float(pca["explained_variance_ratio"].sum()),
                                    metadata=metadata)
    for split, ends in groups.items():
        report["splits"][split] = dict(windows=len(ends), first_input_date=str(panel["dates"][ends[0] - args.lookback + 1]),
                                       first_origin=str(panel["dates"][ends[0]]), last_origin=str(panel["dates"][ends[-1]]),
                                       first_target=str(panel["dates"][ends[0] + args.horizon]),
                                       last_target=str(panel["dates"][ends[-1] + args.horizon]))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.check:
        print("CHECK OK; no training or output files written.")
        return
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    risk = ((panel["risk"] - center) / scale).astype(np.float32)
    datasets = {split: DailyWindows(risk, panel["pos"], panel["neg"], ends, args.lookback, args.horizon, semantic=semantic) for split, ends in groups.items()}
    output = args.output / f"lookback_{args.lookback}" / f"horizon_{args.horizon}"
    if args.semantic_inputs:
        output = output / f"semantic_{args.semantic_dim}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.semantic_inputs:
        np.savez_compressed(output / "semantic_pca.npz", **panel["semantic_pca"])
    model_names = SEMANTIC_MODEL_NAMES if args.semantic_inputs else MODEL_NAMES
    summaries = [fit_one(name, datasets, groups, panel, center, scale, output / name, args, device)
                 for name in (model_names if args.model == "all" else (args.model,))]
    comparison = [dict(model=item["model"], split=split, best_epoch=item["best_epoch"], **numbers)
                  for item in summaries for split, numbers in item["splits"].items()]
    write_csv(output / "comparison_metrics.csv", comparison, list(comparison[0]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("mae", "rmse")):
        ax.bar([s["model"] for s in summaries], [s["splits"]["test"][metric] * 100 for s in summaries])
        ax.axhline(summaries[0]["splits"]["test"][f"persistence_{metric}"] * 100, color="#C44B4B", linestyle="--", label="Persistence")
        ax.set(title=f"Test {metric.upper()}", ylabel="Percentage points")
        ax.tick_params(axis="x", rotation=15); ax.legend()
    fig.savefig(output / "test_model_comparison.png", dpi=150); plt.close(fig)
    print(f"Saved daily comparison: {output}")


if __name__ == "__main__":
    main()
