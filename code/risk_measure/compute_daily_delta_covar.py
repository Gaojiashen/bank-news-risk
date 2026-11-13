"""使用历史银行损失与系统损失估计日度Delta CoVaR。

从项目根目录运行：
    python -m code.risk_measure.compute_daily_delta_covar

沿用500个交易日、95%分位数回归的定义，在每个交易日估计，
包含最后一个未完整结束的自然月；每次拟合仅使用截至当天的观测。
"""

from __future__ import annotations

import json

import numpy as np

from .prepare_bank_prices import OUT as PRICE_OUT, ROOT, SYSTEM_BANK_COUNT
from code.common.quantile_regression import delta_covar, fit_quantile_line


OUT = ROOT / "data/risk/daily_delta_covar"
WINDOW_DAYS = 500
MIN_PAIR_DAYS = 400
TAIL_QUANTILE = 0.95


def main() -> None:
    source = PRICE_OUT / "clean_16_bank_prices_returns_and_system.npz"
    if not source.exists():
        raise FileNotFoundError(f"Run code.risk_measure.prepare_bank_prices first: {source}")
    with np.load(source) as data:
        dates = data["dates"]
        banks = data["bank_names"]
        codes = data["bank_codes"]
        bank_losses = -data["bank_log_returns"]
        system_losses = -data["system_log_return"]
    if bank_losses.shape != (len(dates), len(banks)) or system_losses.shape != (len(dates),):
        raise ValueError("Dates, bank returns and system returns have incompatible shapes")

    shape = bank_losses.shape
    delta = np.full(shape, np.nan)
    var_tail = np.full(shape, np.nan)
    var_median = np.full(shape, np.nan)
    alpha = np.full(shape, np.nan)
    beta = np.full(shape, np.nan)
    covar_distress = np.full(shape, np.nan)
    covar_median = np.full(shape, np.nan)
    n_pairs = np.zeros(shape, dtype=np.int16)
    status = np.full(shape, "insufficient", dtype="<U20")

    for day in range(MIN_PAIR_DAYS - 1, len(dates)):
        begin = max(0, day - WINDOW_DAYS + 1)
        y_window = system_losses[begin:day + 1]
        for bank in range(len(banks)):
            x_window = bank_losses[begin:day + 1, bank]
            valid = np.isfinite(x_window) & np.isfinite(y_window)
            count = int(valid.sum())
            n_pairs[day, bank] = count
            if count < MIN_PAIR_DAYS:
                continue
            x, y = x_window[valid], y_window[valid]
            try:
                a, b = fit_quantile_line(x, y, TAIL_QUANTILE)
            except (ValueError, RuntimeError):
                status[day, bank] = "regression_failed"
                continue
            tail, median = np.quantile(x, [TAIL_QUANTILE, 0.5])
            alpha[day, bank], beta[day, bank] = a, b
            var_tail[day, bank], var_median[day, bank] = tail, median
            covar_distress[day, bank] = a + b * tail
            covar_median[day, bank] = a + b * median
            delta[day, bank] = delta_covar(x, b, TAIL_QUANTILE)
            status[day, bank] = "ok"
        if day % 100 == 0 or day == len(dates) - 1:
            print(f"Estimated through {dates[day]} ({day + 1}/{len(dates)} dates)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "daily_delta_covar.npy", delta)
    np.savez_compressed(
        OUT / "daily_delta_covar_16_banks.npz",
        dates=dates, bank_names=banks, bank_codes=codes,
        delta_covar=delta, bank_var95=var_tail, bank_var50=var_median,
        quantile_alpha=alpha, quantile_beta=beta,
        system_covar_distress=covar_distress,
        system_covar_median=covar_median,
        paired_days=n_pairs, status=status,
        window_days=np.array(WINDOW_DAYS),
        minimum_paired_days=np.array(MIN_PAIR_DAYS),
        tail_quantile=np.array(TAIL_QUANTILE),
    )
    (OUT / "calculation_metadata.json").write_text(
        json.dumps({
            "source": str(source),
            "system_definition": "equal-weight mean of the same 16 banks, requiring all valid returns",
            "system_required_active_banks": SYSTEM_BANK_COUNT,
            "loss_definition": "negative daily adjusted-close log return",
            "quantile_method": "linear quantile regression via exact linear programming",
            "window_trading_days": WINDOW_DAYS,
            "minimum_paired_days": MIN_PAIR_DAYS,
            "loss_tail_quantile": TAIL_QUANTILE,
            "delta_definition": "system CoVaR at bank VaR95 minus system CoVaR at bank VaR50",
            "unit": "fraction; multiply by 100 for percentage points",
            "array_axes": "trading date, bank",
            "missing": "NaN where the rolling window has too few paired returns or regression fails",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Valid estimates: {int(np.isfinite(delta).sum())}/{delta.size}; outputs: {OUT}")


if __name__ == "__main__":
    main()
