"""清洗日度收盘价，并核对16家目标银行。

从项目根目录运行：python -m code.risk_measure.prepare_bank_prices
输出目录：data/risk/price_analysis/
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
OUT = ROOT / 'data' / 'risk' / 'price_analysis'
PRICE_FILE = DATA / 'raw' / 'bank_prices_raw.xlsx'
ALIAS_FILE = DATA / 'raw' / 'bank_aliases.xlsx'
SYSTEM_BANK_COUNT = 16


def extract_code(header: str | None) -> str | None:
    match = re.search(r'\((\d{6})\)', header or '')
    return match.group(1) if match else None


def log_returns(prices: np.ndarray) -> np.ndarray:
    """计算相邻交易行的对数收益率；缺失价格使相邻两组收益率无效。"""
    prices = np.asarray(prices, dtype=float)
    result = np.full_like(prices, np.nan, dtype=float)
    valid = np.isfinite(prices[1:]) & np.isfinite(prices[:-1]) & (prices[1:] > 0) & (prices[:-1] > 0)
    result[1:][valid] = np.log(prices[1:][valid] / prices[:-1][valid])
    return result


def load_bank_names() -> list[str]:
    book = load_workbook(ALIAS_FILE, read_only=True, data_only=True)
    try:
        names = [str(r[1]).strip() for r in list(book.active.values)[1:] if len(r) > 1 and r[0] and r[1]]
    finally:
        book.close()
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError(f'Expected 16 distinct annotated banks, got {names}')
    return names


def load_prices():
    """仅从原始工作簿读取16家目标银行的列。"""
    banks = load_bank_names()
    book = load_workbook(PRICE_FILE, read_only=True, data_only=True)
    try:
        rows = book.active.iter_rows(values_only=True)
        headers = next(rows)
        source_columns = {}
        for col, header in enumerate(headers):
            name = str(header).split('(')[0] if header else ''
            bank = '上海浦东发展银行' if name == '浦发银行' else name
            if bank in banks:
                if bank in source_columns:
                    raise ValueError(f'Duplicate source columns for {bank}')
                source_columns[bank] = (col, extract_code(header))
        if set(source_columns) != set(banks) or any(source_columns[b][1] is None for b in banks):
            raise ValueError(f'Cannot match all 16 annotated banks: {source_columns}')
        codes = [source_columns[b][1] for b in banks]
        dates, values = [], []
        for row in rows:
            date = row[0] if row else None
            if not hasattr(date, 'date'):
                continue
            dates.append(date.date().isoformat())
            vals = []
            for bank in banks:
                col = source_columns[bank][0]
                raw = row[col] if col < len(row) else None
                try:
                    value = float(raw)
                    vals.append(value if np.isfinite(value) and value > 0 else np.nan)
                except (TypeError, ValueError):
                    vals.append(np.nan)
            values.append(vals)
    finally:
        book.close()
    if not dates or len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError('Date column is empty, duplicated, or unsorted')
    return np.array(dates), np.array(values, dtype=float), codes, banks, source_columns


def write_csv(path: Path, header: list[str], rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates, prices, codes, banks, source_columns = load_prices()
    returns = np.column_stack([log_returns(prices[:, j]) for j in range(prices.shape[1])])
    active_banks = np.isfinite(returns).sum(axis=1)
    system_log_return = np.full(len(dates), np.nan)
    system_valid = active_banks == SYSTEM_BANK_COUNT
    system_log_return[system_valid] = returns[system_valid].mean(axis=1)
    np.savez_compressed(OUT / 'clean_16_bank_prices_returns_and_system.npz',
                        dates=dates, bank_names=np.array(banks), bank_codes=np.array(codes),
                        bank_prices=prices, bank_log_returns=returns,
                        system_log_return=system_log_return,
                        system_active_banks=active_banks)
    rows = []
    for j, bank in enumerate(banks):
        valid_p = np.isfinite(prices[:, j])
        valid_r = np.isfinite(returns[:, j])
        rows.append([bank, codes[j], dates[np.flatnonzero(valid_p)[0]] if valid_p.any() else '',
                     dates[np.flatnonzero(valid_p)[-1]] if valid_p.any() else '',
                     int(valid_p.sum()), int((~valid_p).sum()), f'{valid_p.mean():.6f}',
                     int(valid_r.sum()), f'{np.nanmean(returns[:, j]):.8f}',
                     f'{np.nanstd(returns[:, j], ddof=1):.8f}',
                     f'{np.nanmin(returns[:, j]):.8f}', f'{np.nanmax(returns[:, j]):.8f}'])
    write_csv(OUT / 'bank_coverage_and_return_statistics.csv',
              ['bank', 'code', 'first_price_date', 'last_price_date', 'valid_price_days', 'missing_price_days',
               'price_coverage_ratio', 'valid_return_days', 'mean_daily_log_return', 'std_daily_log_return',
               'min_daily_log_return', 'max_daily_log_return'], rows)
    write_csv(OUT / 'source_column_audit.csv', ['bank', 'code', 'selected_source_column'],
              [[bank, codes[j], source_columns[bank][0] + 1] for j, bank in enumerate(banks)])
    write_csv(OUT / 'bank_daily_clean.csv', ['date', 'bank', 'code', 'adjusted_close', 'log_return'],
              ([date, bank, codes[j], '' if not np.isfinite(prices[k, j]) else f'{prices[k, j]:.6f}',
                '' if not np.isfinite(returns[k, j]) else f'{returns[k, j]:.10f}']
               for k, date in enumerate(dates) for j, bank in enumerate(banks)))
    plt.rcParams.update({'figure.dpi': 140, 'font.size': 9})
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(codes, [int(np.isfinite(prices[:, j]).sum()) for j in range(len(banks))], color='#3478A3')
    ax.set(xlabel='Days with valid adjusted close', title='Coverage of the 16 annotated banks')
    fig.tight_layout(); fig.savefig(OUT / 'bank_price_coverage.png'); plt.close(fig)
    months = np.array([d[:7] for d in dates]); unique_months = sorted(set(months))
    coverage = np.array([[np.isfinite(prices[months == m, j]).mean() for m in unique_months] for j in range(len(banks))])
    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(coverage, aspect='auto', vmin=0, vmax=1, cmap='Blues')
    ax.set_yticks(range(16), codes); ticks = np.arange(0, len(unique_months), 12)
    ax.set_xticks(ticks, np.array(unique_months)[ticks], rotation=45)
    ax.set(title='Monthly valid-price coverage', xlabel='Month')
    fig.colorbar(im, ax=ax, label='Fraction of trading days'); fig.tight_layout()
    fig.savefig(OUT / 'monthly_price_coverage_heatmap.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(codes, [np.nanstd(returns[:, j], ddof=1) * 100 for j in range(len(banks))], color='#C37542')
    ax.set(xlabel='Daily log-return standard deviation (%)', title='Daily volatility by bank')
    fig.tight_layout(); fig.savefig(OUT / 'bank_daily_volatility.png'); plt.close(fig)
    print(f'Cleaned {len(dates)} trading dates and {len(banks)} banks; outputs: {OUT}')


if __name__ == '__main__':
    main()
