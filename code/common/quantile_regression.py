"""基于损失的分位数回归与Delta CoVaR计算公式。"""
import numpy as np
from scipy.optimize import linprog
TAIL_QUANTILE=.95

def fit_quantile_line(x: np.ndarray, y: np.ndarray, q: float) -> tuple[float, float]:
    """通过非对称绝对损失的线性规划精确求解线性分位数回归。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3 or not (0 < q < 1):
        raise ValueError('Invalid quantile-regression input')
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or np.ptp(x) < 1e-12:
        raise ValueError('Nonfinite values or constant bank losses')
    # 对两个序列中心化和缩放以提高数值稳定性；残差为y-a-b*x。
    xm, ym = np.mean(x), np.mean(y)
    xs, ys = np.std(x), np.std(y)
    xn, yn = (x - xm) / xs, (y - ym) / ys
    n = len(x)
    aeq = np.zeros((n, 2 + 2*n))
    aeq[:, 0] = 1.0; aeq[:, 1] = xn
    aeq[np.arange(n), 2 + np.arange(n)] = 1.0
    aeq[np.arange(n), 2 + n + np.arange(n)] = -1.0
    costs = np.r_[0.0, 0.0, np.full(n, q), np.full(n, 1 - q)]
    result = linprog(costs, A_eq=aeq, b_eq=yn, bounds=[(None, None), (None, None)] + [(0, None)] * (2*n), method='highs')
    if not result.success:
        raise RuntimeError(f'Quantile regression failed: {result.message}')
    slope = float(result.x[1] * ys / xs)
    intercept = float(ym + ys * result.x[0] - slope * xm)
    return intercept, slope


def delta_covar(bank_losses: np.ndarray, slope: float, q: float = TAIL_QUANTILE) -> float:
    return float(slope * (np.quantile(bank_losses, q) - np.quantile(bank_losses, 0.5)))
