"""
================================================================================
paco_test.py - 偏相关检验 (Partial Correlation Test)
================================================================================

【文件作用】
实现基于偏相关的条件独立性检验 (Partial Correlation Test)，用于检验在给定
条件变量 Z 的情况下，X 和 Y 是否条件独立。这是 CGI 算法的线性条件独立性检验。

【使用方法】
```python
import numpy as np
from paco_test import paco_test

# 1. 无条件检验 (Z=None)
x = np.random.randn(100)
y = 0.5 * x + 0.1 * np.random.randn(100)  # y 与 x 相关
result = paco_test(x, y, None, alpha=0.05)
# result = False (不独立)

# 2. 条件检验 (Z为条件变量)
z = np.random.randn(100)
x = 0.3 * z + 0.1 * np.random.randn(100)
y = 0.3 * z + 0.1 * np.random.randn(100)  # x 和 y 都受 z 影响
result = paco_test(x, y, z, alpha=0.05)
# result = True (在控制 z 后，x 和 y 条件独立)
```

【参数说明】
- x: X 变量数据，shape = (n,) 或 (n,1)
- y: Y 变量数据，shape = (n,) 或 (n,1)
- Z: 条件变量，可以是：
  - None 或空数组：执行无条件检验（检验 X 和 Y 的边际相关性）
  - shape = (n,)：单个条件变量
  - shape = (n, d)：多个条件变量
- alpha: 显著性水平，默认 0.05

【返回值】
- True: X 和 Y 条件独立
- False: X 和 Y 条件不独立

【算法原理】
1. 使用协方差矩阵求逆（Precision Matrix）方法计算偏相关系数
2. 计算 Fisher Z 变换
3. 使用 t 检验判断相关性是否显著

【应用场景】
- 因果发现算法中的条件独立性筛选
- 控制混淆变量后的相关性分析

================================================================================
PaCoTest - Partial Correlation Test

Implements partial correlation test for conditional independence.

Based on partial correlation computation from:
http://en.wikipedia.org/wiki/Partial_correlation
================================================================================
"""

import numpy as np
from scipy import stats


def paco_test(x: np.ndarray, y: np.ndarray, Z: np.ndarray = None, alpha: float = 0.05):
    """
    Test conditional independence using partial correlation.

    Args:
        x: (n,) array of samples for variable X
        y: (n,) array of samples for variable Y
        Z: (n, d) array of conditioning variables, or None for unconditional test
        alpha: significance level

    Returns:
        cit: True if X and Y are conditionally independent, False otherwise
        stat: test statistic
        pval: p-value
    """
    x = x.flatten()
    y = y.flatten()
    n = len(x)

    if Z is None or Z.size == 0:
        # Unconditional test: compute Pearson correlation
        # 注意：stats.pearsonr 使用 ddof=1 计算标准差，与 MATLAB 一致
        if np.std(x, ddof=1) == 0 or np.std(y, ddof=1) == 0:
            pcc = 0
        else:
            pcc, _ = stats.pearsonr(x, y)
        ncit = 0
    else:
        # ========== [新方法] 使用协方差矩阵求逆 (Precision Matrix) ==========
        Z = Z.reshape(n, -1)
        ncit = Z.shape[1]  # 条件变量数量（不含截距）

        # 将 x, y 和 Z 合并为一个矩阵
        # 列顺序: x, y, z1, z2, ...
        data_sub = np.column_stack((x, y, Z))

        # 计算协方差矩阵 (ddof=1 与 MATLAB 一致)
        C = np.cov(data_sub, rowvar=False, ddof=1)

        # 计算精度矩阵（使用伪逆防止奇异矩阵）
        try:
            invC = np.linalg.pinv(C)
        except np.linalg.LinAlgError:
            # 如果求逆失败，回退到残差法
            Z_with_intercept = np.column_stack([np.ones(n), Z])
            wx, _, _, _ = np.linalg.lstsq(Z_with_intercept, x, rcond=None)
            rx = x - Z_with_intercept @ wx
            wy, _, _, _ = np.linalg.lstsq(Z_with_intercept, y, rcond=None)
            ry = y - Z_with_intercept @ wy
            if np.std(rx, ddof=1) == 0 or np.std(ry, ddof=1) == 0:
                pcc = 0
            else:
                pcc, _ = stats.pearsonr(rx, ry)
        else:
            # 从精度矩阵提取偏相关系数
            # invC[0,1] 是 x 和 y 的协方差的逆（条件协方差）
            # r = -C_xy / sqrt(C_xx * C_yy) = -invC[0,1] / sqrt(invC[0,0] * invC[1,1])
            denom = np.sqrt(invC[0, 0] * invC[1, 1])
            if denom > 1e-10:
                pcc = -invC[0, 1] / denom
            else:
                pcc = 0

        # 限制 pcc 在 [-1, 1] 防止 log 报错
        pcc = np.clip(pcc, -0.999999, 0.999999)

    # 自由度 = N - |Z| - 3 (与 MATLAB 一致)
    df = n - ncit - 3

    # Fisher's z-transform
    zpcc = 0.5 * np.log((1 + pcc) / (1 - pcc))

    # Test statistic
    stat = np.sqrt(df) * np.abs(zpcc)

    # Compute p-value (two-tailed test)
    pval = 2 * stats.norm.cdf(-np.abs(stat))

    # Critical value
    crit = stats.norm.ppf(1 - alpha / 2)

    # Return True if independent (fail to reject null)
    cit = stat <= crit

    return cit, stat, pval


def paco_test_stat(x: np.ndarray, y: np.ndarray, Z: np.ndarray = None) -> tuple:
    """
    Compute partial correlation and p-value.

    Returns:
        pcc: partial correlation coefficient
        p_value: p-value for independence test
    """
    x = x.flatten()
    y = y.flatten()
    n = len(x)

    if Z is None or Z.size == 0:
        pcc, p_value = stats.pearsonr(x, y)
        return pcc, p_value

    # 使用协方差矩阵求逆法
    Z = Z.reshape(n, -1)
    data_sub = np.column_stack((x, y, Z))
    C = np.cov(data_sub, rowvar=False, ddof=1)

    try:
        invC = np.linalg.pinv(C)
        denom = np.sqrt(invC[0, 0] * invC[1, 1])
        if denom > 1e-10:
            pcc = -invC[0, 1] / denom
        else:
            pcc = 0
    except:
        pcc = 0

    pcc = np.clip(pcc, -0.999999, 0.999999)

    # 计算 p-value
    ncit = Z.shape[1]
    df = n - ncit - 3
    zpcc = 0.5 * np.log((1 + pcc) / (1 - pcc))
    stat = np.sqrt(df) * np.abs(zpcc)
    p_value = 2 * stats.norm.cdf(-np.abs(stat))

    return pcc, p_value
