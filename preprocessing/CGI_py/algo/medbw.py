"""
================================================================================
medbw.py - RBF 核带宽的中位数启发式选择
================================================================================

【文件作用】
使用中位数距离启发式方法自动设置 RBF (高斯) 核的带宽参数。
这是核方法中的常用技术，可以无需交叉验证自动选择合理的核宽度。

【使用方法】
```python
import numpy as np
from medbw import medbw

# 生成随机数据
np.random.seed(42)
X = np.random.randn(100, 5)  # 100个样本，5维特征

# 计算带宽 (使用最多50个点计算中位数)
sigma = medbw(X, maxpoints=50)
print(f"自动选择的带宽 sigma: {sigma}")

# 计算对应的 length_scale (用于 RBF 核)
length_scale = sigma
print(f"length_scale: {length_scale}")
```

【参数说明】
- X: 数据矩阵，shape = (n, p)
- maxpoints: 计算中位数时使用的最大点数，用于降低计算复杂度

【返回值】
- sigma: 带宽值 (标准差)

【算法原理】
1. 如果样本数超过 maxpoints，随机或取前 maxpoints 个点
2. 计算所有成对点的欧氏距离
3. 只取上三角部分（不含对角线）
4. 计算距离的中位数 median_dist
5. sigma = sqrt(0.5 * median_dist)

【应用场景】
- HSIC 独立性检验中的核宽度选择
- 高斯过程回归中的自动超参数选择
- 核密度估计

【依赖】
- numpy

================================================================================
"""

import numpy as np


def medbw(X: np.ndarray, maxpoints: int) -> float:
    """
    Compute bandwidth using median distance heuristic.

    Args:
        X: (n, p) matrix of n datapoints with dimensionality p
        maxpoints: maximum number of points to use

    Returns:
        sigma: bandwidth value
    """
    if maxpoints < 1 or maxpoints != int(maxpoints):
        raise ValueError('maxpoints must be a positive integer')

    n = X.shape[0]

    # Truncate data if more points than maxpoints
    if n > maxpoints:
        med = X[:maxpoints, :]
        n = maxpoints
    else:
        med = X

    # Find median distance between points
    G = np.sum(med * med, axis=1)
    Q = np.tile(G.reshape(-1, 1), (1, n))
    R = np.tile(G.reshape(1, -1), (n, 1))
    dists = Q + R - 2 * med @ med.T

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists_upper = dists[mask]
    dists_upper = dists_upper[dists_upper > 0]

    if len(dists_upper) == 0:
        return 1.0

    sigma = np.sqrt(0.5 * np.median(dists_upper))

    return sigma
