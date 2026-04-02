"""
================================================================================
CGI_py - 因果图形推理 Python 实现
================================================================================

【包说明】
CGI_py 是 CGI (Causality Graphical Inference) 算法的 Python 实现，
是一个基于高斯过程和核方法的条件独立性检验工具，用于因果发现和因果推理。

【核心功能】
- 基于 HSIC 的独立性检验 (无条件)
- 基于 GP 回归的条件独立性检验
- KCIT (核条件独立性检验)
- PaCoTest (部分相关检验)
- 高斯过程回归与协方差函数

【模块结构】
- algo: 核心算法模块
  - 核函数: kernel, kernel_matrix
  - 距离计算: dist2
  - 矩阵运算: eigdec, pdinv, stack, solve_chol
  - 优化器: minimize
  - 协方差函数: cov_sum, cov_se_iso, cov_se_ard, cov_noise, cov_matern
  - 高斯过程: gpr, fit_gpr
  - 独立性检验: paco_test, kcit, uind_test, cind_test

【使用方法】
```python
import numpy as np
from CGI_py import kcit, paco_test, gpr

# 示例: KCIT 条件独立性检验
x = np.random.randn(100)
y = np.sin(x) + 0.1 * np.random.randn(100)
z = np.random.randn(100)

ind, stat, p_val = kcit(x, y, z, alpha=0.05)
print(f"条件独立: {ind}, p-value: {p_val:.4f}")
```

【依赖】
- numpy >= 1.18.0
- scipy >= 1.5.0
- scikit-learn >= 0.22.0 (可选)

【版本】
1.0.0

================================================================================
"""

from .algo import (
    kernel, kernel_matrix, dist2, eigdec, pdinv, stack, solve_chol,
    minimize, cov_sum, cov_se_iso, cov_se_ard, cov_noise, cov_matern,
    gpr, gpr_multi, medbw, paco_test, kcit, uind_test,
    cind_test_new_with_gp, fit_gpr
)

__version__ = '1.0.0'
