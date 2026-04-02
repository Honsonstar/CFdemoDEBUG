"""
================================================================================
uind_test.py - 无条件独立性检验 (Unconditional Independence Test)
================================================================================

【文件作用】
实现基于 HSIC (Hilbert-Schmidt Independence Criterion) 的无条件独立性检验。
用于检测两个变量之间的非线性依赖关系，是 KCIT 的组成部分。

【使用方法】
```python
import numpy as np
from uind_test import uind_test

# 生成测试数据
np.random.seed(42)
x = np.random.randn(100)
y = 0.5 * x + 0.5 * np.random.randn(100)  # 相关

# 执行无条件独立性检验
# alpha: 显著性水平 (用于计算临界值)
# width: 核宽度参数
stat, cri, p_val, cri_appr, p_appr = uind_test(x, y, alpha=0.8, width=0)

# 结果判断
if p_appr > 0.05:
    print("变量独立")
else:
    print("变量不独立")
```

【参数说明】
- x: X 变量，shape = (n,) 或 (n,1)
- y: Y 变量，shape = (n,) 或 (n,1)
- alpha: 显著性水平参数（用于临界值计算），默认 0.8
- width: 核宽度，默认 0（自动根据样本数选择）

【自动宽度选择】
- n < 200: width = 0.8
- n < 1200: width = 0.5
- 其他: width = 0.3

【返回值】
- stat: 检验统计量 (HSIC)
- cri: 临界值 (基于 alpha)
- p_val: p 值
- cri_appr: 近似临界值
- p_appr: 近似 p 值（用于判断）

【算法原理】
- 使用 HSIC (Hilbert-Schmidt Independence Criterion)
- 通过核方法检测非线性依赖
- 基于特征值分解近似分布

【依赖】
- numpy
- scipy
- kernel
- eigdec

================================================================================
"""

import numpy as np
from scipy import stats
from .kernel import kernel
from .eigdec import eigdec

def uind_test(x: np.ndarray, y: np.ndarray, alpha: float = 0.8, width: float = 0) -> tuple:
    # ========== [UIND_TRACE] 输入形状检查 ==========
    # print(f"[UIND_TRACE] uind_test called | x.shape={x.shape} | y.shape={y.shape}", flush=True)

    # 确保输入是 1D 向量
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    # print(f"[UIND_TRACE] After reshape | x.shape={x.shape} | y.shape={y.shape}", flush=True)
    T = len(y)

    # --- 关键修正 2: 恢复 MATLAB 的硬编码宽度逻辑 ---
    if width == 0:
        if T < 200:
            width = 0.8
        elif T < 1200:
            width = 0.5
        else:
            width = 0.3

    # theta = 1/width^2
    theta = 1.0 / (width ** 2)

    # 标准化 (复现 MATLAB 行为) - 使用 ddof=1 对齐 MATLAB
    x = (x - np.mean(x)) / (np.std(x, ddof=1) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y, ddof=1) + 1e-10)

    H = np.eye(T) - np.ones((T, T)) / T

    # 计算核矩阵
    # print(f"[UIND_TRACE] Before kernel | x.shape={x.shape} | y.shape={y.shape}", flush=True)
    res_x = kernel(x, x, np.array([theta, 1.0]))
    Kx = res_x[0] if isinstance(res_x, (tuple, list)) else res_x
    # print(f"[UIND_TRACE] After kernel x | Kx.shape={Kx.shape}", flush=True)

    res_y = kernel(y, y, np.array([theta, 1.0]))
    Ky = res_y[0] if isinstance(res_y, (tuple, list)) else res_y
    # print(f"[UIND_TRACE] After kernel y | Ky.shape={Ky.shape}", flush=True)

    Kx = H @ Kx @ H
    Ky = H @ Ky @ H

    Sta = np.trace(Kx @ Ky)

    num_eig = min(T // 2, 100)
    
    res_ex = eigdec((Kx + Kx.T) / 2, num_eig)
    eig_Kx = res_ex[0] if isinstance(res_ex, (tuple, list)) else res_ex

    res_ey = eigdec((Ky + Ky.T) / 2, num_eig)
    eig_Ky = res_ey[0] if isinstance(res_ey, (tuple, list)) else res_ey

    eig_prod = (eig_Kx.reshape(-1, 1) @ eig_Ky.reshape(1, -1)).flatten()
    eig_prod = eig_prod[eig_prod > np.max(eig_prod) * 1e-6]

    mean_appr = np.trace(Kx) * np.trace(Ky) / T
    var_appr = 2 * np.trace(Kx @ Kx) * np.trace(Ky @ Ky) / (T ** 2)

    if mean_appr > 0 and var_appr > 0:
        k_appr = mean_appr ** 2 / var_appr
        theta_appr = var_appr / mean_appr

        # 使用传入的 alpha 计算临界值 - 使用 sf 函数更稳定
        Cri_appr = stats.gamma.ppf(1 - alpha, a=k_appr, scale=theta_appr)
        # 修复：使用 sf (Survival Function) 代替 1-cdf，精度更高
        p_appr = stats.gamma.sf(Sta, a=k_appr, scale=theta_appr)
    else:
        Cri_appr = 0
        p_appr = 1.0

    return Sta, Cri_appr, p_appr, Cri_appr, p_appr