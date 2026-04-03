"""
================================================================================
kcit.py - 基于核的条件独立性检验 (Kernel-based Conditional Independence Test)
================================================================================

【文件作用】
实现 KCIT (Kernel-based Conditional Independence Test) 算法，用于检验在给定
条件变量的情况下两个变量之间的非线性条件独立性。这是 CGI 算法的核心组件，
用于检测非线性因果关系。

【使用方法】
```python
import numpy as np
from kcit import kcit

# 1. 无条件检验 (检验 X 和 Y 的非线性相关性)
x = np.random.randn(100)
y = np.sin(x) + 0.1 * np.random.randn(100)  # 非线性关系
ind, stat, p_val = kcit(x, y, np.array([[]]), alpha=0.05)
# ind = False (不独立)

# 2. 条件检验 (检验 X 和 Y 在给定 Z 下的条件独立性)
z = np.random.randn(100)
x = np.sin(z) + 0.1 * np.random.randn(100)
y = np.cos(z) + 0.1 * np.random.randn(100)  # x 和 y 都由 z 生成
ind, stat, p_val = kcit(x, y, z.reshape(-1, 1), alpha=0.05)
# ind = True (条件独立，因为都由 z 决定)
```

【参数说明】
- x: X 变量数据，shape = (n,)
- y: Y 变量数据，shape = (n,)
- z: 条件变量，可以是：
  - None 或空数组：执行无条件检验
  - shape = (n, d)：条件变量矩阵
- width: 核宽度参数，默认 0（自动选择）
- alpha: 显著性水平，默认 0.05

【返回值】
- ind: True 表示条件独立，False 表示不独立
- stat: 检验统计量
- p_val: p 值

【算法原理】
- 使用高斯核 (RBF kernel) 计算特征空间中的相关性
- 通过条件分布的均值 Embedding 构造检验统计量
- 使用 permutation test 或渐近分布计算 p 值

【应用场景】
- 因果发现：检测非线性因果关系
- 特征选择：筛选与目标条件独立的特征
- 独立性检验：复杂数据中的依赖关系检测

【依赖】
- uind_test: 无条件独立性检验
- cind_test_new_with_gp: 条件独立性检验

================================================================================
"""

import numpy as np
from .uind_test import uind_test
from .cind_test import cind_test_new_with_gp

def kcit(x, y, z=None, width=0, alpha=0.05):
    # ========== [KCIT TRACE] 输入形状检查 ==========
    # print(f"[KCIT_TRACE] kcit called | x.shape={np.array(x).shape} | y.shape={np.array(y).shape} | z.shape={np.array(z).shape if z is not None and hasattr(z, 'shape') else 'None'}", flush=True)

    x = np.array(x).flatten()
    y = np.array(y).flatten()
    # print(f"[KCIT_TRACE] After flatten | x.shape={x.shape} | y.shape={y.shape}", flush=True)

    # --- 关键修正 3: 强制对齐 MATLAB 的 hardcoded alpha ---
    # MATLAB KCIT.m 第 47 行使用的是 0.8，而非传入的 alpha
    # 这个 0.8 影响 Cri 的计算，虽然我们主要看 p_val，但为了绝对一致，必须对齐。
    force_alpha_for_calc = 0.8

    if z is None or (isinstance(z, np.ndarray) and z.size == 0):
        # Unconditional
        # print(f"[KCIT_TRACE] Calling uind_test (unconditional)", flush=True)
        stat, cri, p_val, cri_appr, p_appr = uind_test(x, y, force_alpha_for_calc, width)
        # 判定逻辑：p_value > 0.05 (这个阈值是固定的，与 force_alpha 无关)
        ind = (p_appr > 0.05)
        return ind, stat, p_appr
    else:
        # Conditional
        if z.ndim == 1: z = z.reshape(-1, 1)
        # print(f"[KCIT_TRACE] Calling cind_test (conditional) | z.shape={z.shape}", flush=True)
        p_appr, stat = cind_test_new_with_gp(x, y, z, force_alpha_for_calc, width)
        ind = (p_appr > 0.05)
        return ind, stat, p_appr