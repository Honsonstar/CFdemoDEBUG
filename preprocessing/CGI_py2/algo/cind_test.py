"""
================================================================================
cind_test.py - 基于高斯过程的条件独立性检验
================================================================================

【文件作用】
实现基于 HSIC 和高斯过程 (GP) 的条件独立性检验算法。通过使用 GP 回归
建模条件分布，消除条件变量 Z 的影响，然后检验 X 和 Y 的残差是否独立。

【使用方法】
```python
import numpy as np
from cind_test import cind_test_new_with_gp, cind_test

# 生成示例数据
np.random.seed(42)
n = 200
z = np.random.randn(n, 2)  # 条件变量
x = np.sin(z[:, 0]) + 0.1 * np.random.randn(n)  # x 由 z 生成
y = np.cos(z[:, 1]) + 0.1 * np.random.randn(n)  # y 由 z 生成

# 检验 X 和 Y 在给定 Z 下的条件独立性
# 由于 x 和 y 都由 z 生成，它们应该条件独立
p_val, stat = cind_test_new_with_gp(x, y, z, alpha=0.8, width=0)
print(f"p-value: {p_val:.4f}")
print(f"statistic: {stat:.4f}")
print(f"条件独立: {p_val > 0.05}")

# 使用别名
result = cind_test(x, y, z)
```

【参数说明】
- x: X 变量数据，shape = (n,)
- y: Y 变量数据，shape = (n,)
- z: 条件变量矩阵，shape = (n, d)
- alpha: 显著性水平相关参数 (默认 0.8，用于 Cri 计算)
- width: 核宽度参数 (0 表示自动选择)

【返回值】
- p_appr: 近似 p 值
- stat: HSIC 统计量

【算法原理】
1. 使用 GP 回归建模 X ~ Z，得到残差 X_res = X - E[X|Z]
2. 使用 GP 回归建模 Y ~ Z，得到残差 Y_res = Y - E[Y|Z]
3. 检验 X_res 和 Y_res 的无条件独立性 (使用 uind_test)
4. 如果 X 和 Y 在给定 Z 下条件独立，则残差应相互独立

【自动带宽选择】
- n < 200: width = 0.8
- n < 1200: width = 0.5
- 其他: width = 0.3

【应用场景】
- 因果发现：检测变量间的因果关系
- 条件独立性检验：检验在给定其他变量下两个变量是否独立
- 特征选择：筛选与目标条件独立的特征

【依赖】
- uind_test: 无条件独立性检验
- gpr: 高斯过程回归
- scipy.optimize.minimize: GP 超参数优化

================================================================================
"""
import numpy as np
from .uind_test import uind_test
from .gpr import gpr
from scipy.optimize import minimize 

def cind_test_new_with_gp(x, y, z, alpha=0.8, width=0): # 默认改为 0.8

    n = x.shape[0]
    if width == 0:
        width = 0.8 if n < 200 else (0.5 if n < 1200 else 0.3)

    # --- 关键修正: 确保 z 是 2D 矩阵 ---
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # --- 确保 x, y 是 1D 向量 (关键！) ---
    x = x.flatten()
    y = y.flatten()
    # print(f"[MEM_TRACE] After flatten | x.shape={x.shape} | y.shape={y.shape} | z.shape={z.shape}", flush=True)

    # --- 关键修正: 使用 ddof=1 对齐 MATLAB ---
    x = (x - np.mean(x)) / np.std(x, ddof=1)
    y = (y - np.mean(y)) / np.std(y, ddof=1)
    z = (z - np.mean(z, axis=0)) / np.std(z, axis=0, ddof=1)

    # GP 1 (x|z)
    hyp_x = np.array([np.log(1.0), np.log(1.0), np.log(0.1)])
    covfunc = ['covSum', ['covSEiso', 'covNoise']]
    def nlml_x(theta):
        val = gpr(theta, covfunc, z, x)
        if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1: return val[0]
        return val

    res_x = minimize(nlml_x, hyp_x, method='L-BFGS-B',
                     options={'disp': False, 'maxiter': 50, 'maxfun': 50})
    pred_x = gpr(res_x.x, covfunc, z, x, z)
    # 确保 pred_x 是 1D 向量
    if isinstance(pred_x, tuple):
        pred_x_flat = pred_x[0].flatten()
    else:
        pred_x_flat = pred_x.flatten()
    res_x_val = x - pred_x_flat

    # GP 2 (y|z)
    hyp_y = np.array([np.log(1.0), np.log(1.0), np.log(0.1)])
    def nlml_y(theta):
        val = gpr(theta, covfunc, z, y)
        if isinstance(val, (tuple, list, np.ndarray)) and np.size(val) > 1: return val[0]
        return val

    res_y = minimize(nlml_y, hyp_y, method='L-BFGS-B',
                     options={'disp': False, 'maxiter': 50, 'maxfun': 50})
    pred_y = gpr(res_y.x, covfunc, z, y, z)
    # 确保 pred_y 是 1D 向量
    if isinstance(pred_y, tuple):
        pred_y_flat = pred_y[0].flatten()
    else:
        pred_y_flat = pred_y.flatten()
    res_y_val = y - pred_y_flat

    # --- 关键修复: 强制确保残差是 1D 向量 (n,) 而不是 (n, n) ---
    res_x_val = res_x_val.flatten()
    res_y_val = res_y_val.flatten()

    # 强制检查维度
    assert res_x_val.shape[0] == n, "res_x_val dimension error: {}".format(res_x_val.shape)
    assert res_y_val.shape[0] == n, "res_y_val dimension error: {}".format(res_y_val.shape)

    # --- 传递 alpha (应该是 0.8) ---
    stat, cri, p_val, cri_appr, p_appr = uind_test(res_x_val, res_y_val, alpha, width)

    return p_appr, stat

# 别名
cind_test = cind_test_new_with_gp