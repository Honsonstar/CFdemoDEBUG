"""
================================================================================
gpr.py - 高斯过程回归 (Gaussian Process Regression)
================================================================================

【文件作用】
实现高斯过程回归 (GPR) 算法，用于从观测数据中学习非线性函数关系。
GPR 是一种贝叶斯非参数方法，能够给出预测值及其不确定性估计。
在 CGI 算法中用于残差计算和因果方向判定。

【使用方法】
```python
import numpy as np
from gpr import gpr

# 训练数据
X = np.array([[1], [2], [3], [4], [5]])  # 输入，shape = (n, 1) 或 (n, d)
y = np.array([1.1, 2.3, 2.8, 4.2, 4.9])  # 输出

# 超参数 [log(length_scale), log(signal_noise), log(signal_var)]
logtheta = np.array([np.log(1.0), np.log(0.5), np.log(1.0)])

# 预测
y_pred = gpr(logtheta, 'covSEiso', X, y)
# 返回预测值

# 预测新数据点
x_new = np.array([[1.5], [2.5]])
y_new, ys2 = gpr(logtheta, 'covSEiso', X, y, x_new)
# y_new: 预测均值
# ys2: 预测方差
```

【参数说明】
- logtheta: 对数超参数数组
  - logtheta[0]: length_scale 的对数
  - logtheta[1]: signal_noise 的对数
  - logtheta[2]: signal_var 的对数
- covfunc: 协方差函数类型，如 'covSEiso' (RBF/Squared Exponential)
- x: 训练输入，shape = (n, d)
- y: 训练输出，shape = (n,)
- xstar: 测试输入（可选）

【返回值】
- 训练模式：返回预测值
- 测试模式：返回 (预测均值, 预测方差)

【协方差函数】
- covSEiso: 各向同性 Squared Exponential (RBF) 核
- covSEard: 各向异性 RBF 核
- covNoise: 噪声核
- covMatern: Matern 核

【依赖】
- numpy
- scipy.linalg

================================================================================
"""

import numpy as np
from scipy import linalg
from .covariance import cov_sum


def gpr(logtheta, covfunc, x, y, xstar=None):
    """
    Gaussian Process Regression
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # ========== [GPR TRACE] 输入形状检查 ==========
    # print(f"[GPR_TRACE] gpr called | x.shape={x.shape} | y.shape={y.shape} | xstar={xstar.shape if xstar is not None else 'None'}", flush=True)

    n, D = x.shape

    # 训练模式：计算负对数边缘似然 (Negative Log Marginal Likelihood)
    if xstar is None:
        K = cov_sum(covfunc, logtheta, x)

        # --- 修复 1: 增加数值稳定性 (Jitter) ---
        # 如果矩阵不是正定，Cholesky 会失败。我们在对角线加一点点噪声。
        jitter = 1e-6 * np.eye(n)
        K = K + jitter

        try:
            L = linalg.cholesky(K, lower=True)
        except linalg.LinAlgError:
            # 如果还是失败，尝试更大的 jitter
            jitter = 1e-4 * np.eye(n)
            K = K + jitter
            try:
                L = linalg.cholesky(K, lower=True)
            except linalg.LinAlgError:
                # 实在不行返回一个很大的 loss，让优化器跳过这个参数
                return 1e9

        alpha = linalg.cho_solve((L, True), y)

        nlml = 0.5 * np.dot(y.T, alpha) + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

        # 如果 y 是多维的 (虽然这里应该是向量)，nlml 可能是数组，取标量
        if isinstance(nlml, np.ndarray):
            return nlml.item()
        return nlml

    # 预测模式
    else:
        if xstar.ndim == 1:
            xstar = xstar.reshape(-1, 1)

        # ========== [GPR TRACE] 预测模式 ==========
        # print(f"[GPR_TRACE] Prediction mode | x.shape={x.shape} | xstar.shape={xstar.shape}", flush=True)

        K = cov_sum(covfunc, logtheta, x)

        # 同样的 Jitter 逻辑
        jitter = 1e-6 * np.eye(n)
        K = K + jitter

        try:
            L = linalg.cholesky(K, lower=True)
        except linalg.LinAlgError:
            # 预测时如果失败，通常意味着训练失败了，这里简单处理
            jitter = 1e-4 * np.eye(n)
            K = K + jitter
            L = linalg.cholesky(K, lower=True)

        alpha = linalg.cho_solve((L, True), y)
        # print(f"[GPR_TRACE] alpha.shape={alpha.shape} | y.shape={y.shape}", flush=True)

        # 获取自协方差和交叉协方差 (利用我们之前修好的 covariance.py)
        # 注意：cov_sum 现在返回 (kss, kstar)
        kss, kstar = cov_sum(covfunc, logtheta, x, xstar)
        # print(f"[GPR_TRACE] kss.shape={kss.shape} | kstar.shape={kstar.shape}", flush=True)

        mu = np.dot(kstar.T, alpha)
        # print(f"[GPR_TRACE] mu.shape after dot={mu.shape}", flush=True)

        v = linalg.solve_triangular(L, kstar, lower=True)
        s2 = kss - np.sum(v**2, axis=0)

        # print(f"[GPR_TRACE] Returning mu.shape={mu.shape} | s2.shape={s2.shape}", flush=True)
        return mu, s2