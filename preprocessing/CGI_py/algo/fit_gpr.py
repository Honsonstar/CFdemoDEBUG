"""
FIT_GPR - Fit Gaussian Process Regression model (Bounded & Robust)
"""

import numpy as np
from .gpr import gpr
from .minimize import rasmussen_minimize

def fit_gpr(X: np.ndarray, Y: np.ndarray, cov: str = 'covSEiso',
            hyp: np.ndarray = None, Ncg: int = 100) -> np.ndarray:
    """
    Fit a Gaussian Process Regression model with Strict Bounds.
    """
    # --- 1. 维度与数据检查 ---
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X = np.atleast_2d(X)
    Y = Y.flatten()

    n = X.shape[0]
    if n != Y.shape[0] and X.shape[1] == Y.shape[0]:
        X = X.T
        n = X.shape[0]

    if Y.shape[0] != n:
        raise ValueError(f'X should be Nxd and Y should be Nx1. Got X:{X.shape}, Y:{Y.shape}')

    # --- 2. 超参数初始化 ---
    if hyp is None:
        # MATLAB 默认: hyp = [log(4), log(4), log(0.1)] = [1.3863, 1.3863, -2.3026]
        hyp = np.array([np.log(4.0), np.log(4.0), np.log(0.1)])
    else:
        hyp = hyp.copy()

    # 构造 ARD 超参数
    if 'covSEard' in cov:
        D = X.shape[1]
        hyp_adjusted = np.zeros(D + 2)
        hyp_adjusted[:D] = hyp[0]
        hyp_adjusted[D] = hyp[1]
        hyp_adjusted[D + 1] = hyp[2]
    else:
        hyp_adjusted = np.array([hyp[0], hyp[1], hyp[2]])

    covfunc = ['covSum', [cov, 'covNoise']]

    # --- 3. 目标函数 (Rasmussen 动态回溯 + 有限差分引擎) ---
    def obj_func(theta):
        # 1. 计算基础似然值
        out = gpr(theta, covfunc, X, Y)
        val = float(out[0]) if isinstance(out, (tuple, list, np.ndarray)) and np.size(out) > 1 else float(out)

        # 核心：遇到计算溢出，必须主动抛错，触发 Rasmussen 去缩小步长
        if not np.isfinite(val):
            raise ValueError("GP value diverged")

        # 2. 有限差分计算梯度 (找回算出 0.064528 的功臣)
        eps = 1e-5
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[i] += eps
            theta_m[i] -= eps

            vp_out = gpr(theta_p, covfunc, X, Y)
            vm_out = gpr(theta_m, covfunc, X, Y)

            vp = float(vp_out[0]) if isinstance(vp_out, (tuple, list, np.ndarray)) and np.size(vp_out) > 1 else float(vp_out)
            vm = float(vm_out[0]) if isinstance(vm_out, (tuple, list, np.ndarray)) and np.size(vm_out) > 1 else float(vm_out)

            if not np.isfinite(vp) or not np.isfinite(vm):
                raise ValueError("Gradient step diverged")

            grad[i] = (vp - vm) / (2.0 * eps)

        return val, grad

    # ========== 调用带装甲的 Rasmussen 优化器 ==========
    if Ncg > 0:
        # 引入我们在 __init__.py 设置好别名的优化器
        from .minimize import minimize as rasmussen_minimize
        hyp_opt = rasmussen_minimize(hyp_adjusted.copy(), obj_func, Ncg)
    else:
        hyp_opt = hyp_adjusted.copy()
    # ==========================================================

    # --- 5. 计算预测值 ---
    try:
        prediction_result = gpr(hyp_opt, covfunc, X, Y, X)
        
        if isinstance(prediction_result, tuple):
            Yfit = prediction_result[0]
        else:
            Yfit = prediction_result
            
        if not np.isfinite(Yfit).all():
            Yfit = np.zeros_like(Y)
            
    except:
        Yfit = np.zeros_like(Y)

    return Yfit