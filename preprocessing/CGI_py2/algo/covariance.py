"""
================================================================================
covariance.py - 高斯过程协方差函数
================================================================================

【文件作用】
实现 GPML (Gaussian Process for Machine Learning) 工具箱中的多种协方差函数，
用于高斯过程回归。这些函数定义了随机过程的不同性质，如平滑度、噪声等。

【使用方法】
```python
import numpy as np
from covariance import cov_sum, cov_se_iso, cov_se_ard, cov_noise, cov_matern

x = np.random.randn(50, 3)  # 50个样本，3维特征
z = np.random.randn(10, 3)  # 10个测试点

# 组合协方差函数 (Squared Exponential + Noise)
covfunc = ['covSum', ['covSEiso', 'covNoise']]
logtheta = [0.0, 0.0, -2.0]  # [length_scale, variance, noise]

# 训练模式：返回协方差矩阵
K = cov_sum(covfunc, logtheta, x)

# 测试模式：返回自协方差和交叉协方差
K_self, K_cross = cov_sum(covfunc, logtheta, x, z)

# 单独使用
K_se = cov_se_iso(logtheta[:2], x)  # Squared Exponential (Isotropic)
K_noise = cov_noise([logtheta[2]], x)  # Noise
```

【协方差函数说明】
1. cov_se_iso: 各向同性平方指数核 (Squared Exponential, Isotropic)
   - 特点：对所有方向使用相同的长度尺度
   - 参数：2个 [log(length_scale), log(sqrt(variance))]

2. cov_se_ard: 各向异性平方指数核 (Automatic Relevance Determination)
   - 特点：每个维度有独立的长度尺度，用于特征选择
   - 参数：D+1个 (D为维度数)

3. cov_noise: 噪声协方差
   - 特点：添加独立高斯噪声，用于正则化
   - 参数：1个 [log(sqrt(noise_variance))]

4. cov_matern: Matern 协方差核
   - 特点：支持不同的平滑度参数 nu=1,3,5
   - 参数：2个 [log(length_scale), log(sqrt(variance))]

【参数说明】
- covfunc: 协方差函数结构，可以是字符串或嵌套列表
- logtheta: 对数超参数 (优化时使用对数形式更稳定)
- x: 训练数据，shape = (n, d)
- z: 测试数据，shape = (m, d)，可选

【返回值】
- 训练模式：协方差矩阵 K，shape = (n, n)
- 测试模式：(K_self, K_cross)，自协方差和交叉协方差

【应用场景】
- 高斯过程回归中的核函数选择
- 因果发现中的条件独立性检验
- 贝叶斯优化

【依赖】
- dist2: 欧氏距离平方计算

================================================================================
"""

import numpy as np
from .dist2 import dist2


def cov_sum(covfunc, logtheta, x, z=None):
    """
    covSum - Compose a covariance function as the sum of other functions.
    """
    n = x.shape[0]

    # --- 修复核心：解包 covSum 结构 ---
    # 输入通常是 ['covSum', ['covSEiso', 'covNoise']]
    # 我们需要将其解包为 ['covSEiso', 'covNoise']
    if isinstance(covfunc, (list, tuple)) and len(covfunc) > 1 and covfunc[0] == 'covSum':
        covfunc = covfunc[1]
    
    # 如果只是单个字符串，转为列表
    if isinstance(covfunc, str):
        covfunc = [covfunc]

    # Helper to determine number of hyperparameters
    def get_n_hyp(cov_name):
        if 'covSEiso' in cov_name:
            return 2
        elif 'covSEard' in cov_name:
            return x.shape[1] + 1
        elif 'covNoise' in cov_name:
            return 1
        elif 'covMatern' in cov_name:
            return 2
        else:
            return 2

    # Count hyperparameters
    n_hyp_list = []
    for cf in covfunc:
        if isinstance(cf, (list, tuple)):
            cf_name = cf[0]
            # 如果里面还嵌套了 covSum (递归情况)，这里简单处理
            if cf_name == 'covSum': 
                # 这里简化处理，假设没有深层嵌套
                pass
            else:
                n_hyp_list.append(get_n_hyp(cf_name))
        else:
            n_hyp_list.append(get_n_hyp(cf))

    if z is None:
        # --- Training Mode (返回一个矩阵) ---
        A = np.zeros((n, n))
        start_idx = 0
        for i, cf in enumerate(covfunc):
            n_hyp = n_hyp_list[i]
            # 安全切片
            theta_i = logtheta[start_idx:start_idx + n_hyp]
            start_idx += n_hyp

            if isinstance(cf, (list, tuple)):
                cf_name = cf[0]
            else:
                cf_name = cf

            if 'covSEiso' in cf_name:
                A += cov_se_iso(theta_i, x)
            elif 'covSEard' in cf_name:
                A += cov_se_ard(theta_i, x)
            elif 'covNoise' in cf_name:
                A += cov_noise(theta_i, x)
            elif 'covMatern' in cf_name:
                A += cov_matern(theta_i, x, nu=3)
            else:
                A += cov_se_iso(theta_i, x)

        return A
    else:
        # --- Test Mode (返回两个值) ---
        m = z.shape[0]
        A = np.zeros((m,))   # 自协方差
        B = np.zeros((n, m)) # 交叉协方差
        start_idx = 0
        
        for i, cf in enumerate(covfunc):
            n_hyp = n_hyp_list[i]
            theta_i = logtheta[start_idx:start_idx + n_hyp]
            start_idx += n_hyp

            if isinstance(cf, (list, tuple)):
                cf_name = cf[0]
            else:
                cf_name = cf

            if 'covSEiso' in cf_name:
                aa, bb = cov_se_iso(theta_i, x, z)
            elif 'covSEard' in cf_name:
                aa, bb = cov_se_ard(theta_i, x, z)
            elif 'covNoise' in cf_name:
                aa, bb = cov_noise(theta_i, x, z)
            elif 'covMatern' in cf_name:
                aa, bb = cov_matern(theta_i, x, z, nu=3)
            else:
                aa, bb = cov_se_iso(theta_i, x, z)

            A += aa
            B += bb

        return A, B


def cov_se_iso(logtheta, x, z=None):
    """ Squared Exponential (Isotropic) """
    if np.isscalar(logtheta):
        logtheta = np.array([logtheta])

    # --- 防御性裁剪：防止 exp 溢出 ---
    # 允许参数在 e^-30 到 e^30 之间浮动
    logtheta = np.clip(logtheta, -30.0, 30.0)

    length_scale = np.exp(logtheta[0])
    # 确保 logtheta 长度足够，否则说明上层切片错了
    if len(logtheta) < 2:
        raise IndexError(f"cov_se_iso expected 2 params, got {len(logtheta)}. Check cov_sum logic.")

    variance = np.exp(2 * logtheta[1])

    if z is None:
        r2 = dist2(x, x)
        K = variance * np.exp(-0.5 * r2 / (length_scale ** 2))
        return K
    else:
        r2 = dist2(x, z)
        K_cross = variance * np.exp(-0.5 * r2 / (length_scale ** 2))
        K_self = np.full(z.shape[0], variance)
        return K_self, K_cross


def cov_se_ard(logtheta, x, z=None):
    """ Squared Exponential (ARD) """
    # --- 防御性裁剪：防止 exp 溢出 ---
    logtheta = np.clip(logtheta, -30.0, 30.0)

    D = x.shape[1]
    length_scales = np.exp(logtheta[:D])
    variance = np.exp(2 * logtheta[D])

    if z is None:
        x_scaled = x / length_scales
        r2 = dist2(x_scaled, x_scaled)
        K = variance * np.exp(-0.5 * r2)
        return K
    else:
        x_scaled = x / length_scales
        z_scaled = z / length_scales
        r2 = dist2(x_scaled, z_scaled)
        K_cross = variance * np.exp(-0.5 * r2)
        K_self = np.full(z.shape[0], variance)
        return K_self, K_cross


def cov_noise(logtheta, x, z=None):
    """ Noise Covariance """
    # --- 防御性裁剪：防止 exp 溢出 ---
    logtheta = np.clip(logtheta, -30.0, 30.0)

    if len(logtheta) < 1:
        # Default value if something goes wrong, but shouldn't happen
        noise_var = 1e-6
    else:
        noise_var = np.exp(2 * logtheta[0])

    if z is None:
        return np.eye(x.shape[0]) * noise_var
    else:
        m = z.shape[0]
        K_cross = np.zeros((x.shape[0], m))
        K_self = np.full(m, noise_var)
        return K_self, K_cross


def cov_matern(logtheta, x, z=None, nu=3):
    """ Matern Covariance """
    # --- 防御性裁剪：防止 exp 溢出 ---
    logtheta = np.clip(logtheta, -30.0, 30.0)

    length_scale = np.exp(logtheta[0])
    variance = np.exp(2 * logtheta[1])

    def calc_k(dist_sq):
        r = np.sqrt(dist_sq) * np.sqrt(2 * nu) / length_scale
        if nu == 1:
            return variance * (1 + r) * np.exp(-r)
        elif nu == 3:
            return variance * (1 + r + r ** 2 / 3) * np.exp(-r)
        elif nu == 5:
            return variance * (1 + r + r ** 2 * 2 / 5 + r ** 3 / 15) * np.exp(-r)
        else:
            return variance * np.exp(-0.5 * r ** 2)

    if z is None:
        r2 = dist2(x, x)
        return calc_k(r2)
    else:
        r2 = dist2(x, z)
        K_cross = calc_k(r2)
        K_self = np.full(z.shape[0], variance)
        return K_self, K_cross