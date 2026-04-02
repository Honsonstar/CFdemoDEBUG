"""
Python implementation of the algorithms
"""

# 1. 基础算法模块
from .dist2 import dist2
from .eigdec import eigdec
from .pdinv import pdinv
from .stack import stack
from .solve_chol import solve_chol
from .medbw import medbw
from .kernel import kernel, kernel_matrix  # <--- 补上了 kernel_matrix

# 2. 优化与协方差模块
from .minimize import minimize  # <--- 补上了 minimize
from .covariance import cov_sum, cov_se_iso, cov_se_ard, cov_noise, cov_matern # <--- 补上了协方差函数

# 3. 核心功能模块
from .gpr import gpr
from .fit_gpr import fit_gpr
from .paco_test import paco_test
from .uind_test import uind_test
from .cind_test import cind_test, cind_test_new_with_gp
from .kcit import kcit

# 4. 兼容性处理
# 外层试图导入 gpr_multi，但我们在 gpr.py 中移除了它（因为有 bug 且未被使用）。
# 为了防止报错，我们在这里定义一个占位符。
gpr_multi = None 

__all__ = [
    'kernel', 'kernel_matrix', 
    'dist2', 'eigdec', 'pdinv', 'stack', 'solve_chol', 'medbw',
    'minimize', 
    'cov_sum', 'cov_se_iso', 'cov_se_ard', 'cov_noise', 'cov_matern',
    'gpr', 'gpr_multi', 'fit_gpr',
    'paco_test', 'uind_test', 'cind_test', 'cind_test_new_with_gp', 'kcit'
]