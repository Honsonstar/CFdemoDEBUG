"""
================================================================================
normalized.py - 数据归一化工具
================================================================================

【文件作用】
提供 Z-score 标准化功能，将数据转换为均值为0、标准差为1的分布。
这是 CGI 算法预处理的关键步骤，确保不同基因具有可比性。

【使用方法】
```python
import numpy as np
from normalized import normalize, load_and_normalize

# 方法1: 直接归一化数组
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
normalized_data = normalize(data)

# 方法2: 从 .mat 文件加载并归一化
normalized = load_and_normalize('gene_expression.mat')
# 自动查找 .mat 文件中的数据变量 (d, data, D, normalized)

# 归一化公式: (x - mean) / std
# 如果 std = 0，则只减去均值
```

【函数说明】
- normalize(data): 对每一列进行 Z-score 标准化
- load_and_normalize(mat_file): 加载 .mat 文件并自动归一化

【注意事项】
- 按列归一化（每个基因/特征独立归一化）
- 处理标准差为0的列（只减均值，不除以0）

================================================================================
"""


def normalize(data):
    """
    Normalize data by z-score standardization.

    Args:
        data: numpy array of shape (n_samples, n_features)

    Returns:
        Normalized data array
    """
    import numpy as np
    normalized = np.zeros_like(data)
    for i in range(data.shape[1]):
        col = data[:, i]
        if np.std(col) > 0:
            normalized[:, i] = (col - np.mean(col)) / np.std(col)
        else:
            normalized[:, i] = col - np.mean(col)
    return normalized


def load_and_normalize(mat_file):
    """
    Load data from .mat file and normalize.

    Args:
        mat_file: path to .mat file

    Returns:
        Normalized data array
    """
    from scipy.io import loadmat
    data = loadmat(mat_file)
    # Try common variable names
    for key in ['d', 'data', 'D', 'normalized']:
        if key in data:
            return normalize(data[key])
    raise ValueError(f'Could not find data variable in {mat_file}')
