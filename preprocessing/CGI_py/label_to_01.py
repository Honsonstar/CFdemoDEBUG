"""
================================================================================
label_to_01.py - 标签转换工具
================================================================================

【文件作用】
将字符串标签转换为二进制 (0/1) 格式，用于分类任务。
常用于将生存分析中的生存状态（如 "Deceased" vs "Alive"）转换为数值。

【使用方法】
```python
import numpy as np
from label_to_01 import label_to_01, load_data_with_labels

# 方法1: 转换标签数组
labels = np.array(['Alive', 'Deceased', 'Alive', 'Deceased', 'Alive'])
binary_labels = label_to_01(labels, 'Deceased')  # 匹配 'Deceased' 的设为0，其他为1
# 结果: [1, 0, 1, 0, 1]

# 方法2: 从 .mat 文件加载数据和标签
data, labels = load_data_with_labels('survival_data.mat')
binary_labels = label_to_01(labels.flatten(), 'StageIII')
```

【函数说明】
- label_to_01(labels, target_substring): 将匹配目标子串的标签设为0，其他设为1
- load_data_with_labels(mat_file, label_key): 从 .mat 文件加载数据和标签

【应用场景】
- 生存分析：将 "Deceased" -> 0, "Alive" -> 1
- 分期分析：将 "StageIII" -> 0, 其他 -> 1
- 任何二分类标签转换

================================================================================
"""


def label_to_01(labels, target_substring):
    """
    Convert string labels to binary (0/1) format.

    Args:
        labels: array of strings
        target_substring: substring to match for class 1

    Returns:
        Binary labels array (0s and 1s)
    """
    import numpy as np
    labels = np.array(labels).flatten()
    c = np.ones(len(labels), dtype=int)

    for i in range(len(labels)):
        if target_substring in str(labels[i]):
            c[i] = 0

    return c


def load_data_with_labels(mat_file, label_key='labels'):
    """
    Load data and labels from .mat file and create binary labels.

    Args:
        mat_file: path to .mat file
        label_key: key for labels in .mat file

    Returns:
        Tuple of (data, labels)
    """
    from scipy.io import loadmat
    data = loadmat(mat_file)
    labels = data.get(label_key, data.get('label', None))
    if labels is None:
        raise ValueError(f'Could not find labels in {mat_file}')
    return data['data'] if 'data' in data else data['d'], labels
