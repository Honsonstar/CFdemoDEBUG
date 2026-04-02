"""
================================================================================
find_genes_gci.py - 纯线性因果发现极速版
================================================================================

【文件作用】
实现纯线性因果发现算法的核心功能，用于从基因表达数据中
识别因果基因（Causal Genes）。该算法仅基于偏相关检验（PaCoTest）进行线性筛选，
彻底移除了高斯过程回归（GPR）和核条件独立性检验（KCIT），大幅提升运行速度。

【使用方法】
```python
import numpy as np
from CGI_py.find_genes_gci import find_genes_gci, load_data

# 方法1: 从 .mat 文件加载数据
data = load_data('gene_expression.mat')  # 返回 numpy 数组
# 数据格式: 行=样本, 列=基因 (最后一列是目标变量/表型)

# 方法2: 直接传入 numpy 数组
# data = np.loadtxt('data.csv', delimiter=',')

# 运行因果基因发现
results = find_genes_gci(data, alpha=0.05)

# 获取结果
causal_genes = results['found_genes']  # 因果基因索引列表
non_causal = results['non']            # 非因果基因索引列表
potential_parents = results['pa']       # 潜在父节点

print(f"发现 {len(causal_genes)} 个因果基因")
```

【参数说明】
- data: numpy 数组，shape=(n_samples, n_genes+1)，最后一列是目标变量
- alpha: 显著性水平，默认 0.05

【返回值】
- found_genes: 因果基因索引列表
- non: 被排除的非因果基因索引
- pa: 可能的父节点（候选因果基因）

【算法流程】
1. 0阶偏相关检验：检验每个基因与目标变量的独立性
2. 1阶偏相关检验：控制一个其他基因后检验条件独立性

【依赖】
- numpy
- scipy
- pandas (用于CSV加载)

================================================================================
"""

# ========== 第三步：切断 BLAS 线程冲突（防死锁）==========
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
import pandas as pd
import sys

# 处理导入：支持直接运行和模块调用
if __name__ == '__main__':
    # 直接运行时，添加父目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# 确保正确导入您项目中的模块
# 直接运行时避免导入 CGI_py 包（避免触发 __init__.py）
try:
    # 首先尝试相对导入（模块内调用）
    from .algo import paco_test
except ImportError:
    # 直接运行时，直接导入 algo 模块文件
    try:
        # 添加 algo 目录到路径
        algo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algo')
        if algo_dir not in sys.path:
            sys.path.insert(0, algo_dir)
        from paco_test import paco_test
    except ImportError:
        # 最后尝试通过包导入
        from CGI_py.algo import paco_test


# ================================================================================
# ========== 配置区域（更改数据集时需同步修改）=============
# ================================================================================

# 数据集名称（brca, leukemia, blca, hnsc, stad, coadread）
CANCER_TYPE = 'coadread'

# 数据目录
DATA_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/CGI_nested_cv/{CANCER_TYPE}'

# 输出目录
OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/plot_cgi/{CANCER_TYPE}'

# 交叉验证折数
NUM_FOLDS = 5

# 输入文件模板 (自动生成)
# 单数据文件: data_{CANCER_TYPE}.mat
# 交叉验证: train_fold{fold}.mat
USE_CV = True  # 是否使用交叉验证模式

# 输出文件 (自动生成，根据 USE_CV 决定)
FOUND_GENES_FILE = None
NORMALIZED_FILE = None

# ================================================================================

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    数据标准化函数 - 对齐 MATLAB 实现
    MATLAB 的 std() 和 var() 默认使用 ddof=1 (无偏估计)
    """
    normalized = data.copy()
    for i in range(data.shape[1]):
        col = data[:, i]
        # 关键修正: 使用 ddof=1 对齐 MATLAB 的无偏估计
        std = np.std(col, ddof=1)
        if std > 1e-10:
            normalized[:, i] = (col - np.mean(col)) / std
        else:
            normalized[:, i] = col - np.mean(col)
    return normalized


def print_matlab_style_array(name: str, arr: list, cols_per_row: int = 7):
    """模拟 MATLAB 命令行打印一维数组的折行效果"""
    print(f"\n{name} =\n")
    n = len(arr)
    if n == 0:
        print("     []\n")
        return

    for i in range(0, n, cols_per_row):
        end_idx = min(i + cols_per_row, n)
        # 打印列号提示
        if i == end_idx - 1:
            print(f"  列 {i+1}\n")
        else:
            print(f"  列 {i+1} 至 {end_idx}\n")

        # 打印数值，右对齐，占12个字符宽度以对齐 MATLAB 格式
        row_str = "".join([f"{x:12d}" for x in arr[i:end_idx]])
        print(f"{row_str}\n")


def find_genes_gci(data: np.ndarray, alpha: float = 0.05) -> dict:
    # 默认最后一列是目标变量
    n = data.shape[1] - 1
    x = data[:, -1].copy()
    data = data[:, :n]

    # ========== 第一步：极小方差特征过滤 (与 MATLAB 一致) ==========
    # MATLAB: col_var = var(data, [], 1); bad_cols = col_var < 1e-10;
    # 使用 ddof=1 对齐 MATLAB 的无偏估计
    col_var = np.var(data, axis=0, ddof=1)
    bad_cols = col_var < 1e-10
    bad_count = np.sum(bad_cols)
    if bad_count > 0:
        print(f'  - 移除 {bad_count} 列极小方差列')
        data = data[:, ~bad_cols]
        n = data.shape[1]
        print(f'  数据形状更新: {data.shape}')

    # 标准化数据和目标变量
    data = normalize_data(data)

    # --- 关键修正: 对齐 MATLAB 的目标变量标准化 ---
    # MATLAB: x_std = std(x); if x_std > 1e-10, x = (x - mean(x)) / x_std
    x_std = np.std(x, ddof=1)
    if x_std > 1e-10:
        x = (x - np.mean(x)) / x_std

    # --- DEBUG: Check Data Shape ---
    print(f"DEBUG: Data Shape: {data.shape} (Rows=Samples, Cols=Genes)")
    print(f"DEBUG: Target(x) Shape: {x.shape}")
    print(f"DEBUG: First 5 values of x: {x[:5]}")
    print(f"DEBUG: alpha: {alpha}")

    # ========== 关键修正：确保向量为 2D 列向量 (与 MATLAB 一致) ==========
    # MATLAB 中 x 和 data(:,i) 是 (n, 1) 形状，Python 需要手动确保
    x = x.reshape(-1, 1)  # 确保 x 是 (n, 1) 列向量

    # 基础进度打印
    print(f"Data loaded: data.shape={data.shape}, x.shape={x.shape}")

    non = []

    # 0-order CI tests
    print('--------------- 0-order CI tests')
    for i in range(n):
        data_i = data[:, i].reshape(-1, 1)

        # 使用偏相关检验判断独立性
        ind1, _, _ = paco_test(x, data_i, np.array([]), alpha)

        if i % 50 == 0:
            print(f"  0-order: processing gene {i}/{n}")

        # 如果偏相关检验判定为独立，则排除该基因
        if ind1 == True:
            non.append(i)

    # --- 新增打印 ---
    print(f"DEBUG: Total genes: {n}")
    print(f"DEBUG: Genes removed in 0-order: {len(non)}")
    print(f"DEBUG: Genes remaining: {n - len(non)}")

    # 1-order CI tests
    print('--------------- 1-order CI tests')
    idx1 = [i for i in range(n) if i not in non]
    len1 = len(idx1)
    print(f'  Testing {len1} genes...')

    # 核心：使用动态集合追踪存活基因
    active_genes = set(idx1)

    for j in range(len1):
        idx1_j = idx1[j]
        # 如果当前基因已经被之前的测试杀死了，直接跳过
        if idx1_j not in active_genes:
            continue

        if j % 50 == 0:
            print(f"  1-order: processing {j}/{len1} (Alive: {len(active_genes)})", flush=True)

        y = data[:, idx1_j].reshape(-1, 1)

        for k in range(len1):
            idx1_k = idx1[k]
            # 只把"依然存活"的基因当作条件变量 Z
            if j == k or idx1_k not in active_genes:
                continue

            z = data[:, idx1_k].reshape(-1, 1)

            # 使用偏相关检验判断条件独立性
            ind_paco, _, _ = paco_test(x, y, z, alpha)

            # 如果偏相关检验判定为独立，则排除该基因
            if ind_paco:
                active_genes.remove(idx1_j)
                non.append(idx1_j)
                break  # 跳出 k 循环，当前基因已死，测试下一个候选基因

    pa = [i for i in range(n) if i not in non]
    print(f"  Genes remaining after 1st order: {len(pa)}")

    # 经过 1-order 筛选后，pa 即为候选因果基因
    # found_genes 就是 pa（因为只进行线性筛选）
    found_genes = sorted(list(set(pa)))

    # 使用 MATLAB 风格的数组打印
    print_matlab_style_array('found_Genes', found_genes)

    return {
        'non': non,
        'pa': pa,
        'found_genes': found_genes
    }


def load_data(file_path: str) -> np.ndarray:
    """
    Load data from .mat or .csv file.
    """
    # 获取文件后缀名
    _, ext = os.path.splitext(file_path)

    # 情况 1: 读取 .mat 文件 (保持原有逻辑)
    if ext == '.mat':
        data = loadmat(file_path)
        # 尝试查找常用的变量名
        for key in ['d', 'data', 'D', 'normalized']:
            if key in data:
                return data[key]
        raise ValueError(f'Could not find data variable in {file_path}')

    # 情况 2: 读取 .csv 文件 (新增逻辑)
    elif ext == '.csv':
        try:
            # header=None 假设 CSV 没有表头，全是数据
            # 如果你的 CSV 第一行是列名，请改为 header=0
            df = pd.read_csv(file_path, header=0)

            # 转换为 numpy 数组
            return df.values
        except Exception as e:
            raise ValueError(f'Failed to read CSV file {file_path}: {e}')

    # 情况 3: 不支持的格式
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Please use .mat or .csv")


# ================================================================================
# ========== 主程序入口 (直接运行此文件) ==============
# ================================================================================
if __name__ == '__main__':
    import scipy.io as sio

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if USE_CV:
        # ==================== 交叉验证模式 ====================
        print('='*60)
        print(f'  CGI 因果基因发现 - {CANCER_TYPE} ({NUM_FOLDS}折交叉验证)')
        print('='*60)

        all_found_genes = []
        total_time = 0

        for fold in range(NUM_FOLDS):
            print('\n')
            print('-'*50)
            print(f'  Fold {fold}/{NUM_FOLDS}')
            print('-'*50)

            # 输入文件 (MATLAB 索引从0开始)
            mat_file = os.path.join(DATA_DIR, f'train_fold{fold}.mat')

            # 输出文件
            found_genes_file = os.path.join(OUTPUT_DIR, f'{CANCER_TYPE}_found_Genes_fold{fold}.mat')
            normalized_file = os.path.join(OUTPUT_DIR, f'{CANCER_TYPE}_normalized_fold{fold}.mat')

            print(f'加载数据: {mat_file}')
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f'数据文件不存在: {mat_file}')

            data = load_data(mat_file)
            print(f'数据形状: {data.shape}')

            # 运行因果基因发现
            import time
            start_time = time.time()
            results = find_genes_gci(data, alpha=0.05)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            found_genes = results['found_genes']
            all_found_genes.append(found_genes)

            # 保存结果
            sio.savemat(found_genes_file, {'found_Genes': found_genes})

            # 标准化数据
            raw_data = load_data(mat_file)
            n_original = raw_data.shape[1] - 1
            # 修复索引: Python 使用 0-based 索引
            labels = raw_data[:, -1]
            data_for_normalized = raw_data[:, :n_original]
            data_for_normalized = normalize_data(data_for_normalized)
            data_with_label = np.column_stack([data_for_normalized, labels])
            sio.savemat(normalized_file, {'data_with_label': data_with_label})

            print(f'  Fold {fold} 结果: {len(found_genes)} 个因果基因')
            print(f'  耗时: {elapsed_time:.2f} 秒')

        # 汇总结果
        print('\n')
        print('='*60)
        print('  各折因果基因汇总')
        print('='*60)

        all_genes_flat = []
        for fold in range(NUM_FOLDS):
            print(f'  Fold {fold}: {len(all_found_genes[fold])} 个因果基因')
            all_genes_flat.extend(all_found_genes[fold])

        # 统计各折共同发现的基因
        from collections import Counter
        gene_counts = Counter(all_genes_flat)
        common_genes = [gene for gene, count in gene_counts.items() if count == NUM_FOLDS]
        gene_in_half = [gene for gene, count in gene_counts.items() if count >= (NUM_FOLDS + 1) // 2]

        print(f'\n  各折共同发现的基因数: {len(common_genes)}')
        print(f'  在至少{NUM_FOLDS//2 + 1}折中出现的基因数: {len(gene_in_half)}')

        if common_genes:
            print(f'\n  各折共同发现的基因索引:')
            print(f'  {sorted(common_genes)}')

        print(f'\n  各折总耗时: {total_time:.2f} 秒')
        print(f'  平均每折耗时: {total_time/NUM_FOLDS:.2f} 秒')

        # 保存汇总结果
        # 注意：all_found_genes 是长度不同的列表组成的列表，需用 dtype=object 保存
        summary_file = os.path.join(OUTPUT_DIR, f'{CANCER_TYPE}_all_folds_summary.mat')
        sio.savemat(summary_file, {
            'all_found_genes': np.array(all_found_genes, dtype=object),
            'common_genes': np.array(common_genes, dtype=object) if common_genes else np.array([], dtype=object),
            'gene_in_at_least_half': np.array(gene_in_half, dtype=object) if gene_in_half else np.array([], dtype=object)
        })
        print(f'\n已保存汇总结果: {summary_file}')

    else:
        # ==================== 单数据模式 ====================
        print('='*60)
        print(f'  CGI 因果基因发现 - {CANCER_TYPE}')
        print('='*60)

        # 输入文件
        mat_file = os.path.join(DATA_DIR, f'data_{CANCER_TYPE}.mat')

        # 输出文件
        found_genes_file = os.path.join(OUTPUT_DIR, f'{CANCER_TYPE}_found_Genes.mat')
        normalized_file = os.path.join(OUTPUT_DIR, f'{CANCER_TYPE}_normalized.mat')

        # 加载数据
        print(f'\n加载数据: {mat_file}')
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f'数据文件不存在: {mat_file}')

        data = load_data(mat_file)
        print(f'数据形状: {data.shape} (样本数 × 基因数+1)')

        # 运行因果基因发现
        print('\n开始因果基因筛选...')
        results = find_genes_gci(data, alpha=0.05)

        # 获取结果
        found_genes = results['found_genes']

        # 标准化数据（与交叉验证模式保持一致）
        n_original = data.shape[1] - 1
        labels = data[:, -1]
        data_for_normalized = data[:, :n_original]
        data_for_normalized = normalize_data(data_for_normalized)

        # ========== 保存结果 ==========
        # 保存发现的基因索引
        sio.savemat(found_genes_file, {'found_Genes': found_genes})

        # 加载原始数据获取标签列
        raw_data = load_data(mat_file)
        n_original = raw_data.shape[1] - 1
        labels = raw_data[:, -1]

        # 使用标准化后的 data (不含标签)，再拼接标签列
        data_with_label = np.column_stack([data_for_normalized, labels])

        # 保存预处理后的数据
        sio.savemat(normalized_file, {'data_with_label': data_with_label})

        # 对齐 MATLAB 的打印输出格式
        print('====================')
        print('  结果已保存到 plot_cgi 目录')
        print('====================')
        print(f'  found_Genes_{CANCER_TYPE}.mat: {len(found_genes)} 个因果基因')
        print(f'  normalized_{CANCER_TYPE}.mat: {data_with_label.shape[0]} 样本 × {data_with_label.shape[1]} 列')
        print('====================')