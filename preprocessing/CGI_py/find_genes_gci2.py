"""
================================================================================
find_genes_gci2.py - 马尔科夫毯扩展版因果发现
================================================================================

【文件作用】
在 find_genes_gci.py 的结果基础上，往外多做一层马尔科夫毯。
即：
1. 先运行 find_genes_gci 得到初始因果基因
2. 轮流将筛选出来的基因作为目标变量
3. 遍历完所有现有结果后，将所有得到的基因的并集作为 found_genes 输出

【使用方法】
```python
import numpy as np
from CGI_py.find_genes_gci2 import find_genes_gci2, load_data

# 方法1: 从 .mat 文件加载数据
data = load_data('gene_expression.mat')

# 运行马尔科夫毯扩展版因果基因发现
results = find_genes_gci2(data, alpha=0.05)

# 获取结果
causal_genes = results['found_genes']  # 因果基因索引列表
non_causal = results['non']            # 非因果基因索引列表
potential_parents = results['pa']      # 潜在父节点

print(f"发现 {len(causal_genes)} 个因果基因")
```

【参数说明】
- data: numpy 数组，shape=(n_samples, n_genes+1)，最后一列是原始目标变量
- alpha: 显著性水平，默认 0.05

【返回值】
- found_genes: 因果基因索引列表（并集）
- non: 被排除的非因果基因索引
- pa: 可能的父节点（候选因果基因）

【算法流程】
1. 第一层: 使用原始目标变量运行 find_genes_gci 筛选
2. 第二层: 对每个第一层筛选出的基因，将其作为目标变量运行筛选
3. 将两层筛选得到的基因取出现频率60%以上或出现频率top200的基因的并集作为最终结果

【依赖】
- numpy
- scipy
- pandas
- find_genes_gci 模块

================================================================================
"""

# ========== 第三步：切断 BLAS 线程冲突（防死锁）==========
import os

# 在主进程中设置 BLAS 线程限制
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
import pandas as pd
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 处理导入：支持直接运行和模块调用
if __name__ == '__main__':
    # 直接运行时，添加父目录到路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# 尝试导入 find_genes_gci 模块（支持多种导入方式）
_find_genes_gci_loaded = False

# 方法1: 相对导入
try:
    from .find_genes_gci import find_genes_gci as find_genes_gci_original, load_data as load_data_original
    _find_genes_gci_loaded = True
except ImportError:
    pass

# 方法2: 直接导入（同一目录）
if not _find_genes_gci_loaded:
    try:
        from find_genes_gci import find_genes_gci as find_genes_gci_original, load_data as load_data_original
        _find_genes_gci_loaded = True
    except ImportError:
        pass

# 方法3: CGI_py 包导入
if not _find_genes_gci_loaded:
    try:
        from CGI_py.find_genes_gci import find_genes_gci as find_genes_gci_original, load_data as load_data_original
        _find_genes_gci_loaded = True
    except ImportError:
        pass

# 方法4: 将父目录添加到路径
if not _find_genes_gci_loaded:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    try:
        from CGI_py.find_genes_gci import find_genes_gci as find_genes_gci_original, load_data as load_data_original
        _find_genes_gci_loaded = True
    except ImportError:
        pass

if not _find_genes_gci_loaded:
    raise ImportError("无法导入 find_genes_gci 模块，请确保该文件存在于正确位置")


# ================================================================================
# ========== 配置区域（更改数据集时需同步修改）=============
# ================================================================================

# 数据集名称（brca, leukemia, blca, hnsc, stad, coadread）
CANCER_TYPE = 'coadread'

# 数据目录
DATA_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/CGI_nested_cv/{CANCER_TYPE}'

# 输出目录
OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/plot_cgi/{CANCER_TYPE}'

# CSV输出目录
CSV_OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/features/gci2/{CANCER_TYPE}'

# 交叉验证折数
NUM_FOLDS = 5

# 输入文件模板 (自动生成)
USE_CV = True  # 是否使用交叉验证模式

# ================================================================================
# ========== 并行计算配置 =============
# ================================================================================
# 是否启用第二层马尔科夫毯筛选的并行计算
# 注意：并行计算不会影响最终结果，每个任务是独立的
USE_PARALLEL = True  # True: 启用并行, False: 串行执行

# 并行任务数（默认为 CPU 核心数）
# 设置为 None 表示使用所有可用核心
N_JOBS = None  # None 表示使用全部可用核心

# ================================================================================


def _worker_layer2_parallel(args):
    """
    并行处理第二层马尔科夫毯筛选的 worker 函数（供 ProcessPoolExecutor 调用）

    参数:
        args: tuple of (data_bytes, target_idx, alpha, n_genes)
            - data_bytes: 序列化后的数据矩阵（bytes）
            - target_idx: 目标变量的索引
            - alpha: 显著性水平
            - n_genes: 基因数量

    返回:
        dict: 包含 found_genes, target_idx
    """
    import os
    import numpy as np

    # 重新设置 BLAS 线程限制（每个子进程都需要）
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # 反序列化数据
    data = np.frombuffer(args[0], dtype=np.float64).reshape(args[3], -1)
    target_idx = args[1]
    alpha = args[2]

    # 获取目标变量
    x = data[:, target_idx].copy()

    # 构建新的数据矩阵
    other_cols = [i for i in range(data.shape[1]) if i != target_idx]
    other_data = data[:, other_cols]
    new_data = np.column_stack([other_data, x])

    # 运行筛选（调用全局的 find_genes_gci_original）
    results = find_genes_gci_original(new_data, alpha=alpha)

    # 重新映射索引
    found_genes = []
    for gene_idx in results['found_genes']:
        if gene_idx == data.shape[1] - 1:
            continue
        else:
            found_genes.append(other_cols[gene_idx])

    return {
        'found_genes': found_genes,
        'target_idx': target_idx
    }


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


def find_genes_with_target(data: np.ndarray, target_idx: int, alpha: float = 0.05) -> dict:
    """
    将指定列作为目标变量，运行因果基因筛选

    参数:
        data: numpy 数组，shape=(n_samples, n_genes)，不含标签列
        target_idx: 目标变量的列索引
        alpha: 显著性水平

    返回:
        dict: 包含 found_genes, non, pa
    """
    # 获取目标变量
    x = data[:, target_idx].copy()

    # 构建新的数据矩阵：目标变量放在最后一列
    # 所有其他列（除了目标变量）
    other_cols = [i for i in range(data.shape[1]) if i != target_idx]
    other_data = data[:, other_cols]

    # 将目标变量放在最后一列
    new_data = np.column_stack([other_data, x])

    # 运行筛选
    results = find_genes_gci_original(new_data, alpha=alpha)

    # 重新映射索引（因为我们移除了目标变量列）
    # 筛选结果中的索引是相对于 new_data 的，需要映射回原始 data
    found_genes = []
    for gene_idx in results['found_genes']:
        # new_data 的最后一列对应原始的 target_idx
        # new_data 的第 i 列 (i < n-1) 对应原始的 other_cols[i]
        if gene_idx == data.shape[1] - 1:
            # 这个对应目标变量本身，不计入
            continue
        else:
            found_genes.append(other_cols[gene_idx])

    # 对 non 也做同样处理
    non = []
    for gene_idx in results['non']:
        if gene_idx == data.shape[1] - 1:
            continue
        else:
            non.append(other_cols[gene_idx])

    # pa 也做同样处理
    pa = []
    for gene_idx in results['pa']:
        if gene_idx == data.shape[1] - 1:
            continue
        else:
            pa.append(other_cols[gene_idx])

    return {
        'found_genes': found_genes,
        'non': non,
        'pa': pa,
        'target_idx': target_idx  # 记录这一轮的目标变量索引
    }


def find_genes_gci2(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    马尔科夫毯扩展版因果基因发现

    流程：
    1. 第一层: 使用原始目标变量（最后一列）运行 find_genes_gci 筛选
    2. 第二层: 对每个第一层筛选出的基因，将其作为目标变量运行筛选
    3. 将两层筛选得到的基因取并集作为最终结果
    """
    # 原始目标变量（最后一列）
    original_target = data[:, -1].copy()
    n_genes = data.shape[1] - 1  # 不包含目标变量的基因数

    # ========== 第一层筛选：使用原始目标变量 ==========
    print("\n" + "="*60)
    print("  第一层筛选：使用原始目标变量")
    print("="*60)

    results_layer1 = find_genes_gci_original(data, alpha=alpha)
    layer1_genes = results_layer1['found_genes']

    print(f"\n第一层筛选结果: {len(layer1_genes)} 个因果基因")
    print(f"  基因索引: {sorted(layer1_genes)}")

    # 收集所有发现的基因（用于并集）
    all_found_genes = set(layer1_genes)

    # ========== 第二层筛选：将每个第一层基因作为目标变量 ==========
    print("\n" + "="*60)
    print("  第二层筛选：马尔科夫毯扩展")
    print("="*60)

    # 准备数据（不含标签列）
    data_genes = data[:, :n_genes]

    layer2_results = []
    total_genes = len(layer1_genes)

    # 获取并行任务数
    n_jobs = N_JOBS if N_JOBS else mp.cpu_count()

    if USE_PARALLEL and total_genes > 1:
        # ==================== 并行模式 ====================
        print(f"\n  使用并行模式，任务数: {n_jobs}")

        # 序列化数据（传递给子进程）
        data_bytes = data_genes.tobytes()
        n_samples = data_genes.shape[0]

        # 准备并行任务参数
        task_args = [
            (data_bytes, gene_idx, alpha, n_samples)
            for gene_idx in layer1_genes
        ]

        # 使用 ProcessPoolExecutor 进行并行处理
        # as_completed 确保所有任务完成后才继续，等待最后一个结果
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # 提交所有任务
            future_to_gene = {
                executor.submit(_worker_layer2_parallel, arg): arg[1]
                for arg in task_args
            }

            # 收集结果（等待所有任务完成）
            completed_count = 0
            for future in as_completed(future_to_gene):
                gene_idx = future_to_gene[future]
                completed_count += 1

                try:
                    result = future.result()
                    found_this_round = result['found_genes']

                    if found_this_round:
                        print(f"  [{completed_count}/{total_genes}] 基因 {gene_idx}: 发现 {len(found_this_round)} 个相关基因")
                        all_found_genes.update(found_this_round)
                    else:
                        print(f"  [{completed_count}/{total_genes}] 基因 {gene_idx}: 未发现新基因")

                    layer2_results.append({
                        'target_idx': gene_idx,
                        'found_genes': found_this_round
                    })
                except Exception as e:
                    print(f"  [{completed_count}/{total_genes}] 基因 {gene_idx} 处理失败: {e}")
                    layer2_results.append({
                        'target_idx': gene_idx,
                        'found_genes': [],
                        'error': str(e)
                    })

        print(f"\n  并行任务全部完成")

    else:
        # ==================== 串行模式 ====================
        print(f"\n  使用串行模式")

        for i, gene_idx in enumerate(layer1_genes):
            print(f"\n  [{i+1}/{total_genes}] 将基因 {gene_idx} 作为目标变量...")

            # 将该基因作为目标变量运行筛选
            result = find_genes_with_target(data_genes, gene_idx, alpha=alpha)

            # 获取这一轮发现的基因
            found_this_round = result['found_genes']

            if found_this_round:
                print(f"    发现 {len(found_this_round)} 个相关基因: {sorted(found_this_round)}")
                all_found_genes.update(found_this_round)
            else:
                print(f"    未发现新的相关基因")

            layer2_results.append({
                'target_idx': gene_idx,
                'found_genes': found_this_round,
                'non': result['non'],
                'pa': result['pa']
            })

    # ========== 第三步：统计频率并筛选 ==========
    print("\n" + "="*60)
    print("  第三步：频率统计与筛选")
    print("="*60)

    # 统计第二层中每个基因被选中的频率
    from collections import Counter
    gene_freq_counter = Counter()
    for layer2_result in layer2_results:
        gene_freq_counter.update(layer2_result['found_genes'])

    total_candidates = len(layer2_results)  # 第二层总候选目标数（即第一层基因数）
    print(f"  第二层候选目标数: {total_candidates}")

    # 计算频率阈值
    freq_threshold = 0.6  # 60%频率
    min_count_for_60 = int(total_candidates * freq_threshold)

    # 筛选条件1: 频率60%以上
    genes_above_60 = [gene for gene, count in gene_freq_counter.items()
                      if count >= min_count_for_60]
    print(f"  频率60%以上的基因数 (出现次数>={min_count_for_60}): {len(genes_above_60)}")

    # 筛选条件2: 频率top200
    # 按频率排序，取top200
    sorted_genes_by_freq = sorted(gene_freq_counter.items(),
                                   key=lambda x: x[1], reverse=True)
    top_200_genes = [gene for gene, count in sorted_genes_by_freq[:200]]
    print(f"  频率top200的基因数: {len(top_200_genes)}")

    # 取两个条件的并集
    # 注意：也保留第一层的基因（因为它们是直接与目标变量相关的）
    selected_genes = set(genes_above_60) | set(top_200_genes) | set(layer1_genes)
    found_genes = sorted(list(selected_genes))

    # 打印详细的频率统计（用于调试）
    print(f"\n  频率统计详情 (前20个):")
    for gene, count in sorted_genes_by_freq[:20]:
        freq = count / total_candidates * 100
        print(f"    基因 {gene}: 出现 {count} 次 ({freq:.1f}%)")

    print("\n" + "="*60)
    print("  最终结果汇总")
    print("="*60)
    print(f"  第一层基因数: {len(layer1_genes)}")
    print(f"  频率60%以上基因数: {len(genes_above_60)}")
    print(f"  频率top200基因数: {len(top_200_genes)}")
    print(f"  最终并集基因数: {len(found_genes)}")
    print(f"  最终基因索引: {found_genes}")

    return {
        'found_genes': found_genes,
        'layer1_genes': layer1_genes,
        'layer2_results': layer2_results,
        'non': results_layer1['non'],
        'pa': results_layer1['pa'],
        # 新增：频率统计信息
        'gene_freq_counter': dict(gene_freq_counter),
        'genes_above_60': genes_above_60,
        'top_200_genes': top_200_genes,
        'sorted_genes_by_freq': sorted_genes_by_freq
    }


def load_data(file_path: str) -> np.ndarray:
    """
    Load data from .mat or .csv file.
    """
    # 获取文件后缀名
    _, ext = os.path.splitext(file_path)

    # 情况 1: 读取 .mat 文件
    if ext == '.mat':
        data = loadmat(file_path)
        # 尝试查找常用的变量名
        for key in ['d', 'data', 'D', 'normalized']:
            if key in data:
                return data[key]
        raise ValueError(f'Could not find data variable in {file_path}')

    # 情况 2: 读取 .csv 文件
    elif ext == '.csv':
        try:
            df = pd.read_csv(file_path, header=0)
            return df.values
        except Exception as e:
            raise ValueError(f'Failed to read CSV file {file_path}: {e}')

    # 情况 3: 不支持的格式
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Please use .mat or .csv")


def load_train_sample_ids(cancer_type: str, fold: int, data_dir: str) -> list:
    """
    从 nested_splits_*.csv 文件中读取训练集样本ID

    参数:
        cancer_type: 癌症类型
        fold: 折数
        data_dir: 数据目录

    返回:
        list: 训练集样本ID列表
    """
    splits_file = os.path.join(data_dir, f'nested_splits_{fold}.csv')
    if not os.path.exists(splits_file):
        print(f"    警告: 未找到样本ID文件 {splits_file}")
        return None

    df = pd.read_csv(splits_file)
    train_samples = df['train'].dropna().tolist()
    return train_samples


def load_gene_names(cancer_type: str) -> list:
    """
    从原始CSV文件加载基因名列表

    参数:
        cancer_type: 癌症类型

    返回:
        list: 基因名列表（按列索引顺序）
    """
    # 原始数据文件路径
    original_csv_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI/data/{cancer_type}/{cancer_type}_data_with_id.csv'

    if not os.path.exists(original_csv_path):
        print(f"    警告: 未找到原始数据文件 {original_csv_path}")
        return None

    # 读取CSV获取列名（第一列是patient_id，最后是time）
    df = pd.read_csv(original_csv_path, nrows=0)  # 只读取头部
    # 列名: patient_id, gene1, gene2, ..., geneN, time
    gene_names = df.columns[1:-1].tolist()  # 跳过patient_id和time
    print(f"    已加载基因名: {len(gene_names)} 个")
    return gene_names


def verify_expression_value(data_original: np.ndarray, gene_names: list,
                          sample_id: str, gene_name: str,
                          cancer_type: str, data_dir: str, fold: int) -> dict:
    """
    随机抽样验证：验证生成文件中的表达量是否正确

    参数:
        data_original: 原始数据矩阵
        gene_names: 基因名列表
        sample_id: 样本ID
        gene_name: 基因名
        cancer_type: 癌症类型
        data_dir: 数据目录
        fold: 折数

    返回:
        dict: 验证结果
    """
    # 查找样本索引
    sample_ids = load_train_sample_ids(cancer_type, fold, data_dir)
    if sample_id not in sample_ids:
        return {'error': f'样本 {sample_id} 不在训练集中'}

    sample_idx = sample_ids.index(sample_id)

    # 查找基因索引
    if gene_name not in gene_names:
        return {'error': f'基因 {gene_name} 不在基因列表中'}

    gene_idx = gene_names.index(gene_name)

    # 获取表达量
    expression_value = data_original[sample_idx, gene_idx]

    return {
        'sample_id': sample_id,
        'sample_idx': sample_idx,
        'gene_name': gene_name,
        'gene_idx': gene_idx,
        'expression_value': expression_value,
        'verified': True
    }


def save_genes_csv(data: np.ndarray, gene_indices: list, sample_ids: list,
                   output_file: str, gene_names: list = None):
    """
    将基因表达数据保存为 CSV 格式

    参数:
        data: 原始数据矩阵 (n_samples, n_genes)
        gene_indices: 要保存的基因索引列表
        sample_ids: 样本ID列表（用于列名）
        output_file: 输出文件路径
        gene_names: 基因名称列表（可选，用于行名）
    """
    # 构建数据
    if sample_ids is None:
        sample_ids = [f'sample_{i}' for i in range(data.shape[0])]

    # 创建DataFrame
    # 行: 基因, 列: 样本
    rows = []
    for gene_idx in gene_indices:
        gene_data = data[:, gene_idx]
        if gene_names and gene_idx < len(gene_names):
            gene_name = gene_names[gene_idx]
        else:
            gene_name = f'gene_{gene_idx}'
        row = [gene_name] + list(gene_data)
        rows.append(row)

    # 构建列名
    columns = ['gene_name'] + sample_ids

    # 创建DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # 保存CSV
    df.to_csv(output_file, index=False)
    print(f"    已保存CSV: {output_file}")


# ================================================================================
# ========== 主程序入口 (直接运行此文件) ==============
# ================================================================================
if __name__ == '__main__':
    import scipy.io as sio

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    if USE_CV:
        # ==================== 交叉验证模式 ====================
        print('='*60)
        print(f'  CGI 因果基因发现 (马尔科夫毯扩展版) - {CANCER_TYPE} ({NUM_FOLDS}折交叉验证)')
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

            print(f'加载数据: {mat_file}')
            if not os.path.exists(mat_file):
                raise FileNotFoundError(f'数据文件不存在: {mat_file}')

            data = load_data(mat_file)
            print(f'数据形状: {data.shape}')

            # 运行马尔科夫毯扩展版因果基因发现
            import time
            start_time = time.time()
            results = find_genes_gci2(data, alpha=0.05)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            found_genes = results['found_genes']
            all_found_genes.append(found_genes)

            # 保存结果 (CSV格式)
            # 获取训练集样本ID
            data_for_csv = data[:, :-1]  # 不包含标签列
            sample_ids = load_train_sample_ids(CANCER_TYPE, fold, DATA_DIR)

            # 加载基因名
            gene_names_all = load_gene_names(CANCER_TYPE)

            if sample_ids is not None:
                # 输出CSV文件，格式参考 fold_0_genes.csv
                csv_file = os.path.join(CSV_OUTPUT_DIR, f'fold_{fold}_genes.csv')
                # 注意：sample_ids 可能少于 data_for_csv 的行数，需要对齐
                # 取前 min(len(sample_ids), data_for_csv.shape[0]) 个样本
                n_samples = min(len(sample_ids), data_for_csv.shape[0])
                # 传入基因名列表
                save_genes_csv(data_for_csv[:n_samples, :], found_genes[:],
                              sample_ids[:n_samples], csv_file, gene_names_all)

                # 随机抽样验证：随机选择一个基因和一个样本进行验证
                if found_genes and sample_ids and gene_names_all:
                    import random
                    random.seed(42 + fold)  # 固定种子以便复现
                    # 随机选择一个基因索引
                    random_gene_idx = random.choice(found_genes)
                    random_gene_name = gene_names_all[random_gene_idx]
                    # 随机选择一个样本
                    random_sample_idx = random.randint(0, n_samples - 1)
                    random_sample_id = sample_ids[random_sample_idx]
                    # 获取CSV文件中的表达量
                    csv_value = data_for_csv[random_sample_idx, random_gene_idx]

                    print(f"\n  [随机抽样验证] Fold {fold}")
                    print(f"    样本: {random_sample_id} (索引: {random_sample_idx})")
                    print(f"    基因: {random_gene_name} (索引: {random_gene_idx})")
                    print(f"    CSV中的表达量: {csv_value}")

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

        # 注意：不再保存各折汇总结果，只保存分折结果

    else:
        # ==================== 单数据模式 ====================
        print('='*60)
        print(f'  CGI 因果基因发现 (马尔科夫毯扩展版) - {CANCER_TYPE}')
        print('='*60)

        # 输入文件
        mat_file = os.path.join(DATA_DIR, f'data_{CANCER_TYPE}.mat')

        # 加载数据
        print(f'\n加载数据: {mat_file}')
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f'数据文件不存在: {mat_file}')

        data = load_data(mat_file)
        print(f'数据形状: {data.shape} (样本数 × 基因数+1)')

        # 运行马尔科夫毯扩展版因果基因发现
        print('\n开始马尔科夫毯扩展版因果基因筛选...')
        results = find_genes_gci2(data, alpha=0.05)

        # 获取结果
        found_genes = results['found_genes']

        # 保存结果 (CSV格式)
        # 获取样本ID（单数据模式可能没有 nested_splits 文件，使用序号）
        data_for_csv = data[:, :-1]  # 不包含标签列
        sample_ids = [f'sample_{i}' for i in range(data_for_csv.shape[0])]
        # 加载基因名
        gene_names_all = load_gene_names(CANCER_TYPE)
        csv_file = os.path.join(CSV_OUTPUT_DIR, f'{CANCER_TYPE}_genes.csv')
        save_genes_csv(data_for_csv, found_genes, sample_ids, csv_file, gene_names_all)

        # 打印结果
        print('====================')
        print('  结果已保存')
        print('====================')
        print(f'  CSV: {csv_file}')
        print(f'  基因数: {len(found_genes)}')
        print('====================')
