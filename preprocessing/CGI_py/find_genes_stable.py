"""
================================================================================
find_genes_stable.py - 基因稳定性验证与频率排序 Pipeline
================================================================================

【文件作用】
对每折训练集进行多次抽样运行 find_genes_gci2（马尔科夫毯扩展版），
统计基因出现频率并排序输出 top100（不进行降维，直接使用原始数据）

【重要更新 - 2026-03-20】
- 原版本调用 find_genes_gci (第一版)
- 新版本调用 find_genes_gci2 (马尔科夫毯扩展版)
- find_genes_gci2 在 find_genes_gci 的结果基础上，往外多做一层马尔科夫毯：
  1. 先运行 find_genes_gci 得到初始因果基因
  2. 轮流将筛选出来的基因作为目标变量进行筛选
  3. 将所有得到的基因的并集作为 found_genes 输出

【算法流程】
1. 加载每折的训练数据 (train_fold*.mat)
2. 对该fold进行N次抽样（Bootstrap或Random）
3. 运行 find_genes_gci2 筛选因果基因
4. 仅基于该fold的N次运行结果，统计基因出现频率并排序
5. 对每个fold分别输出Top K基因

【使用方法】
```bash
python find_genes_stable.py
```

【配置参数】
- SAMPLE_MODE: 'bootstrap' (有放回) / 'random' (无放回) / 'partitioned' (分区抽样)
- SAMPLE_RATIO: 抽样比例 (仅 random 模式生效)
- NUM_BOOTSTRAP: 迭代次数 (bootstrap/random 模式)
- NUM_PARTITIONS: 分区数量 (仅 partitioned 模式)
- TOP_K: 输出前K个基因

【输出文件】
- stable_genes_fold{fold}_top100.mat: 该fold的前100基因 (MATLAB格式)
- stable_genes_fold{fold}_top100.txt: 该fold的前100基因 (文本格式)

【依赖】
- numpy
- scipy.io
- find_genes_gci2 (同目录下)

================================================================================
"""

import os
import sys
import datetime
import numpy as np
from scipy.io import savemat, loadmat
from collections import Counter
import time
import pandas as pd

# 添加当前目录到路径，用于导入find_genes_gci模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ================================================================================
# 原始版本使用 find_genes_gci (第一版)
# from find_genes_gci import find_genes_gci, load_data
#
# 新版本使用 find_genes_gci2 (马尔科夫毯扩展版)
# 改动说明：
#   - find_genes_gci2 在 find_genes_gci 的结果基础上，往外多做一层马尔科夫毯
#   - 轮流将筛选出来的基因作为目标变量进行筛选
#   - 将所有得到的基因的并集作为 found_genes 输出
# ================================================================================
# 动态导入：根据 MARKOV_BLANKET_LAYER 配置选择使用哪个函数
from find_genes_gci import load_data


# ================================================================================
# ========== 配置区域 =============
# ================================================================================

# 癌症类型 (brca, blca, hnsc, stad, coadread)
CANCER_TYPE = 'coadread' 

# 数据目录
DATA_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/CGI_nested_cv/{CANCER_TYPE}'

# 获取当前日期，用于输出目录
CURRENT_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

# 输出目录
OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/plot_cgi/{CANCER_TYPE}'

# CSV特征文件输出目录
CSV_OUTPUT_DIR = rf'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/features/stable/{CANCER_TYPE}'

# 交叉验证折数
NUM_FOLDS = 5

# 抽样模式: 'bootstrap' (有放回) / 'random' (无放回) / 'partitioned' (分区抽样)
SAMPLE_MODE = 'random'

# 每次抽取的比例 (仅在 random 模式下生效)
# bootstrap 模式下忽略此参数，固定为1.0
SAMPLE_RATIO = 0.9

# 迭代次数 (bootstrap/random 模式)
NUM_BOOTSTRAP = 50

# 分区数量 (仅 partitioned 模式生效)
# 将数据分成n份，每次取n-1份，循环n轮
NUM_PARTITIONS = 20

# 输出前K个基因
TOP_K = 100

# 基因频次阈值（用于最终特征导出规则）
# - <= 1: 按频次排序输出前 TOP_K 个基因
# - > 1 : 输出“出现频次 > 该阈值”的所有基因
GENE_FREQ_THRESHOLD = 1

# 随机种子
RANDOM_SEED = 42

# 显著性水平
ALPHA = 0.05

# 马尔科夫毯层数: 1 (一层) 或 2 (两层，马尔科夫毯扩展版)
MARKOV_BLANKET_LAYER = 1

# 根据配置动态导入对应的函数
if MARKOV_BLANKET_LAYER == 1:
    from find_genes_gci import find_genes_gci as find_genes_gci_func
    print(f"使用一层马尔科夫毯 (find_genes_gci)")
elif MARKOV_BLANKET_LAYER == 2:
    from find_genes_gci2 import find_genes_gci2 as find_genes_gci_func
    print(f"使用两层马尔科夫毯 (find_genes_gci2)")
else:
    raise ValueError(f"MARKOV_BLANKET_LAYER 必须是 1 或 2，当前值: {MARKOV_BLANKET_LAYER}")

# ================================================================================


def sample_data(data: np.ndarray, mode: str, ratio: float = 1.0, seed: int = None,
                iteration: int = 0, n_partitions: int = 10) -> np.ndarray:
    """
    数据抽样函数，支持三种模式

    参数:
        data: 原始数据，形状为 (n_samples, n_features)
        mode: 'bootstrap' / 'random' / 'partitioned'
        ratio: 抽样比例 (仅 random 模式生效)
        seed: 随机种子
        iteration: 当前迭代轮次 (仅 partitioned 模式生效)
        n_partitions: 分区数量 (仅 partitioned 模式生效)

    返回:
        抽取后的数据
    """
    n_samples = data.shape[0]

    if seed is not None:
        np.random.seed(seed)

    if mode == 'bootstrap':
        # 有放回抽样：每个样本可能被多次选中
        n_select = n_samples
        indices = np.random.choice(n_samples, size=n_select, replace=True)
    elif mode == 'random':
        # 无放回抽样：每个样本最多被选中一次
        n_select = int(n_samples * ratio)
        indices = np.random.choice(n_samples, size=n_select, replace=False)
    elif mode == 'partitioned':
        # 分区抽样：将数据分成n份，每次取n-1份
        # 先打乱数据顺序，然后分成n份
        np.random.seed(seed if seed is not None else 42)
        shuffled_indices = np.random.permutation(n_samples)
        chunk_size = n_samples // n_partitions

        # 本次要排除的样本索引范围
        val_start = iteration * chunk_size
        val_end = val_start + chunk_size if iteration < n_partitions - 1 else n_samples
        val_indices = shuffled_indices[val_start:val_end]

        # 本次使用的训练样本（排除val_indices）
        train_indices = np.concatenate([
            shuffled_indices[:val_start],
            shuffled_indices[val_end:]
        ])
        return data[train_indices]
    else:
        raise ValueError(f"Invalid sampling mode: {mode}. Must be 'bootstrap', 'random', or 'partitioned'.")

    return data[indices]


def load_train_sample_ids(cancer_type: str, fold: int, data_dir: str) -> list:
    """
    从 nested_splits_{fold}.csv 文件中读取训练集样本ID

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


def load_all_sample_ids(cancer_type: str, fold: int, data_dir: str) -> list:
    """
    从 nested_splits_{fold}.csv 文件中读取所有样本ID（train + val + test）

    参数:
        cancer_type: 癌症类型
        fold: 折数
        data_dir: 数据目录

    返回:
        list: 所有样本ID列表（按 train -> val -> test 顺序）
    """
    splits_file = os.path.join(data_dir, f'nested_splits_{fold}.csv')
    if not os.path.exists(splits_file):
        print(f"    警告: 未找到样本ID文件 {splits_file}")
        return None

    df = pd.read_csv(splits_file)
    # 依次读取 train, val, test 列，拼接成完整列表
    all_samples = []
    for col in ['train', 'val', 'test']:
        samples = df[col].dropna().tolist()
        all_samples.extend(samples)

    return all_samples


def load_full_data(cancer_type: str) -> tuple:
    """
    加载完整数据（包含 train + val + test 所有样本）

    参数:
        cancer_type: 癌症类型

    返回:
        tuple: (data_matrix, gene_names, patient_ids)
            - data_matrix: 完整数据矩阵 (n_samples, n_genes+1)，最后一列是time
            - gene_names: 基因名列表
            - patient_ids: 与 data_matrix 行一一对应的 patient_id 列表
    """
    # 完整数据文件路径（注意：在 coadread 子目录下）
    data_dir = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI/data'
    full_data_path = os.path.join(data_dir, cancer_type, f'{cancer_type}_data_with_id.csv')

    if not os.path.exists(full_data_path):
        print(f"    警告: 未找到完整数据文件 {full_data_path}")
        return None, None, None

    df = pd.read_csv(full_data_path)

    # 提取基因名（跳过第一列 patient_id 和最后一列 time）
    gene_names = df.columns[1:-1].tolist()
    patient_ids = df['patient_id'].astype(str).tolist()

    # 提取数据矩阵（跳过 patient_id 列，保留基因和 time）
    data_matrix = df.iloc[:, 1:].values  # shape: (n_samples, n_genes+1)

    print(f"    已加载完整数据: {data_matrix.shape[0]} 样本, {len(gene_names)} 基因")
    return data_matrix, gene_names, patient_ids


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


def save_stable_genes_csv(data: np.ndarray, gene_indices: list, sample_ids: list,
                          output_file: str, gene_names: list = None,
                          index_name: str = 'gene_name'):
    """
    将基因表达数据保存为 CSV 格式

    参数:
        data: 原始数据矩阵 (n_samples, n_genes)
        gene_indices: 要保存的基因索引列表
        sample_ids: 样本ID列表（用于列名）
        output_file: 输出文件路径
        gene_names: 基因名称列表（可选，用于行名）
        index_name: 索引名称（默认 'gene_name'）
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

    # 设置索引名称
    df.index.name = index_name

    # 保存CSV（包含索引名称）
    df.to_csv(output_file, index=True)
    print(f"    已保存CSV: {output_file}")


def run_stable_genes_pipeline():
    """
    主函数：运行基因稳定性验证pipeline
    """
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 根据抽样模式确定迭代次数
    if SAMPLE_MODE == 'partitioned':
        num_iterations = NUM_PARTITIONS
        sample_ratio_display = f"{NUM_PARTITIONS - 1}/{NUM_PARTITIONS}"
    elif SAMPLE_MODE == 'random':
        num_iterations = NUM_BOOTSTRAP
        sample_ratio_display = f"{int(SAMPLE_RATIO * 100)}%"
    else:  # bootstrap
        num_iterations = NUM_BOOTSTRAP
        sample_ratio_display = "100% (等量有放回)"

    # 规则：random 模式下，先对训练集全集跑 1 次，再进行 NUM_BOOTSTRAP 次随机抽样
    include_full_iteration = (SAMPLE_MODE == 'random')
    total_iterations = num_iterations + (1 if include_full_iteration else 0)

    print("=" * 70)
    print(f"  基因稳定性验证 Pipeline - {CANCER_TYPE}")
    print("=" * 70)
    print(f"  抽样模式: {SAMPLE_MODE}")
    if SAMPLE_MODE == 'partitioned':
        print(f"  分区数量: {NUM_PARTITIONS}, 每次取: {sample_ratio_display}")
    elif SAMPLE_MODE == 'random':
        print(f"  抽样比例: {sample_ratio_display} (无放回)")
    else:
        print(f"  抽样比例: {sample_ratio_display}")
    print(f"  交叉验证折数: {NUM_FOLDS}")
    if include_full_iteration:
        print(f"  迭代次数: {total_iterations} (全集1次 + 随机抽样{num_iterations}次)")
    else:
        print(f"  迭代次数: {num_iterations}")
    if GENE_FREQ_THRESHOLD <= 1:
        print(f"  基因筛选规则: 频次Top {TOP_K}（GENE_FREQ_THRESHOLD={GENE_FREQ_THRESHOLD}）")
    else:
        print(f"  基因筛选规则: 保留频次 > {GENE_FREQ_THRESHOLD} 的基因")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 70)

    total_start_time = time.time()

    # 遍历每个fold
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*60}")
        print(f"  Fold {fold + 1}/{NUM_FOLDS}")
        print(f"{'='*60}")

        # 加载该fold的训练数据
        mat_file = os.path.join(DATA_DIR, f'train_fold{fold}.mat')
        print(f"  加载数据: {mat_file}")

        if not os.path.exists(mat_file):
            print(f"  警告: 数据文件不存在，跳过 fold {fold}")
            continue

        # 加载数据（不进行降维）
        data = load_data(mat_file)
        n_samples, n_cols = data.shape
        n_genes = n_cols - 1
        print(f"  原始数据: {n_samples} 样本 × {n_genes} 基因")

        # 该fold的基因记录
        fold_all_genes = []

        # random 模式下先跑一次全集迭代（不抽样）
        if include_full_iteration:
            print(f"\n  Fold {fold}, Iter 1/{total_iterations} (full training set)")
            sampled_data = data
            print(f"    样本数: {sampled_data.shape[0]}")

            iter_start = time.time()
            results = find_genes_gci_func(sampled_data, alpha=ALPHA)
            iter_time = time.time() - iter_start

            found_genes = results['found_genes']
            print(f"    发现基因数: {len(found_genes)}, 耗时: {iter_time:.2f}s")
            fold_all_genes.extend(found_genes)

        # 迭代抽样
        for iteration in range(num_iterations):
            iter_seed = RANDOM_SEED + fold * 10000 + iteration if RANDOM_SEED else None
            iter_display = iteration + 1 + (1 if include_full_iteration else 0)

            print(f"\n  Fold {fold}, Iter {iter_display}/{total_iterations} (seed={iter_seed})")

            # 数据抽样
            sampled_data = sample_data(
                data,
                mode=SAMPLE_MODE,
                ratio=SAMPLE_RATIO,
                seed=iter_seed,
                iteration=iteration,
                n_partitions=NUM_PARTITIONS
            )
            print(f"    样本数: {sampled_data.shape[0]}")

            # 运行 find_genes_gci (根据 MARKOV_BLANKET_LAYER 配置选择一层或两层)
            iter_start = time.time()
            results = find_genes_gci_func(sampled_data, alpha=ALPHA)
            iter_time = time.time() - iter_start

            found_genes = results['found_genes']
            print(f"    发现基因数: {len(found_genes)}, 耗时: {iter_time:.2f}s")

            fold_all_genes.extend(found_genes)

        # 该fold的统计与输出
        print(f"\n{'='*60}")
        print(f"  Fold {fold} 统计结果")
        print(f"{'='*60}")

        gene_counts = Counter(fold_all_genes)
        total_unique_genes = len(gene_counts)

        print(f"  迭代次数: {total_iterations}")
        print(f"  该fold筛选出的基因总数（含重复）: {len(fold_all_genes)}")
        print(f"  唯一基因数: {total_unique_genes}")

        sorted_genes = sorted(gene_counts.items(), key=lambda x: -x[1])
        if GENE_FREQ_THRESHOLD <= 1:
            selected_genes = sorted_genes[:TOP_K]
            select_rule_desc = f"Top {TOP_K} 基因 (按出现频率排序)"
        else:
            selected_genes = [g for g in sorted_genes if g[1] > GENE_FREQ_THRESHOLD]
            select_rule_desc = f"出现频次 > {GENE_FREQ_THRESHOLD} 的基因"

        if len(selected_genes) == 0:
            raise ValueError(
                f"Fold {fold} 在当前阈值配置下未筛出任何基因: GENE_FREQ_THRESHOLD={GENE_FREQ_THRESHOLD}"
            )

        print(f"\n  Fold {fold} {select_rule_desc}:")
        print(f"  {'排名':<6} {'基因索引':<12} {'出现次数':<12} {'出现频率':<10}")
        print(f"  {'-'*45}")

        for rank, (gene_idx, count) in enumerate(selected_genes, 1):
            freq = count / total_iterations * 100
            print(f"  {rank:<6} {gene_idx:<12} {count:<12} {freq:.1f}%")

        # 保存结果
        # 1. MATLAB格式
        top_genes_mat = {
            'top_genes_indices': [g[0] for g in selected_genes],
            'top_genes_counts': [g[1] for g in selected_genes],
            'all_gene_counts': gene_counts,
            'num_iterations': total_iterations,
            'num_bootstrap_iterations': num_iterations,
            'include_full_iteration': include_full_iteration,
            'sample_mode': SAMPLE_MODE,
            'sample_ratio': SAMPLE_RATIO if SAMPLE_MODE == 'random' else (NUM_PARTITIONS - 1) / NUM_PARTITIONS if SAMPLE_MODE == 'partitioned' else 1.0,
            'num_partitions': NUM_PARTITIONS if SAMPLE_MODE == 'partitioned' else 0,
            'cancer_type': CANCER_TYPE,
            'fold': fold,
            'gene_freq_threshold': GENE_FREQ_THRESHOLD,
            'top_k_when_threshold_disabled': TOP_K
        }
        mat_output_file = os.path.join(OUTPUT_DIR, f'stable_genes_fold{fold}_top100.mat')
        savemat(mat_output_file, top_genes_mat)
        print(f"\n  已保存: {mat_output_file}")

        # 2. 文本格式 (新实验在上面)
        txt_output_file = os.path.join(OUTPUT_DIR, f'stable_genes_fold{fold}_top100.txt')
        if SAMPLE_MODE == 'random':
            sample_mode_desc = f"Random (无放回抽样, 比例={SAMPLE_RATIO})"
        elif SAMPLE_MODE == 'partitioned':
            sample_mode_desc = f"Partitioned (分区抽样, {NUM_PARTITIONS}份, 每次取{NUM_PARTITIONS - 1}份)"
        else:
            sample_mode_desc = "Bootstrap (有放回抽样, 比例=1.0)"

        # 构建新内容
        content_lines = []
        content_lines.append(f"# 实验时间: {CURRENT_DATE}")
        content_lines.append(f"# 基因稳定性验证结果 - Fold {fold} - {CANCER_TYPE}")
        content_lines.append(f"# 抽样模式: {sample_mode_desc}")
        content_lines.append(f"# 迭代次数: {total_iterations}")
        if include_full_iteration:
            content_lines.append(f"# 其中随机抽样次数: {num_iterations} (另含全集1次)")
        content_lines.append(f"# 唯一基因数: {total_unique_genes}")
        content_lines.append(f"# 频次阈值: {GENE_FREQ_THRESHOLD}")
        content_lines.append(f"# TopK(阈值<=1时生效): {TOP_K}")
        content_lines.append(f"#")
        content_lines.append(f"# 排名\t基因索引\t出现次数\t出现频率")
        content_lines.append(f"{'='*50}")
        for rank, (gene_idx, count) in enumerate(selected_genes, 1):
            freq = count / total_iterations * 100
            content_lines.append(f"{rank}\t{gene_idx}\t{count}\t{freq:.1f}%")

        new_content = '\n'.join(content_lines) + '\n'

        # 如果文件已存在，将新内容添加到最前面
        if os.path.exists(txt_output_file):
            with open(txt_output_file, 'r') as f:
                existing_content = f.read()
            with open(txt_output_file, 'w') as f:
                f.write(new_content)
                f.write(f"\n{'='*70}\n")
                f.write(existing_content)
        else:
            with open(txt_output_file, 'w') as f:
                f.write(new_content)
        print(f"  已保存: {txt_output_file}")

        # 3. CSV特征文件格式（包含完整数据集 train + val + test）
        # 确保CSV输出目录存在
        os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

        # 获取该fold的所有样本ID（train + val + test）
        sample_ids = load_all_sample_ids(CANCER_TYPE, fold, DATA_DIR)

        # 加载完整数据（包含所有样本）
        full_data, gene_names_all, patient_ids_all = load_full_data(CANCER_TYPE)

        if sample_ids is not None and full_data is not None and patient_ids_all is not None:
            # 获取top基因索引列表
            top_gene_indices = [g[0] for g in selected_genes]

            # 获取数据矩阵（不含标签列time）
            data_for_csv = full_data[:, :-1]  # 不包含最后一列（time）

            # 显式按 patient_id 对齐，避免假设 full_data 行顺序与 split 顺序一致
            patient_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_all)}
            missing_sample_ids = [sid for sid in sample_ids if sid not in patient_to_idx]
            if missing_sample_ids:
                preview = ', '.join(missing_sample_ids[:5])
                raise KeyError(
                    f"Fold {fold} has {len(missing_sample_ids)} sample IDs missing from full data. "
                    f"Examples: {preview}"
                )

            aligned_data = np.zeros((len(sample_ids), data_for_csv.shape[1]), dtype=data_for_csv.dtype)
            for i, sample_id in enumerate(sample_ids):
                aligned_data[i, :] = data_for_csv[patient_to_idx[sample_id], :]

            # 保存CSV
            csv_file = os.path.join(CSV_OUTPUT_DIR, f'fold_{fold}_genes.csv')
            save_stable_genes_csv(
                aligned_data,
                top_gene_indices,
                sample_ids,
                csv_file,
                gene_names_all
            )

        print(f"\n  Fold {fold} 完成!")

    # 汇总
    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"  所有Fold汇总")
    print(f"{'='*70}")
    print(f"  总Fold数: {NUM_FOLDS}")
    print(f"  每Fold迭代次数: {total_iterations}")
    print(f"  总运行次数: {NUM_FOLDS * total_iterations}")
    print(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")

    print(f"\n{'='*70}")
    print(f"  Pipeline 完成!")
    print(f"{'='*70}")
    print(f"\n  输出文件:")
    print(f"  - {OUTPUT_DIR}/stable_genes_fold{{fold}}_top100.mat")
    print(f"  - {OUTPUT_DIR}/stable_genes_fold{{fold}}_top100.txt")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_stable_genes_pipeline()
