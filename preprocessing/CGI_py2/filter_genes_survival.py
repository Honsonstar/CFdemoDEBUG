"""
================================================================================
filter_genes_survival.py - 基于单变量生存分析的基因预筛选 + 稳定性选择
================================================================================

【文件作用】
1. 第一阶段：使用Spearman秩相关分析对高维基因进行预筛选
2. 第二阶段：对预筛选后的基因进行Bootstrap稳定性选择

【数据背景】
- train_fold*.mat: 约288样本，5000基因，最后一列是生存时间
- 所有样本均为已死亡样本（无截尾数据）
- 使用Spearman秩相关等价于单变量Cox模型的Wald检验

【两阶段筛选逻辑】
阶段1 - Spearman预筛选：
1. 计算每个基因与生存时间的Spearman秩相关系数及p值
2. 保留 p < 0.05 的显著相关基因
3. 数量兜底机制：
   - 显著基因 < 200个 → 取Top 300
   - 200 ≤ 显著基因 ≤ 500个 → 保留实际数量
   - 显著基因 > 500个 → 取Top 500

阶段2 - Bootstrap稳定性选择：
1. 对预筛选后的数据进行Bootstrap有放回抽样
2. 每次运行find_genes_gci筛选因果基因
3. 统计各基因出现频率，输出Top K

【使用方法】
```bash
python filter_genes_survival.py
```

【输出文件】
阶段1输出：
- train_fold{fold}_filtered.mat: Spearman筛选后的数据矩阵
- selected_gene_indices_fold{fold}.mat: 选中基因在原始5000个特征中的索引

阶段2输出：
- filter_genes_fold{fold}_top100.mat: 该fold的Top100基因 (MATLAB格式)
- filter_genes_fold{fold}_top100.txt: 该fold的Top100基因 (文本格式)

【依赖】
- numpy
- scipy.io
- scipy.stats
- find_genes_gci (同目录下)

================================================================================
"""

import os
import sys
import datetime
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import spearmanr
from collections import Counter
import time

# 添加当前目录到路径，用于导入find_genes_gci模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入find_genes_gci的函数
from find_genes_gci import find_genes_gci, load_data as load_data_gci

# ================================================================================
# ========== 配置区域 =============
# ================================================================================

# 癌症类型
CANCER_TYPE = 'coadread'

# 原始数据目录
RAW_DATA_DIR = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/CGI_nested_cv/{CANCER_TYPE}'

# 降维后数据输出目录 (新目录)
FILTERED_DATA_DIR = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/data/filter_gene/{CANCER_TYPE}'

# 结果输出目录
OUTPUT_DIR = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/preprocessing/CGI_py/plot_cgi/{CANCER_TYPE}_filtered'

# 交叉验证折数
NUM_FOLDS = 2

# ========== 阶段1: Spearman预筛选参数 ==========
ALPHA = 0.05  # 显著性水平

# 基因数量兜底阈值
MIN_GENES = 200   # 最小保留基因数
MAX_GENES = 500   # 最大保留基因数
FALLBACK_GENES = 300  # 触发兜底时的保留数

# ========== 阶段2: Bootstrap稳定性选择参数 ==========
# 抽样模式: 'bootstrap' (有放回) 或 'random' (无放回)
SAMPLE_MODE = 'random'

# 每次抽取的比例 (仅在 random 模式下生效)
SAMPLE_RATIO = 0.9

# Bootstrap迭代次数
NUM_BOOTSTRAP = 50

# 输出前K个基因
TOP_K = 100

# 随机种子
RANDOM_SEED = 42

# 显著性水平 (传给find_genes_gci)
ALPHA_GCI = 0.05

# ================================================================================


def load_data(file_path: str) -> np.ndarray:
    """
    加载 .mat 文件
    """
    data = loadmat(file_path)
    for key in ['d', 'data', 'D', 'normalized']:
        if key in data:
            return data[key]
    raise ValueError(f'Could not find data variable in {file_path}')


def sample_data(data: np.ndarray, mode: str, ratio: float = 1.0, seed: int = None) -> np.ndarray:
    """
    数据抽样函数，支持两种模式

    参数:
        data: 原始数据，形状为 (n_samples, n_features)
        mode: 'bootstrap' (有放回) 或 'random' (无放回)
        ratio: 抽样比例
        seed: 随机种子

    返回:
        抽取后的数据
    """
    n_samples = data.shape[0]

    if seed is not None:
        np.random.seed(seed)

    if mode == 'bootstrap':
        n_select = n_samples
        indices = np.random.choice(n_samples, size=n_select, replace=True)
    elif mode == 'random':
        n_select = int(n_samples * ratio)
        indices = np.random.choice(n_samples, size=n_select, replace=False)
    else:
        raise ValueError(f"Invalid sampling mode: {mode}")

    return data[indices]


def filter_genes_by_survival(data: np.ndarray) -> tuple:
    """
    基于生存时间进行单变量基因筛选 (阶段1)
    """
    n_samples, n_cols = data.shape
    n_genes = n_cols - 1

    X = data[:, :n_genes]
    T = data[:, -1]

    print(f"    原始数据维度: {n_samples} 样本 × {n_genes} 基因")

    p_values = []
    print(f"    开始计算 Spearman 秩相关系数...")

    for i in range(n_genes):
        gene_expr = X[:, i]
        corr, p_val = spearmanr(gene_expr, T)
        p_values.append(p_val)

        if (i + 1) % 1000 == 0:
            print(f"      已处理 {i + 1}/{n_genes} 个基因")

    p_values = np.array(p_values)

    # 筛选 p < ALPHA 的基因
    significant_mask = p_values < ALPHA
    significant_indices = np.where(significant_mask)[0]
    n_significant = len(significant_indices)

    print(f"    显著相关基因数量 (p<{ALPHA}): {n_significant}")

    # 应用数量兜底机制
    if n_significant < MIN_GENES:
        print(f"    触发兜底机制: 显著基因 < {MIN_GENES}，取 Top {FALLBACK_GENES}")
        sorted_idx = np.argsort(p_values)
        selected_indices = sorted_idx[:FALLBACK_GENES]
        final_n_genes = FALLBACK_GENES
    elif n_significant > MAX_GENES:
        print(f"    显著基因 > {MAX_GENES}，取 Top {MAX_GENES}")
        sorted_idx = np.argsort(p_values)
        selected_indices = sorted_idx[:MAX_GENES]
        final_n_genes = MAX_GENES
    else:
        selected_indices = significant_indices
        final_n_genes = n_significant

    print(f"    最终保留维度: {n_samples} 样本 × {final_n_genes} 基因")

    X_filtered = X[:, selected_indices]
    filtered_data = np.column_stack([X_filtered, T])

    return filtered_data, selected_indices


def run_bootstrap_stability_selection(filtered_data: np.ndarray, fold: int, output_dir: str) -> list:
    """
    Bootstrap稳定性选择 (阶段2)
    """
    n_samples = filtered_data.shape[0]

    print(f"\n  === 阶段2: Bootstrap稳定性选择 ===")
    print(f"    迭代次数: {NUM_BOOTSTRAP}")
    print(f"    抽样模式: {SAMPLE_MODE}")

    # 该fold的基因记录
    fold_all_genes = []

    for iteration in range(NUM_BOOTSTRAP):
        iter_seed = RANDOM_SEED + fold * 10000 + iteration if RANDOM_SEED else None

        print(f"\n    Fold {fold}, Iter {iteration + 1}/{NUM_BOOTSTRAP} (seed={iter_seed})")

        # 数据抽样
        sampled_data = sample_data(
            filtered_data,
            mode=SAMPLE_MODE,
            ratio=SAMPLE_RATIO,
            seed=iter_seed
        )
        print(f"      样本数: {sampled_data.shape[0]}")

        # 运行find_genes_gci
        iter_start = time.time()
        results = find_genes_gci(sampled_data, alpha=ALPHA_GCI)
        iter_time = time.time() - iter_start

        found_genes = results['found_genes']
        print(f"      发现基因数: {len(found_genes)}, 耗时: {iter_time:.2f}s")

        fold_all_genes.extend(found_genes)

    # 统计基因频率
    print(f"\n  === Fold {fold} 统计结果 ===")

    gene_counts = Counter(fold_all_genes)
    total_unique_genes = len(gene_counts)

    print(f"    迭代次数: {NUM_BOOTSTRAP}")
    print(f"    筛选出的基因总数（含重复）: {len(fold_all_genes)}")
    print(f"    唯一基因数: {total_unique_genes}")

    # 按出现次数降序排序
    sorted_genes = sorted(gene_counts.items(), key=lambda x: -x[1])
    top_genes = sorted_genes[:TOP_K]

    # 打印Top K基因
    print(f"\n    Fold {fold} Top {TOP_K} 基因 (按出现频率排序):")
    print(f"    {'排名':<6} {'基因索引':<12} {'出现次数':<12} {'出现频率':<10}")
    print(f"    {'-'*45}")

    for rank, (gene_idx, count) in enumerate(top_genes, 1):
        freq = count / NUM_BOOTSTRAP * 100
        print(f"    {rank:<6} {gene_idx:<12} {count:<12} {freq:.1f}%")

    return top_genes, gene_counts


def run_full_pipeline():
    """
    主函数：运行完整流程
    """
    # 替换目录中的占位符
    raw_data_dir = RAW_DATA_DIR.format(CANCER_TYPE=CANCER_TYPE)
    filtered_data_dir = FILTERED_DATA_DIR.format(CANCER_TYPE=CANCER_TYPE)
    output_dir = OUTPUT_DIR.format(CANCER_TYPE=CANCER_TYPE, CURRENT_DATE=CURRENT_DATE)

    print("=" * 70)
    print("  基因筛选完整 Pipeline")
    print("  阶段1: Spearman预筛选 + 阶段2: Bootstrap稳定性选择")
    print("=" * 70)
    print(f"  癌症类型: {CANCER_TYPE}")
    print(f"  原始数据目录: {raw_data_dir}")
    print(f"  降维数据输出目录: {filtered_data_dir}")
    print(f"  结果输出目录: {output_dir}")
    print(f"  交叉验证折数: {NUM_FOLDS}")
    print(f"  Bootstrap迭代次数: {NUM_BOOTSTRAP}")
    print(f"  输出Top K基因: {TOP_K}")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(filtered_data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    total_start_time = time.time()

    # 遍历每个fold
    for fold in range(NUM_FOLDS):
        print(f"\n{'='*60}")
        print(f"  Fold {fold}")
        print(f"{'='*60}")

        # ========== 阶段1: 加载原始数据并进行Spearman筛选 ==========
        print(f"\n  === 阶段1: Spearman预筛选 ===")

        mat_file = os.path.join(raw_data_dir, f'train_fold{fold}.mat')
        print(f"  加载数据: {mat_file}")

        if not os.path.exists(mat_file):
            print(f"  警告: 文件不存在，跳过 fold {fold}")
            continue

        data = load_data(mat_file)
        n_samples, n_cols = data.shape
        n_genes = n_cols - 1
        print(f"  原始数据: {n_samples} 样本 × {n_genes} 基因")

        # 执行Spearman筛选
        filtered_data, selected_indices = filter_genes_by_survival(data)

        # 保存筛选后的数据
        filtered_file = os.path.join(filtered_data_dir, f'train_fold{fold}_filtered.mat')
        savemat(filtered_file, {'data': filtered_data})
        print(f"  已保存: {filtered_file}")

        # 保存选中的基因索引
        indices_file = os.path.join(filtered_data_dir, f'selected_gene_indices_fold{fold}.mat')
        savemat(indices_file, {'selected_indices': selected_indices})
        print(f"  已保存: {indices_file}")

        # ========== 阶段2: Bootstrap稳定性选择 ==========
        top_genes, gene_counts = run_bootstrap_stability_selection(filtered_data, fold, output_dir)

        # 保存该fold的Top100结果
        # 1. MATLAB格式
        top_genes_mat = {
            'top_genes_indices': [g[0] for g in top_genes],
            'top_genes_counts': [g[1] for g in top_genes],
            'all_gene_counts': gene_counts,
            'num_iterations': NUM_BOOTSTRAP,
            'sample_mode': SAMPLE_MODE,
            'sample_ratio': SAMPLE_RATIO if SAMPLE_MODE == 'random' else 1.0,
            'cancer_type': CANCER_TYPE,
            'fold': fold
        }
        mat_output_file = os.path.join(output_dir, f'filter_genes_fold{fold}_top100.mat')
        savemat(mat_output_file, top_genes_mat)
        print(f"\n  已保存: {mat_output_file}")

        # 2. 文本格式 (追加模式)
        txt_output_file = os.path.join(output_dir, f'filter_genes_fold{fold}_top100.txt')
        if SAMPLE_MODE == 'random':
            sample_mode_desc = f"Random (无放回抽样, 比例={SAMPLE_RATIO})"
        else:
            sample_mode_desc = "Bootstrap (有放回抽样, 比例=1.0)"

        # 构建内容
        content_lines = []
        content_lines.append(f"# 实验时间: {CURRENT_DATE}")
        content_lines.append(f"# 基因筛选结果 - Fold {fold} - {CANCER_TYPE}")
        content_lines.append(f"# 阶段1: Spearman预筛选 (p<{ALPHA})")
        content_lines.append(f"# 阶段2: {sample_mode_desc}")
        content_lines.append(f"# 迭代次数: {NUM_BOOTSTRAP}")
        content_lines.append(f"#")
        content_lines.append(f"# 排名\t基因索引\t出现次数\t出现频率")
        content_lines.append(f"{'='*50}")
        for rank, (gene_idx, count) in enumerate(top_genes, 1):
            freq = count / NUM_BOOTSTRAP * 100
            content_lines.append(f"{rank}\t{gene_idx}\t{count}\t{freq:.1f}%")

        content = '\n'.join(content_lines) + '\n'

        # 新实验放在最前面：如果文件存在，将新内容添加到最前面
        if os.path.exists(txt_output_file):
            with open(txt_output_file, 'r') as f:
                existing_content = f.read()
            with open(txt_output_file, 'w') as f:
                f.write(content)
                f.write(f"\n{'='*70}\n")
                f.write(existing_content)
        else:
            with open(txt_output_file, 'w') as f:
                f.write(content)
        print(f"  已保存: {txt_output_file}")

        print(f"\n  Fold {fold} 完成!")

    # 汇总
    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print("  Pipeline 完成!")
    print(f"{'='*70}")
    print(f"  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"\n  输出文件说明:")
    print(f"  阶段1 (降维数据):")
    print(f"    - {filtered_data_dir}/train_fold{{fold}}_filtered.mat")
    print(f"    - {filtered_data_dir}/selected_gene_indices_fold{{fold}}.mat")
    print(f"  阶段2 (稳定性选择):")
    print(f"    - {output_dir}/filter_genes_fold{{fold}}_top100.mat")
    print(f"    - {output_dir}/filter_genes_fold{{fold}}_top100.txt")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_full_pipeline()
