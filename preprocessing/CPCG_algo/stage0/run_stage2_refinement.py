#!/usr/bin/env python3
"""
Stage 2 特征精炼脚本 - 基于 PC 算法对 mRMR 筛选的特征进行二次筛选

使用 PC (Peter-Clark) 算法找出与生存时间 (OS) 直接关联的基因 (Markov Blanket)。

用法：
    # 方式1: 处理单个 fold
    python run_stage2_refinement.py --study brca --fold 0 \\
        --split_dir ../../../splits/nested_cv \\
        --data_root_dir ../../../datasets_csv/raw_rna_data/combine \\
        --clinical_dir ../../../datasets_csv/clinical_data

    # 方式2: 批量处理所有 folds
    python run_stage2_refinement.py --study brca --fold all \\
        --split_dir ../../../splits/nested_cv \\
        --data_root_dir ../../../datasets_csv/raw_rna_data/combine \\
        --clinical_dir ../../../datasets_csv/clinical_data
"""

import os
import sys
import numpy as np
import pandas as pd
from pingouin import partial_corr
from itertools import combinations
import networkx as nx
from joblib import Parallel, delayed
from scipy import stats as scipy_stats


# ============================================================
# PC 算法核心函数 (从 Stage2/main.py 复制)
# ============================================================

def skeleton(data, alpha: float = 0.05, max_l: int = 2):
    """
    PC 算法骨架构建（优化版）

    优化点：
    1. Depth 0 使用 df.corr() 向量化计算
    2. Depth 1+ 使用 joblib.Parallel 并行化条件独立性测试
    """
    n_nodes = data.shape[1]
    labels = data.columns.to_list()
    G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    # ============================================================
    # 【优化1】Depth 0: 使用 df.corr() 向量化计算
    # ============================================================
    if max_l >= 0:
        print(f"    [Depth 0] 向量化计算皮尔逊相关性矩阵...")

        # 向量化计算所有对的皮尔逊相关系数
        corr_matrix = data.corr(method='pearson')

        # 将相关系数转换为 p-value (t-test)
        n_samples = len(data)
        for x, y in pairs:
            if G[x][y]:
                r = corr_matrix.iloc[x, y]
                if pd.isna(r):
                    G[x][y] = G[y][x] = False
                    continue

                # 计算 t 统计量
                t_stat = r * np.sqrt((n_samples - 2) / (1 - r**2))
                p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n_samples - 2))

                if p_value >= alpha:
                    G[x][y] = G[y][x] = False

        print(f"    [Depth 0] 向量化计算完成，处理 {len(pairs)} 对变量")

    # ============================================================
    # 主循环: l = 0, 1, 2, ...
    # ============================================================
    l = 0
    while l < max_l and any([any(row) for row in G]):
        print(f"    [Depth {l}] 正在处理 {sum(sum(1 for j in row if j) for row in G)//2} 条边...")

        # ============================================================
        # 【优化2】Depth 1+: 使用 joblib.Parallel 并行化
        # ============================================================
        if l >= 1:
            # 收集所有需要测试的 (x, y, K) 组合
            test_tasks = []
            for x, y in pairs:
                if G[x][y]:
                    neighbors = [i for i in range(n_nodes) if G[x][i] and i != y]
                    if len(neighbors) >= l:
                        for K in combinations(neighbors, l):
                            test_tasks.append((x, y, K))

            if test_tasks:
                # 并行执行所有条件独立性测试
                def test_ci(task):
                    x, y, K = task
                    cc = [labels[k] for k in K]
                    try:
                        stats = partial_corr(data=data, x=labels[x], y=labels[y], covar=cc)
                        p_value = stats.loc['pearson', 'p-val']
                        return (x, y, p_value)
                    except:
                        return (x, y, 0.0)  # 出错时保持连接

                results = Parallel(n_jobs=-1, verbose=5)(
                    delayed(test_ci)(task) for task in test_tasks
                )

                # 根据测试结果更新图
                for x, y, p_value in results:
                    if p_value >= alpha:
                        G[x][y] = G[y][x] = False

        else:
            # Depth 0 已在上方处理
            pass

        l += 1

    return np.asarray(G, dtype=int)


def cs_step_2(result_cs1, hazard_type):
    """
    Stage 2: 使用 PC 算法筛选与目标变量直接相关的特征
    
    Args:
        result_cs1: DataFrame (行=样本, 列=特征+目标变量)
        hazard_type: 目标变量列名 (如 'OS')
        
    Returns:
        筛选后的 DataFrame
    """
    data = result_cs1.copy()
    # 确保数据是数值型
    data = data.select_dtypes(include=[np.number])
    labels = data.columns.to_list()

    try:
        # 1. 构建骨架
        G = skeleton(data, alpha=0.10, max_l=2)
        G_nx = nx.from_numpy_array(np.array(G))
        
        # 2. 找到与 OS (hazard_type) 相关的邻居
        if hazard_type in labels:
            OS_idx = labels.index(hazard_type)
            # 扩展到距离2的邻居（Markov Blanket 近似）
            neighbors = list(nx.single_source_shortest_path_length(G_nx, OS_idx, cutoff=2).keys())
            c_label = [labels[i] for i in neighbors]

            # 计算原始基因数（排除 OS）
            original_genes = len(labels) - 1

            # 统计保留的基因数
            retained_genes = len(c_label) - 1 if hazard_type in c_label else len(c_label)

            print(f"    [Stage2] PC算法骨架边数: {G.sum()//2}")
            print(f"    [Stage2] PC算法筛选完成: 从 {original_genes} -> {retained_genes} 个基因")

            return data.loc[:, c_label]
        else:
            print(f"    [Stage2] 警告: 数据中找不到 {hazard_type} 列，返回原始数据")
            return data
            
    except Exception as e:
        print(f"    [Stage2] 算法运行出错: {e}，启用兜底策略(返回输入数据)")
        return data


# ============================================================
# Stage 2 特征精炼主类
# ============================================================

class Stage2Refiner:
    """基于 PC 算法的特征精炼器，用于 mRMR 后的二次筛选"""
    
    # 输入/输出根目录
    INPUT_ROOT = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/features'
    OUTPUT_ROOT = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/features'
    
    def __init__(self, study, clinical_dir=None):
        """
        初始化 Stage 2 特征精炼器
        
        Args:
            study: 癌症类型 (e.g., 'brca', 'luad')
            clinical_dir: 临床数据目录，包含 tcga_{study}_clinical.csv
        """
        self.study = study
        self.clinical_dir = clinical_dir
        
    def refine_features_for_fold(self, fold):
        """
        为指定的 fold 执行 Stage 2 特征精炼
        
        Args:
            fold: fold 编号
            
        Returns:
            输出文件路径
        """
        print(f"\n[{self.study}] Fold {fold}: 开始 Stage 2 特征精炼...")
        
        # 1. 读取 mRMR 筛选的基因文件
        mrmr_file = os.path.join(
            self.INPUT_ROOT, 
            f'mrmr_{self.study}', 
            f'fold_{fold}_genes.csv'
        )
        
        if not os.path.exists(mrmr_file):
            print(f"  ❌ 找不到 mRMR 输入文件: {mrmr_file}")
            return None
        
        # 读取并转置：(行=基因, 列=样本) -> (行=样本, 列=基因)
        df_genes = pd.read_csv(mrmr_file, index_col=0)
        print(f"  [数据] mRMR 基因文件: {df_genes.shape[0]} 基因 x {df_genes.shape[1]} 样本")
        
        df_genes_T = df_genes.T  # 转置: 行=样本, 列=基因
        df_genes_T.index = [str(i)[:12] for i in df_genes_T.index]  # 截断样本 ID
        
        # 2. 读取临床数据获取 OS 列
        clinical_data = self._load_clinical_data()
        
        # 3. 对齐样本并合并数据
        df_merged = self._merge_data(df_genes_T, clinical_data)
        
        if df_merged is None or len(df_merged) < 10:
            print("  ⚠️ Warning: 合并后样本数不足，跳过 Stage 2")
            return self._copy_input_as_output(fold, df_genes)
        
        # 4. 执行 PC 算法筛选
        df_selected = cs_step_2(df_merged, hazard_type='OS')
        
        # 5. 移除 OS 列并转置回 (行=基因, 列=样本)
        if 'OS' in df_selected.columns:
            df_selected = df_selected.drop(columns=['OS'])
        
        # 6. 保存结果
        return self._save_output(fold, df_selected)
    
    def _load_clinical_data(self):
        """加载临床数据并提取 OS 列"""
        clinical_file = None
        
        # 优先使用指定的 clinical_dir
        if self.clinical_dir:
            # 尝试两种文件名格式
            file_patterns = [
                os.path.join(self.clinical_dir, f'tcga_{self.study}', 'clinical.CSV'),
                os.path.join(self.clinical_dir, f'tcga_{self.study}_clinical.csv'),
            ]
            for pattern in file_patterns:
                if os.path.exists(pattern):
                    clinical_file = pattern
                    break
        else:
            # 默认路径
            default_paths = [
                f'../raw_data/tcga_{self.study}/clinical.CSV',
                f'datasets_csv/clinical_data/tcga_{self.study}_clinical.csv',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    clinical_file = path
                    break
        
        if clinical_file is None or not os.path.exists(clinical_file):
            raise FileNotFoundError(
                f"找不到临床数据文件。尝试过的路径:\n"
                f"  - ../raw_data/tcga_{self.study}/clinical.CSV\n"
                f"  - datasets_csv/clinical_data/tcga_{self.study}_clinical.csv\n"
                f"请使用 --clinical_dir 参数指定临床数据目录"
            )
        
        clinical_data = pd.read_csv(clinical_file)
        
        # 识别样本 ID 列
        id_col = None
        for col in ['case_id', 'case_submitter_id', 'CASE_ID', 'patient_id', 'PATIENT_ID']:
            if col in clinical_data.columns:
                id_col = col
                break
        
        if id_col is None:
            id_col = clinical_data.columns[0]
            print(f"  [警告] 未找到标准 ID 列，使用第一列作为 ID: {id_col}")
        
        # 截断 ID 为 12 位
        clinical_data['case_id_truncated'] = clinical_data[id_col].astype(str).str[:12]
        
        # 识别 OS 列
        os_col = None
        for col in ['OS', 'survival_months', 'os_months', 'OS_MONTHS']:
            if col in clinical_data.columns:
                os_col = col
                break
        
        if os_col is None:
            raise ValueError(
                f"临床数据中找不到 OS (生存时间) 列。\n"
                f"可用列: {list(clinical_data.columns)}"
            )
        
        # 构建 ID -> OS 的映射
        df_os = clinical_data[['case_id_truncated', os_col]].copy()
        df_os.columns = ['case_id', 'OS']
        df_os = df_os.set_index('case_id')
        df_os['OS'] = pd.to_numeric(df_os['OS'], errors='coerce')  # 转换为数值型
        
        print(f"  [数据] 临床数据: {len(clinical_data)} 样本, OS列: {os_col}")
        print(f"         有效 OS 值: {df_os['OS'].notna().sum()} / {len(df_os)}")
        
        return df_os
    
    def _merge_data(self, df_genes, df_os):
        """
        合并基因表达数据和临床 OS 数据
        
        Args:
            df_genes: 基因表达数据 (行=样本, 列=基因)
            df_os: 临床 OS 数据 (行=样本, 列=['OS'])
            
        Returns:
            合并后的 DataFrame
        """
        # Inner merge: 只保留同时存在于基因数据和临床数据中的样本
        df_merged = pd.merge(
            df_os, 
            df_genes, 
            left_index=True, 
            right_index=True, 
            how='inner'
        )
        
        # 移除 OS 为 NaN 的样本
        df_merged = df_merged[df_merged['OS'].notna()]
        
        print(f"  [合并] 基因数据 {len(df_genes)} 样本 + 临床数据 {len(df_os)} 样本")
        print(f"         -> 合并后 {len(df_merged)} 样本 (OS + {df_merged.shape[1]-1} 基因)")
        
        return df_merged
    
    def _save_output(self, fold, df_selected):
        """
        保存筛选后的特征文件
        
        Args:
            fold: fold 编号
            df_selected: 筛选后的数据 (行=样本, 列=基因)
            
        Returns:
            输出文件路径
        """
        # 输出目录: features/mrmr_stage2_{study}/
        output_dir = os.path.join(self.OUTPUT_ROOT, f'mrmr_stage2_{self.study}')
        os.makedirs(output_dir, exist_ok=True)
        
        out_file = os.path.join(output_dir, f'fold_{fold}_genes.csv')
        
        # 转置回 (行=基因, 列=样本)
        df_output = df_selected.T
        df_output.index.name = 'gene_name'
        df_output.to_csv(out_file)
        
        print(f"  [保存] {out_file}")
        print(f"         基因数: {df_output.shape[0]}, 样本数: {df_output.shape[1]}")
        
        return out_file
    
    def _copy_input_as_output(self, fold, df_genes):
        """当 Stage 2 无法执行时，直接复制输入作为输出"""
        output_dir = os.path.join(self.OUTPUT_ROOT, f'mrmr_stage2_{self.study}')
        os.makedirs(output_dir, exist_ok=True)
        
        out_file = os.path.join(output_dir, f'fold_{fold}_genes.csv')
        df_genes.to_csv(out_file)
        
        print(f"  [保存] {out_file} (复制输入)")
        
        return out_file


# ============================================================
# 命令行入口
# ============================================================

def main():
    """主函数：命令行入口"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description='Stage 2 特征精炼: 使用 PC 算法对 mRMR 筛选的特征进行二次筛选',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 处理单个 fold
    python run_stage2_refinement.py --study brca --fold 0 --clinical_dir /path/to/clinical
    
    # 批量处理所有 folds
    python run_stage2_refinement.py --study brca --fold all --clinical_dir /path/to/clinical
        """
    )
    parser.add_argument('--study', type=str, required=True,
                        help='癌症类型 (e.g., brca, luad, stad)')
    parser.add_argument('--fold', type=str, required=True,
                        help='Fold 编号 (0-4) 或 "all" 运行所有 folds')
    parser.add_argument('--clinical_dir', type=str, default=None,
                        help='临床数据目录，包含 tcga_{study}_clinical.csv (可选)')
    
    args = parser.parse_args()
    
    # 确定要运行的 folds
    if args.fold.lower() == 'all':
        # 自动发现所有 mrmr 输入文件
        input_dir = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/features/mrmr_{args.study}'
        if not os.path.exists(input_dir):
            raise FileNotFoundError(
                f"找不到 mRMR 输入目录: {input_dir}\n"
                f"请先运行 run_mrmr.py 生成 mRMR 特征"
            )
        
        pattern = os.path.join(input_dir, 'fold_*_genes.csv')
        input_files = sorted(glob.glob(pattern))
        if not input_files:
            raise FileNotFoundError(f"在 {input_dir} 下找不到 fold_*_genes.csv 文件")
        
        folds = [int(os.path.basename(f).replace('fold_', '').replace('_genes.csv', '')) 
                 for f in input_files]
        print(f"[Info] 发现 {len(folds)} 个 folds: {folds}")
    else:
        folds = [int(args.fold)]
    
    # 创建特征精炼器
    refiner = Stage2Refiner(
        study=args.study,
        clinical_dir=args.clinical_dir
    )
    
    output_files = []
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"[Fold {fold}] 处理中...")
        print(f"{'='*60}")
        
        # 运行 Stage 2 特征精炼
        output_file = refiner.refine_features_for_fold(fold=fold)
        
        if output_file:
            output_files.append(output_file)
            print(f"\n✅ Fold {fold} 完成!")
            print(f"   输出文件: {output_file}")
        else:
            print(f"\n⚠️  Fold {fold} 跳过")
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"全部完成! 共处理 {len(output_files)} 个 folds")
    for f in output_files:
        print(f"  - {f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
