#!/usr/bin/env python3
"""
MRMR基因签名比对工具
对比全局CPCG、嵌套CV中MRMR筛选的基因重合度
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse

def load_global_cpog_genes(study):
    """加载全局CPCG筛选的基因"""
    # 调整路径优先级：results/comparison 下的新文件优先！
    paths = [
        f'results/comparison/{study}/global_genes.csv',  # 1. 优先读取 run_fresh_global.sh 生成的文件
        f'preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_{study}/tcga_{study}_M2M3base_0916.csv', # 2. 旧版备份
        f'preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_{study}/clinical_final.CSV'
    ]
    
    for file_path in paths:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # 排除非基因列
                exclude = ['sample_id', 'OS', 'Censor', 'case_id', 'Unnamed: 0', 'survival_months', 'censorship']
                gene_cols = [col for col in df.columns if col not in exclude]
                print(f"✓ 全局CPCG基因数: {len(gene_cols)} (来源: {os.path.basename(file_path)})")
                return set(gene_cols)
            except:
                continue
                
    print(f"⚠️ 找不到全局CPCG结果文件，将只对比Nested CV内部一致性。")
    return set()

def load_nested_mrmr_genes(study, use_stage2=False):
    """加载嵌套CV各折MRMR筛选的基因，同时统计训练样本数
    
    Args:
        study: 癌种名称
        use_stage2: 是否使用 Stage2 (PC算法) 精炼后的基因
                    False: 使用 mrmr_{study} (mRMR 原始输出)
                    True: 使用 mrmr_stage2_{study} (PC算法精炼后)
    """
    # 根据参数选择目录
    if use_stage2:
        features_dir = f'features/mrmr_stage2_{study}'
        print(f"📊 使用 Stage2 精炼后的基因 (路径: {features_dir})")
    else:
        features_dir = f'features/mrmr_{study}'
        print(f"📊 使用 mRMR 原始筛选的基因 (路径: {features_dir})")
    
    split_dir = f'splits/nested_cv/{study}'

    if not os.path.exists(features_dir):
        print(f"❌ 找不到目录: {features_dir}")
        return {}

    nested_genes = {}

    for fold in range(5):
        gene_file = f'{features_dir}/fold_{fold}_genes.csv'

        if not os.path.exists(gene_file):
            print(f"⚠️  找不到: {gene_file}")
            continue

        try:
            df = pd.read_csv(gene_file)

            # 基因名在第一列（列名为 'gene_name'）
            if 'gene_name' in df.columns:
                genes = df['gene_name'].dropna().unique().tolist()
            else:
                # 兜底：尝试找可能的基因名列
                id_cols = ['sample_id', 'case_id', 'Unnamed: 0', 'patient_id']
                gene_col = [c for c in df.columns if c not in id_cols and c not in ['OS', 'Censor', 'survival_months']]
                if gene_col:
                    genes = df[gene_col[0]].dropna().unique().tolist()
                else:
                    genes = []

            # 统计训练样本数
            n_train = 0
            split_file = f'{split_dir}/nested_splits_{fold}.csv'
            if os.path.exists(split_file):
                split_df = pd.read_csv(split_file)
                n_train = split_df['train'].notna().sum()

            if not genes:
                print(f"⚠️ Fold {fold} 文件为空或无基因列")
            else:
                nested_genes[fold] = {'genes': set(genes), 'n_train': n_train}
                mode_label = "Stage2精炼" if use_stage2 else "MRMR筛选"
                print(f"✓ {mode_label} Fold {fold}: {len(genes)} 基因, {n_train} 训练样本")

        except Exception as e:
            print(f"❌ 读取 Fold {fold} 出错: {e}")

    return nested_genes

def calculate_overlap(set1, set2):
    """计算两个基因集的交集和重合率"""
    if not set1 or not set2:
        return set(), 0.0, 0.0
        
    intersection = set1 & set2
    union = set1 | set2
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    # Overlap rate relative to the smaller set size
    overlap_rate = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
    
    return intersection, jaccard, overlap_rate

def compare_mrmr_signatures(study, use_stage2=False):
    """比对MRMR筛选的基因重合度
    
    Args:
        study: 癌种名称
        use_stage2: 是否使用 Stage2 精炼后的基因
    """
    print(f"\n{'='*60}")
    mode_str = "MRMR + Stage2 (PC算法)" if use_stage2 else "MRMR"
    print(f"{mode_str} 基因签名比对: {study}")
    print(f"{'='*60}")
    
    nested_genes = load_nested_mrmr_genes(study, use_stage2=use_stage2)
    global_genes = load_global_cpog_genes(study)
    
    if not nested_genes:
        print("❌ 无法加载任何嵌套CV MRMR基因，请检查 features/mrmr_* 目录")
        return

    # 提取基因集合和训练样本数
    folds = sorted(nested_genes.keys())
    genes_dict = {f: nested_genes[f]['genes'] for f in folds}
    n_train_dict = {f: nested_genes[f]['n_train'] for f in folds}

    # 1. 嵌套CV内部一致性 (这是重点)
    mode_label = "Stage2精炼" if use_stage2 else "MRMR筛选"
    print(f"\n📊 1. {mode_label}基因的嵌套CV内部稳定性 (Fold间重合度)")
    print("-" * 60)

    matrix = np.zeros((len(folds), len(folds)))

    consistency_scores = []

    # 表头
    print(f"{'Folds':<10} | {'交集数':<8} | {'重合率(%)':<10}")
    print("-" * 50)

    for i in range(len(folds)):
        for j in range(len(folds)):
            f_i, f_j = folds[i], folds[j]
            inter, jac, rate = calculate_overlap(genes_dict[f_i], genes_dict[f_j])
            matrix[i, j] = rate  # 使用重合率

            if i < j:
                consistency_scores.append(rate)
                print(f"{f_i} vs {f_j:<4} | {len(inter):<8} | {rate*100:.1f}%")

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    print("-" * 50)
    print(f"👉 平均一致性 (重合率): {avg_consistency:.4f}")
    
    # 2. 全局 vs 嵌套 (如果有全局结果)
    if global_genes:
        mode_label = "Stage2精炼" if use_stage2 else "MRMR"
        print(f"\n📊 2. 全局CPCG vs 嵌套CV各折的{mode_label}基因")
        print("-" * 60)
        for fold in folds:
            inter, jac, rate = calculate_overlap(global_genes, genes_dict[fold])
            print(f"Global vs Fold {fold}: 交集 {len(inter)} 个 (重合率 {rate*100:.1f}%)")
    
    # 3. 生成热力图
    try:
        plt.figure(figsize=(7, 5))

        # 计算平均训练样本数
        avg_train = int(np.mean(list(n_train_dict.values())))

        # 根据模式选择配色
        cmap = 'Purples' if use_stage2 else 'Oranges'
        mode_label = 'MRMR+Stage2' if use_stage2 else 'MRMR'
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap,
                   xticklabels=[f'F{f}' for f in folds],
                   yticklabels=[f'F{f}' for f in folds])
        plt.title(f'{study} {mode_label} Gene Consistency (Overlap Rate)\nAvg. Training Samples: {avg_train}', fontsize=11)
        # 输出文件名添加标识
        suffix = 'stage2' if use_stage2 else 'mrmr'
        out_png = f'results/{suffix}_gene_overlap_heatmap_{study}.png'
        os.makedirs('results', exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"\n✅ 热图已保存: {out_png}")
    except Exception as e:
        print(f"无法生成热图: {e}")

    # 4. 保存统计CSV
    rows = []
    for i in range(len(folds)):
        for j in range(i+1, len(folds)):
            fi, fj = folds[i], folds[j]
            inter, jac, rate = calculate_overlap(genes_dict[fi], genes_dict[fj])
            rows.append({
                'Fold_A': fi, 'Fold_B': fj,
                'Intersection': len(inter),
                'Jaccard': jac,
                'Overlap_Rate': rate
            })
    # 输出文件名添加标识
    suffix = 'stage2' if use_stage2 else 'mrmr'
    pd.DataFrame(rows).to_csv(f'results/{study}_{suffix}_overlap_stats.csv', index=False)
    print(f"✅ 统计已保存: results/{study}_{suffix}_overlap_stats.csv")
    
    # 5. 保存所有基因的详细信息
    all_genes_info = []
    for fold in folds:
        for gene in genes_dict[fold]:
            all_genes_info.append({
                'gene': gene,
                'fold': fold
            })
    pd.DataFrame(all_genes_info).to_csv(f'results/{study}_{suffix}_all_genes.csv', index=False)
    print(f"✅ 所有基因列表已保存: results/{study}_{suffix}_all_genes.csv")

def main():
    parser = argparse.ArgumentParser(
        description='比对嵌套CV中MRMR筛选的基因重合度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 比对 mRMR 原始筛选的基因
    python compare_mrmr_gene_signatures.py --study brca
    
    # 比对 Stage2 (PC算法) 精炼后的基因
    python compare_mrmr_gene_signatures.py --study brca --stage2
        """
    )
    parser.add_argument('--study', type=str, required=True, 
                        help='癌种名称 (如: brca, blca)')
    parser.add_argument('--stage2', action='store_true',
                        help='使用 Stage2 (PC算法) 精炼后的基因，而非 mRMR 原始输出')
    args = parser.parse_args()
    compare_mrmr_signatures(args.study, use_stage2=args.stage2)

if __name__ == '__main__':
    main()
