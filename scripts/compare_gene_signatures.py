#!/usr/bin/env python3
"""
基因签名比对工具
对比全局CPCG、嵌套CV和外部签名的基因重合度
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
    file_path = f'preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_{study}/tcga_{study}_M2M3base_0916.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return []
    
    df = pd.read_csv(file_path)
    
    # 获取基因列 (除了OS和索引列)
    gene_cols = [col for col in df.columns if col not in ['OS', 'Unnamed: 0']]
    
    print(f"✓ 全局CPCG基因数: {len(gene_cols)}")
    return set(gene_cols)

def load_nested_cpog_genes(study):
    """加载嵌套CV各折筛选的基因"""
    features_dir = f'features/{study}'
    
    if not os.path.exists(features_dir):
        print(f"❌ 找不到目录: {features_dir}")
        return {}
    
    nested_genes = {}
    
    for fold in range(5):
        gene_file = f'{features_dir}/fold_{fold}_genes.csv'
        
        if not os.path.exists(gene_file):
            print(f"⚠️  找不到: {gene_file}")
            continue
        
        df = pd.read_csv(gene_file)
        nested_genes[fold] = set(df['gene'].tolist())
        
        print(f"✓ 嵌套CV Fold {fold} 基因数: {len(nested_genes[fold])}")
    
    return nested_genes

def load_external_signatures():
    """加载外部签名基因"""
    signatures = {}
    
    # 1. Hallmarks signatures
    hallmarks_file = 'datasets_csv/metadata/hallmarks_signatures.csv'
    if os.path.exists(hallmarks_file):
        df = pd.read_csv(hallmarks_file)
        # 展平所有列的基因
        hallmark_genes = set()
        for col in df.columns:
            hallmark_genes.update(df[col].dropna().tolist())
        signatures['hallmarks'] = hallmark_genes
        print(f"✓ Hallmarks基因数: {len(hallmark_genes)}")
    
    # 2. Combine signatures
    combine_file = 'datasets_csv/metadata/combine_signatures.csv'
    if os.path.exists(combine_file):
        df = pd.read_csv(combine_file)
        combine_genes = set()
        for col in df.columns:
            combine_genes.update(df[col].dropna().tolist())
        signatures['combine'] = combine_genes
        print(f"✓ Combine基因数: {len(combine_genes)}")
    
    # 3. Xena signatures
    xena_file = 'datasets_csv/metadata/xena_signatures.csv'
    if os.path.exists(xena_file):
        df = pd.read_csv(xena_file)
        xena_genes = set()
        for col in df.columns:
            xena_genes.update(df[col].dropna().tolist())
        signatures['xena'] = xena_genes
        print(f"✓ Xena基因数: {len(xena_genes)}")
    
    return signatures

def calculate_overlap(set1, set2):
    """计算两个基因集的交集和重合率"""
    intersection = set1 & set2
    union = set1 | set2
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    overlap_rate = len(intersection) / len(set1) if len(set1) > 0 else 0
    
    return intersection, jaccard, overlap_rate

def compare_signatures(study):
    """比对所有签名的基因重合度"""
    print(f"\n{'='*60}")
    print(f"基因签名比对: {study}")
    print(f"{'='*60}")
    
    # 加载基因集
    global_genes = load_global_cpog_genes(study)
    nested_genes = load_nested_cpog_genes(study)
    external_signatures = load_external_signatures()
    
    if not global_genes:
        print("❌ 无法加载全局CPCG基因")
        return
    
    if not nested_genes:
        print("❌ 无法加载嵌套CV基因")
        return
    
    print(f"\n{'='*60}")
    print("比对结果")
    print(f"{'='*60}")
    
    # 1. 全局CPCG vs 嵌套CV各折
    print("\n📊 1. 全局CPCG vs 嵌套CV各折")
    print("-" * 60)
    
    overlap_matrix = []
    jaccard_matrix = []
    
    for fold in range(5):
        if fold in nested_genes:
            intersection, jaccard, overlap_rate = calculate_overlap(global_genes, nested_genes[fold])
            
            overlap_matrix.append(len(intersection))
            jaccard_matrix.append(jaccard)
            
            print(f"\nFold {fold}:")
            print(f"  全局基因数: {len(global_genes)}")
            print(f"  嵌套基因数: {len(nested_genes[fold])}")
            print(f"  交集基因数: {len(intersection)}")
            print(f"  Jaccard系数: {jaccard:.4f}")
            print(f"  重合率: {overlap_rate:.4f} ({overlap_rate*100:.1f}%)")
            
            # 显示部分共同基因
            common_genes = list(intersection)[:10]
            print(f"  共同基因 (前10个): {', '.join(common_genes)}")
    
    # 计算平均重合度
    if overlap_matrix:
        avg_overlap = np.mean(overlap_matrix)
        avg_jaccard = np.mean(jaccard_matrix)
        print(f"\n平均重合度:")
        print(f"  平均交集基因数: {avg_overlap:.1f}")
        print(f"  平均Jaccard系数: {avg_jaccard:.4f}")
    
    # 2. 全局CPCG vs 外部签名
    print("\n📊 2. 全局CPCG vs 外部签名")
    print("-" * 60)
    
    external_results = {}
    
    for sig_name, sig_genes in external_signatures.items():
        intersection, jaccard, overlap_rate = calculate_overlap(global_genes, sig_genes)
        
        external_results[sig_name] = {
            'intersection': intersection,
            'jaccard': jaccard,
            'overlap_rate': overlap_rate
        }
        
        print(f"\n{sig_name.upper()}:")
        print(f"  外部基因数: {len(sig_genes)}")
        print(f"  交集基因数: {len(intersection)}")
        print(f"  Jaccard系数: {jaccard:.4f}")
        print(f"  重合率: {overlap_rate:.4f} ({overlap_rate*100:.1f}%)")
        
        common_genes = list(intersection)[:10]
        print(f"  共同基因 (前10个): {', '.join(common_genes)}")
    
    # 3. 嵌套CV内部一致性
    print("\n📊 3. 嵌套CV内部一致性 (各折间)")
    print("-" * 60)
    
    folds = sorted(nested_genes.keys())
    consistency_matrix = np.zeros((len(folds), len(folds)))
    
    for i, fold_i in enumerate(folds):
        for j, fold_j in enumerate(folds):
            if i != j:
                intersection, jaccard, overlap_rate = calculate_overlap(
                    nested_genes[fold_i], nested_genes[fold_j]
                )
                consistency_matrix[i, j] = overlap_rate
    
    # 计算平均一致性
    avg_consistency = np.mean(consistency_matrix[consistency_matrix > 0])
    print(f"平均一致性: {avg_consistency:.4f} ({avg_consistency*100:.1f}%)")
    
    # 4. 生成热图
    print("\n📈 生成重合度热图...")
    generate_heatmap(study, global_genes, nested_genes, external_signatures, external_results)
    
    # 5. 保存结果
    save_results(study, global_genes, nested_genes, external_signatures, external_results)
    
    return {
        'global_genes': global_genes,
        'nested_genes': nested_genes,
        'external_signatures': external_signatures,
        'external_results': external_results
    }

def generate_heatmap(study, global_genes, nested_genes, external_signatures, external_results):
    """生成基因重合度热图"""
    plt.figure(figsize=(12, 8))
    
    # 创建矩阵
    labels = ['Global CPCG'] + [f'Nested {i}' for i in range(len(nested_genes))]
    
    if external_signatures:
        labels.extend([sig.upper() for sig in external_signatures.keys()])
    
    matrix = np.zeros((len(labels), len(labels)))
    
    # 填充矩阵
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                matrix[i, j] = 1.0
            elif i == 0 and j > 0:  # Global vs Nested/External
                if j <= len(nested_genes):
                    _, _, overlap = calculate_overlap(global_genes, nested_genes[j-1])
                else:
                    sig_idx = j - len(nested_genes) - 1
                    sig_name = list(external_signatures.keys())[sig_idx]
                    _, _, overlap = calculate_overlap(global_genes, external_signatures[sig_name])
                matrix[i, j] = overlap
            elif j == 0 and i > 0:  # Nested/External vs Global
                if i <= len(nested_genes):
                    _, _, overlap = calculate_overlap(nested_genes[i-1], global_genes)
                else:
                    sig_idx = i - len(nested_genes) - 1
                    sig_name = list(external_signatures.keys())[sig_idx]
                    _, _, overlap = calculate_overlap(external_signatures[sig_name], global_genes)
                matrix[i, j] = overlap
            elif i > 0 and j > 0 and i != j:  # Nested vs Nested/External
                if i <= len(nested_genes) and j <= len(nested_genes):
                    _, _, overlap = calculate_overlap(nested_genes[i-1], nested_genes[j-1])
                elif i <= len(nested_genes):
                    sig_idx = j - len(nested_genes) - 1
                    sig_name = list(external_signatures.keys())[sig_idx]
                    _, _, overlap = calculate_overlap(nested_genes[i-1], external_signatures[sig_name])
                else:
                    sig_idx_i = i - len(nested_genes) - 1
                    sig_idx_j = j - len(nested_genes) - 1
                    sig_name_i = list(external_signatures.keys())[sig_idx_i]
                    sig_name_j = list(external_signatures.keys())[sig_idx_j]
                    _, _, overlap = calculate_overlap(external_signatures[sig_name_i], external_signatures[sig_name_j])
                matrix[i, j] = overlap
    
    # 绘制热图
    sns.heatmap(matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                square=True,
                cbar_kws={'label': 'Overlap Rate'})
    
    plt.title(f'Gene Signature Overlap Heatmap - {study}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片
    output_file = f'results/gene_overlap_heatmap_{study}.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 热图保存到: {output_file}")
    
    plt.close()

def save_results(study, global_genes, nested_genes, external_signatures, external_results):
    """保存比对结果到CSV"""
    os.makedirs('results', exist_ok=True)
    
    # 1. 保存所有基因列表
    all_genes = {
        'global_cpog': sorted(list(global_genes))
    }
    
    for fold, genes in nested_genes.items():
        all_genes[f'nested_fold_{fold}'] = sorted(list(genes))
    
    for sig_name, genes in external_signatures.items():
        all_genes[f'external_{sig_name}'] = sorted(list(genes))
    
    # 保存到CSV
    max_len = max(len(genes) for genes in all_genes.values())
    
    gene_df = pd.DataFrame({k: v + [''] * (max_len - len(v)) for k, v in all_genes.items()})
    gene_df.to_csv(f'results/{study}_all_genes.csv', index=False)
    
    # 2. 保存重合度统计
    overlap_stats = []
    
    # 全局 vs 嵌套各折
    for fold in range(5):
        if fold in nested_genes:
            intersection, jaccard, overlap_rate = calculate_overlap(global_genes, nested_genes[fold])
            overlap_stats.append({
                'signature1': 'global_cpog',
                'signature2': f'nested_fold_{fold}',
                'overlap_count': len(intersection),
                'jaccard': jaccard,
                'overlap_rate': overlap_rate
            })
    
    # 全局 vs 外部签名
    for sig_name, results in external_results.items():
        overlap_stats.append({
            'signature1': 'global_cpog',
            'signature2': f'external_{sig_name}',
            'overlap_count': len(results['intersection']),
            'jaccard': results['jaccard'],
            'overlap_rate': results['overlap_rate']
        })
    
    # 嵌套各折间
    folds = sorted(nested_genes.keys())
    for i in range(len(folds)):
        for j in range(i+1, len(folds)):
            intersection, jaccard, overlap_rate = calculate_overlap(
                nested_genes[folds[i]], nested_genes[folds[j]]
            )
            overlap_stats.append({
                'signature1': f'nested_fold_{folds[i]}',
                'signature2': f'nested_fold_{folds[j]}',
                'overlap_count': len(intersection),
                'jaccard': jaccard,
                'overlap_rate': overlap_rate
            })
    
    overlap_df = pd.DataFrame(overlap_stats)
    overlap_df.to_csv(f'results/{study}_overlap_stats.csv', index=False)
    
    print(f"\n✅ 结果保存到:")
    print(f"   - results/{study}_all_genes.csv")
    print(f"   - results/{study}_overlap_stats.csv")

def main():
    parser = argparse.ArgumentParser(description='Compare gene signatures')
    parser.add_argument('--study', type=str, required=True, 
                        help='Study name (e.g., blca, brca)')
    
    args = parser.parse_args()
    
    results = compare_signatures(args.study)
    
    print(f"\n{'='*60}")
    print("✅ 基因签名比对完成!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
