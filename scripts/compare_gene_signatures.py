#!/usr/bin/env python3
"""
基因签名比对工具 [优先读取 Fresh Global 修复版]
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
    # 【修复点】调整路径优先级：results/comparison 下的新文件优先！
    paths = [
        f'results/comparison/{study}/global_genes.csv',  # <--- 1. 优先读取 run_fresh_global.sh 生成的文件
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
        
        try:
            df = pd.read_csv(gene_file)

            # 【关键修复】基因名在行中（第一列，列名为 'gene_name'），不是列名
            # 需要读取第一列的值作为基因列表
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

            if not genes:
                print(f"⚠️ Fold {fold} 文件为空或无基因列")
            else:
                nested_genes[fold] = set(genes)
                print(f"✓ 嵌套CV Fold {fold} 基因数: {len(nested_genes[fold])}")
                
        except Exception as e:
            print(f"❌ 读取 Fold {fold} 出错: {e}")
    
    return nested_genes

def load_external_signatures():
    """加载外部签名基因"""
    signatures = {}
    
    base_dir = 'datasets_csv/metadata'
    files = {
        'hallmarks': 'hallmarks_signatures.csv',
        'combine': 'combine_signatures.csv',
        'xena': 'xena_signatures.csv'
    }
    
    for name, fname in files.items():
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                genes = set()
                for col in df.columns:
                    genes.update(df[col].dropna().astype(str).tolist())
                signatures[name] = genes
                print(f"✓ {name.capitalize()} 基因数: {len(genes)}")
            except:
                pass
    
    return signatures

def calculate_overlap(set1, set2):
    """计算两个基因集的交集和重合率"""
    if not set1 or not set2:
        return set(), 0.0, 0.0
        
    intersection = set1 & set2
    union = set1 | set2
    
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    # Overlap rate relative to the smaller set size (usually set1 is global or fold i)
    overlap_rate = len(intersection) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
    
    return intersection, jaccard, overlap_rate

def compare_signatures(study):
    """比对所有签名的基因重合度"""
    print(f"\n{'='*60}")
    print(f"基因签名比对: {study}")
    print(f"{'='*60}")
    
    nested_genes = load_nested_cpog_genes(study)
    global_genes = load_global_cpog_genes(study)
    external_signatures = load_external_signatures()
    
    if not nested_genes:
        print("❌ 无法加载任何嵌套CV基因，请检查 features/ 目录")
        return

    # 1. 嵌套CV内部一致性 (这是重点)
    print("\n📊 1. 嵌套CV 内部稳定性 (Fold间重合度)")
    print("-" * 60)
    
    folds = sorted(nested_genes.keys())
    matrix = np.zeros((len(folds), len(folds)))
    
    consistency_scores = []
    
    # 表头
    print(f"{'Folds':<10} | {'交集数':<8} | {'Jaccard':<8} | {'重合率(%)'}")
    print("-" * 50)
    
    for i in range(len(folds)):
        for j in range(len(folds)):
            f_i, f_j = folds[i], folds[j]
            inter, jac, rate = calculate_overlap(nested_genes[f_i], nested_genes[f_j])
            matrix[i, j] = jac
            
            if i < j:
                consistency_scores.append(jac)
                print(f"{f_i} vs {f_j:<4} | {len(inter):<8} | {jac:.4f}   | {rate*100:.1f}%")

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    print("-" * 50)
    print(f"👉 平均一致性 (Jaccard): {avg_consistency:.4f}")
    
    # 2. 全局 vs 嵌套 (如果有全局结果)
    if global_genes:
        print("\n📊 2. 全局CPCG vs 嵌套CV各折")
        print("-" * 60)
        for fold in folds:
            inter, jac, rate = calculate_overlap(global_genes, nested_genes[fold])
            print(f"Global vs Fold {fold}: 交集 {len(inter)} 个 (重合率 {rate*100:.1f}%)")
    
    # 3. 生成热图
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=[f'F{f}' for f in folds], 
                   yticklabels=[f'F{f}' for f in folds])
        plt.title(f'{study} Nested CV Consistency (Jaccard Index)')
        out_png = f'results/gene_overlap_heatmap_{study}.png'
        os.makedirs('results', exist_ok=True)
        plt.savefig(out_png)
        print(f"\n✅ 热图已保存: {out_png}")
    except Exception as e:
        print(f"无法生成热图: {e}")

    # 4. 保存统计CSV
    rows = []
    for i in range(len(folds)):
        for j in range(i+1, len(folds)):
            fi, fj = folds[i], folds[j]
            inter, jac, rate = calculate_overlap(nested_genes[fi], nested_genes[fj])
            rows.append({
                'Fold_A': fi, 'Fold_B': fj, 
                'Intersection': len(inter), 
                'Jaccard': jac, 
                'Overlap_Rate': rate
            })
    pd.DataFrame(rows).to_csv(f'results/{study}_overlap_stats.csv', index=False)
    print(f"✅ 统计已保存: results/{study}_overlap_stats.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=str, required=True)
    args = parser.parse_args()
    compare_signatures(args.study)

if __name__ == '__main__':
    main()
