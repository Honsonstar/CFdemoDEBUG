#!/usr/bin/env python3
"""
运行嵌套CV的CPCG特征筛选
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.CPCG_algo.nested_cv_wrapper import NestedCVFeatureSelector

def run_cpog_for_fold(study, fold, splits_file, data_root_dir):
    """为指定折运行CPCG筛选"""
    
    print(f"\n{'='*60}")
    print(f"[{study}] Fold {fold}: 开始CPCG特征筛选")
    print(f"{'='*60}")
    
    # 读取划分文件
    splits_df = pd.read_csv(splits_file)
    train_ids = splits_df['train'].dropna().astype(str).tolist()
    val_ids = splits_df['val'].dropna().astype(str).tolist()
    test_ids = splits_df['test'].dropna().astype(str).tolist()
    
    print(f"\n📊 数据统计:")
    print(f"   训练集: {len(train_ids)} 样本")
    print(f"   验证集: {len(val_ids)} 样本")
    print(f"   测试集: {len(test_ids)} 样本")
    
    # 使用NestedCVFeatureSelector运行CPCG
    with NestedCVFeatureSelector(study, data_root_dir, threshold=50) as selector:
        # 为该折筛选特征
        feature_file = selector.select_features_for_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids
        )
        
        # 复制到features目录
        features_dir = f'features/{study}'
        os.makedirs(features_dir, exist_ok=True)
        
        final_file = f'{features_dir}/fold_{fold}_features.csv'
        import shutil
        shutil.copy2(feature_file, final_file)
        
        print(f"\n✅ 特征筛选完成!")
        print(f"   输出文件: {final_file}")
        
        # 读取并显示基因信息
        result_df = pd.read_csv(final_file)
        gene_cols = [col for col in result_df.columns if col not in ['Unnamed: 0', 'OS']]
        
        print(f"   筛选基因数: {len(gene_cols)}")
        print(f"   样本数: {len(result_df)}")
        
        if len(gene_cols) > 0:
            print(f"\n📋 前10个基因:")
            for i, gene in enumerate(gene_cols[:10]):
                print(f"   {i+1}. {gene}")
        
        return final_file, gene_cols

def main():
    if len(sys.argv) < 2:
        print("用法: python run_cpog_nested_cv.py <study> [fold]")
        print("示例: python run_cpog_nested_cv.py brca 0")
        sys.exit(1)
    
    study = sys.argv[1]
    fold = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    splits_file = f'splits/nested_cv/{study}/nested_splits_{fold}.csv'
    data_root_dir = 'preprocessing/CPCG_algo/raw_data'
    
    if not os.path.exists(splits_file):
        print(f"❌ 错误: 找不到划分文件 {splits_file}")
        print(f"请先运行: python create_nested_splits_temp3.py {study}")
        sys.exit(1)
    
    if not os.path.exists(data_root_dir):
        print(f"❌ 错误: 找不到数据目录 {data_root_dir}")
        sys.exit(1)
    
    feature_file, genes = run_cpog_for_fold(study, fold, splits_file, data_root_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ 折 {fold} CPCG筛选完成!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
