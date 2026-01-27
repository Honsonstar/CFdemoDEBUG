#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd
import shutil
import traceback

# 确保能导入 preprocessing 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocessing.CPCG_algo.nested_cv_wrapper import NestedCVFeatureSelector
except ImportError:
    print("❌ Error: 无法导入 NestedCVFeatureSelector，请检查文件路径")
    sys.exit(1)

def get_column_data(df, candidates, required=True):
    """尝试从多个候选列名中找到存在的列"""
    for col in candidates:
        if col in df.columns:
            print(f"   Using column: '{col}'")
            return df[col].dropna().astype(str).tolist()
    
    if required:
        print(f"❌ Error: 找不到列，尝试过: {candidates}")
        print(f"   CSV 实际包含列: {df.columns.tolist()}")
        raise ValueError(f"Missing required column (candidates: {candidates})")
    return []

def main():
    parser = argparse.ArgumentParser(description='Run CPCG Nested CV')
    parser.add_argument('--study', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--split_file', type=str, required=True)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"🚀 [Python] {args.study} Fold {args.fold} CPCG 筛选")
    print(f"📂 划分文件: {args.split_file}")
    print(f"{'='*60}")

    if not os.path.exists(args.split_file):
        print(f"❌ 错误: 找不到划分文件 {args.split_file}")
        sys.exit(1)
        
    data_root_dir = 'preprocessing/CPCG_algo/raw_data'
    if not os.path.exists(data_root_dir):
        print(f"❌ 错误: 找不到 CPCG 数据目录 {data_root_dir}")
        sys.exit(1)

    try:
        splits_df = pd.read_csv(args.split_file)
        
        # 智能识别列名
        train_ids = get_column_data(splits_df, ['train', 'train_idx', 'train_ids', 'Train', 'training'])
        val_ids = get_column_data(splits_df, ['val', 'val_idx', 'val_ids', 'Validation', 'validation'], required=False)
        test_ids = get_column_data(splits_df, ['test', 'test_idx', 'test_ids', 'Test', 'testing'], required=False)
        
        print(f"📊 样本统计: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        
        # 确保有训练数据
        if len(train_ids) == 0:
             raise ValueError("训练集为空！")

        # 运行 CPCG (threshold=100 保证特征数量)
        selector = NestedCVFeatureSelector(args.study, data_root_dir, threshold=100)
        
        # 【关键修复】使用 with 语句管理生命周期 (temp_dir 创建与清理)
        with selector:
            temp_feature_file = selector.select_features_for_fold(
                fold=args.fold,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids
            )
            
            features_dir = f'features/{args.study}'
            os.makedirs(features_dir, exist_ok=True)
            final_file = f'{features_dir}/fold_{args.fold}_genes.csv'
            
            # 必须在 with 块内复制文件，因为退出块后临时目录会被删除
            shutil.copy2(temp_feature_file, final_file)
            print(f"\n✅ Fold {args.fold} 完成! 文件已保存至: {final_file}")
            
            # 验证结果
            res_df = pd.read_csv(final_file)
            feat_count = max(0, len(res_df.columns) - 2)
            print(f"   最终筛选基因数: {feat_count}")
            
    except Exception as e:
        print(f"\n❌ CPCG 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
