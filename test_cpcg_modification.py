#!/usr/bin/env python3
"""
测试修改后的CPCG算法
验证完整的Stage1+Stage2流程是否正常工作
"""
import os
import sys
import pandas as pd
import numpy as np

# 添加CPCG路径
sys.path.insert(0, 'preprocessing/CPCG_algo')

from nested_cv_wrapper import NestedCVFeatureSelector

def test_single_fold_cpcg(study='blca', fold=0):
    """测试单折CPCG筛选"""
    print("=" * 60)
    print(f"测试修改后的CPCG算法 - {study.upper()} Fold {fold}")
    print("=" * 60)

    # 读取划分文件
    splits_file = f'splits/nested_cv/{study}/nested_splits_{fold}.csv'
    if not os.path.exists(splits_file):
        print(f"❌ 错误: 找不到划分文件 {splits_file}")
        return False

    splits_df = pd.read_csv(splits_file)
    train_ids = splits_df['train'].dropna().tolist()
    val_ids = splits_df['val'].dropna().tolist()
    test_ids = splits_df['test'].dropna().tolist()

    print(f"\n📊 数据划分:")
    print(f"  训练集: {len(train_ids)} 样本")
    print(f"  验证集: {len(val_ids)} 样本")
    print(f"  测试集: {len(test_ids)} 样本")

    # 创建特征选择器
    data_root_dir = 'preprocessing/CPCG_algo/raw_data'
    selector = NestedCVFeatureSelector(
        study=study,
        data_root_dir=data_root_dir,
        threshold=100,
        n_jobs=-1
    )

    try:
        # 运行特征筛选
        with selector:
            feature_file = selector.select_features_for_fold(
                fold=fold,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids
            )

            print(f"\n✅ 测试完成!")
            print(f"  输出文件: {feature_file}")

            # 验证输出文件
            if os.path.exists(feature_file):
                df = pd.read_csv(feature_file)
                print(f"  基因数量: {df.shape[1] - 1}")  # 减去OS列
                print(f"  样本数量: {df.shape[0]}")

                # 显示前5个基因
                gene_cols = [col for col in df.columns if col != 'OS']
                print(f"\n📋 前5个基因: {gene_cols[:5]}")

                return True
            else:
                print(f"❌ 错误: 文件未生成 {feature_file}")
                return False

    except Exception as e:
        print(f"❌ 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_fold_cpcg(study='blca', fold=0)

    if success:
        print("\n" + "=" * 60)
        print("✅ 测试通过! CPCG算法修改成功")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败! 请检查错误")
        print("=" * 60)
        sys.exit(1)
