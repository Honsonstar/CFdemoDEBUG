#!/usr/bin/env python3
"""
使用嵌套CV特征训练模型
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

def train_fold(fold, study, features_dir, splits_dir):
    """训练单个折"""
    
    print(f"\n{'='*70}")
    print(f"训练 Fold {fold}")
    print(f"{'='*70}")
    
    # 读取特征文件
    feature_file = f'{features_dir}/fold_{fold}_features.csv'
    if not os.path.exists(feature_file):
        print(f"❌ 特征文件不存在: {feature_file}")
        return None
    
    features_df = pd.read_csv(feature_file)
    print(f"✅ 加载特征: {features_df.shape}")
    
    # 读取划分
    split_file = f'{splits_dir}/nested_splits_{fold}.csv'
    splits_df = pd.read_csv(split_file)
    
    train_ids = splits_df['train'].dropna().tolist()
    val_ids = splits_df['val'].dropna().tolist()
    test_ids = splits_df['test'].dropna().tolist()
    
    print(f"   训练集: {len(train_ids)} 样本")
    print(f"   验证集: {len(val_ids)} 样本")
    print(f"   测试集: {len(test_ids)} 样本")
    
    # 准备数据
    # 获取基因列
    gene_cols = [col for col in features_df.columns 
                 if col not in ['sample_id', 'OS']]
    
    print(f"   基因数: {len(gene_cols)}")
    
    # 准备训练数据
    train_mask = features_df['sample_id'].isin(train_ids)
    val_mask = features_df['sample_id'].isin(val_ids)
    test_mask = features_df['sample_id'].isin(test_ids)

    X_train = features_df.loc[train_mask, gene_cols].fillna(0).values
    y_train_time = features_df.loc[train_mask, 'OS'].values
    # 创建结构化数组：(event, time)
    y_train = np.array(
        [(True, t) for t in y_train_time],
        dtype=[('event', bool), ('time', float)]
    )

    X_test = features_df.loc[test_mask, gene_cols].fillna(0).values
    y_test_time = features_df.loc[test_mask, 'OS'].values
    y_test = np.array(
        [(True, t) for t in y_test_time],
        dtype=[('event', bool), ('time', float)]
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n🧠 训练Cox模型...")
    
    # 训练Cox模型
    model = CoxnetSurvivalAnalysis(
        l1_ratio=0.5,
        max_iter=1000
    )

    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_test_scaled)

    # 计算C-index
    c_index = concordance_index_censored(
        y_test['event'], y_test['time'], y_pred
    )[0]
    
    print(f"\n✅ Fold {fold} 结果:")
    print(f"   C-index: {c_index:.4f}")
    
    return {
        'fold': fold,
        'c_index': c_index,
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids),
        'n_genes': len(gene_cols)
    }

def main():
    if len(sys.argv) < 2:
        print("用法: python train_nested_cv.py <study>")
        print("示例: python train_nested_cv.py brca")
        sys.exit(1)
    
    study = sys.argv[1]
    features_dir = f'features/{study}'
    splits_dir = f'splits/nested_cv/{study}'
    results_dir = f'results/nested_cv_{study}'
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"使用嵌套CV特征训练模型: {study}")
    print(f"{'='*80}")
    print(f"特征目录: {features_dir}")
    print(f"划分目录: {splits_dir}")
    print(f"结果目录: {results_dir}")
    
    # 训练所有折
    all_results = []
    
    for fold in range(5):
        result = train_fold(fold, study, features_dir, splits_dir)
        if result:
            all_results.append(result)
    
    # 汇总结果
    if all_results:
        print(f"\n{'='*80}")
        print(f"📊 所有折结果汇总")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{results_dir}/summary.csv', index=False)
        
        mean_cindex = np.mean([r['c_index'] for r in all_results])
        std_cindex = np.std([r['c_index'] for r in all_results])
        
        print(f"\n🎯 最终结果 (嵌套CV):")
        print(f"{'='*80}")
        print(f"C-index: {mean_cindex:.4f} ± {std_cindex:.4f}")
        print(f"\n各折详情:")
        for result in all_results:
            print(f"   Fold {result['fold']}: {result['c_index']:.4f} "
                  f"(Train={result['train_size']}, Test={result['test_size']})")
        
        print(f"\n💾 结果保存到: {results_dir}/summary.csv")
        
        print(f"\n{'='*80}")
        print(f"✅ 嵌套CV训练完成!")
        print(f"{'='*80}")
        
        return mean_cindex, std_cindex
    else:
        print("❌ 没有成功的训练结果")
        return None, None

if __name__ == '__main__':
    main()
