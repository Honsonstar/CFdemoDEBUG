import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse
import time

def make_splits(study, seed=None):
    # ================= 配置区域 =================
    # 假设你的 preprocess.py 生成的文件放在 tcga_{study} 文件夹下
    # 例如: tcga_brca/clinical.CSV
    
    base_dir = f"preprocessing/CPCG_algo/raw_data/tcga_{study}" # 或者是你 preprocess.py 中 output_path 的逻辑
    source_csv_path = os.path.join(base_dir, 'clinical.CSV')
    
    # 为了兼容，如果找不到上面那个，再试试 datasets_csv 目录
    if not os.path.exists(source_csv_path):
        # 尝试回退到旧路径逻辑
        source_csv_path = os.path.join('datasets_csv', 'clinical_data', f'tcga_{study}_clinical.csv')

    # 输出目录
    output_root = 'splits/5foldcv_ramdom'
    output_dir = os.path.join(output_root, f'tcga_{study}')

    # ================= 执行逻辑 =================

    if not os.path.exists(source_csv_path):
        print(f"❌ 错误：找不到文件 {source_csv_path}")
        print(f"   请确认 preprocess.py 是否已运行，并生成了 clinical.CSV")
        return

    print(f"📂 读取临床数据: {source_csv_path}")
    df = pd.read_csv(source_csv_path)

    # 1. 自动适配 ID 列名
    id_col = None
    if 'case_submitter_id' in df.columns:
        id_col = 'case_submitter_id'
        print("✅ 检测到预处理后的 ID 列: 'case_submitter_id'")
    elif 'case_id' in df.columns:
        id_col = 'case_id'
        print("✅ 检测到原始 ID 列: 'case_id'")
    else:
        print(f"❌ 错误：找不到 ID 列 (case_id 或 case_submitter_id)")
        print(f"   现有列: {list(df.columns)}")
        return

    patient_ids = df[id_col].values

    # 2. 自动适配生存状态列 (优先使用 Censor)
    target_col = None
    if 'Censor' in df.columns:
        target_col = 'Censor'
        print("✅ 使用修正后的生存状态列: 'Censor' (1=Event, 0=Censored)")
    elif 'censorship' in df.columns:
        target_col = 'censorship'
        print("⚠️ 警告: 使用原始 'censorship' 列。请确认其定义 (通常需反转)。")
    elif 'censorship_dss' in df.columns:
        target_col = 'censorship_dss'
        print("✅ 使用 'censorship_dss' 列")
    
    # 3. 清洗缺失标签
    if target_col:
        # 检查 NaN
        nan_count = df[target_col].isna().sum()
        if nan_count > 0:
            print(f"⚠️  发现 {nan_count} 个样本缺失标签 ({target_col})，正在剔除...")
            original_len = len(df)
            df = df.dropna(subset=[target_col])
            # 更新 patient_ids
            patient_ids = df[id_col].values
            print(f"    -> 剔除后剩余: {len(df)} (原: {original_len})")
        
        labels = df[target_col].values
    else:
        print("⚠️ 未找到生存标签列，退回随机划分。")
        labels = np.zeros(len(df))

    # 4. 生成划分
    if seed is None:
        seed = int(time.time()) % 10000 
        print(f"🎲 未指定种子，使用随机种子: {seed}")
    else:
        print(f"🔒 使用固定种子: {seed}")

    os.makedirs(output_dir, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print(f"🚀 开始生成划分文件...")

    try:
        for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
            train_ids = patient_ids[train_idx]
            test_ids = patient_ids[test_idx]
            
            max_len = max(len(train_ids), len(test_ids))
            train_col = list(train_ids) + [''] * (max_len - len(train_ids))
            test_col = list(test_ids) + [''] * (max_len - len(test_ids))
            
            split_df = pd.DataFrame({'train': train_col, 'test': test_col})
            
            save_path = os.path.join(output_dir, f'splits_{fold}.csv')
            split_df.to_csv(save_path, index=False)
            print(f"   💾 已保存: {save_path} (Train: {len(train_ids)}, Val: {len(test_ids)})")
            
        print(f"\n🎉 splits 文件夹已更新。")
        
    except ValueError as e:
        print(f"\n❌ 切分失败: {e}")
        print("   原因可能是某类样本太少(Event数 < 5)。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-Fold Splits")
    parser.add_argument('--study', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    make_splits(args.study, args.seed)