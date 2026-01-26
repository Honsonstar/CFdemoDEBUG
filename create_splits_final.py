import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

# ================= 配置 =================
study = 'blca'
# 这里的路径指向我们刚才"鸠占鹊巢"后确认正确的那个全量文件
source_csv_path = 'datasets_csv/clinical_data/tcga_blca_clinical.csv'
output_dir = 'splits/5foldcv/tcga_blca'

# ================= 逻辑 =================
if not os.path.exists(source_csv_path):
    print(f"❌ 找不到文件: {source_csv_path}")
    exit()

print(f"📂 读取数据: {source_csv_path}")
df = pd.read_csv(source_csv_path)

# 确保索引列正确（清洗 Unnamed）
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# 获取 ID 和 标签
# 此时第一列应该是 case_id
if 'case_id' in df.columns:
    ids = df['case_id'].values
else:
    ids = df.iloc[:, 0].values

# 获取生存状态用于分层
if 'censorship' in df.columns:
    labels = df['censorship'].values
elif 'censorship_dss' in df.columns:
    labels = df['censorship_dss'].values
else:
    print("⚠️ 没找到 censorship 列，随机划分")
    labels = np.zeros(len(ids))

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 5折交叉验证 (Outer Loop: Train+Val vs Test)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"🚀 正在生成 Train / Val / Test 划分...")

for fold, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
    train_val_ids = ids[train_val_idx]
    train_val_labels = labels[train_val_idx]
    test_ids = ids[test_idx]
    
    # Inner Loop: 从 Train+Val 中划分出 15% 作为 Val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ids)), 
        test_size=0.15, 
        stratify=train_val_labels, 
        random_state=42
    )
    
    train_ids = train_val_ids[train_idx]
    val_ids = train_val_ids[val_idx]
    
    # 补齐长度以便保存为CSV (DataFrame列长必须一致)
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    
    train_col = list(train_ids) + [''] * (max_len - len(train_ids))
    val_col   = list(val_ids)   + [''] * (max_len - len(val_ids))
    test_col  = list(test_ids)  + [''] * (max_len - len(test_ids))
    
    split_df = pd.DataFrame({
        'train': train_col,
        'val': val_col,
        'test': test_col
    })
    
    save_path = os.path.join(output_dir, f'splits_{fold}.csv')
    split_df.to_csv(save_path, index=False)
    print(f"   💾 Fold {fold}: Saved (Train:{len(train_ids)}, Val:{len(val_ids)}, Test:{len(test_ids)})")

print("\n🎉 完美！包含 val 列的 splits 文件已生成！")