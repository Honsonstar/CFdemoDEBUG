"""
CGI 数据预处理 - 测试集版本 + 5折交叉验证
1. 输出所有死亡样本的数据到 data 文件夹
2. 基于所有死亡样本进行5折划分
3. 输出训练集对应的 .mat 文件到 splits/CGI_nested_cv/{cancer_type}/
"""
import os
import pandas as pd
import numpy as np
import scipy.io
from sklearn.model_selection import StratifiedKFold, train_test_split

# =====================
# 1. 路径配置 (修改这里)
# =====================
# 输入路径
cancer_type = 'stad'  # 癌症类型: 'blca', 'brca', 'coadread', 'hnsc', 'stad'
# 5折划分配置
n_splits = 5
cv_seed = 42

# 基础路径
base_path = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy'
clinical_input_path = f'{base_path}/datasets_csv/metadata/tcga_{cancer_type}.csv'
rna_input_path = f'{base_path}/datasets_csv/raw_rna_data/combine/{cancer_type}/rna_clean.csv'

# 输出路径
data_output_path = f'{base_path}/preprocessing/CGI/data'
splits_output_path = f'{base_path}/splits/CGI_nested_cv/{cancer_type}'

print(f"当前处理癌症类型: {cancer_type}")
print(f"数据输出文件夹: {data_output_path}")
print(f"划分输出文件夹: {splits_output_path}")
print(f"交叉验证折数: {n_splits}")

# 确保输出目录存在
os.makedirs(data_output_path, exist_ok=True)
os.makedirs(splits_output_path, exist_ok=True)

# =====================
# 2. 处理临床数据 (DSS + 所有死亡样本)
# =====================
print("\n正在处理临床数据...")
df_clinical_raw = pd.read_csv(clinical_input_path)

# 只提取 case_id 用于对齐，以及 DSS 生存数据
clinical_cols = ['case_id', 'survival_months_dss', 'censorship_dss', 'train']
clinical_data = df_clinical_raw[clinical_cols].copy()

# 只保留死亡样本 (event=1)
clinical_data = clinical_data[clinical_data['censorship_dss'] == 1].copy()
clinical_data = clinical_data[['case_id', 'survival_months_dss']]
clinical_data.columns = ['patient_id', 'time']

print(f"所有死亡样本数: {len(clinical_data)}")

# =====================
# 3. 处理基因表达数据
# =====================
print("\n正在处理基因数据...")
df_rna_raw = pd.read_csv(rna_input_path, header=0)

# 将第一列命名为 case_id
first_col_name = df_rna_raw.columns[0]
df_rna_raw.rename(columns={first_col_name: 'case_id'}, inplace=True)

# 设置 case_id 为索引
df_rna_raw.set_index('case_id', inplace=True)

# 重命名基因列
df_rna_raw.columns = [c.replace('_rnaseq', '') for c in df_rna_raw.columns]

print(f"基因数量: {len(df_rna_raw.columns)}")
print(f"基因数据病人数: {len(df_rna_raw)}")

# =====================
# 4. 对齐数据 (保留所有死亡样本)
# =====================
print("\n正在对齐数据...")

# 获取共同的病人ID
clinical_patients = set(clinical_data['patient_id'])
rna_patients = set(df_rna_raw.index)
common_patients = list(clinical_patients & rna_patients)

print(f"临床死亡样本数: {len(clinical_patients)}")
print(f"基因数据病人数: {len(rna_patients)}")

# 移除重复病人ID，保留第一条记录
clinical_data_dedup = clinical_data.drop_duplicates(subset=['patient_id'], keep='first')
df_rna_dedup = df_rna_raw[~df_rna_raw.index.duplicated(keep='first')]

print(f"临床去重后样本数: {len(clinical_data_dedup)}")
print(f"RNA去重后样本数: {len(df_rna_dedup)}")

# 重新获取共同病人
clinical_patients_dedup = set(clinical_data_dedup['patient_id'])
rna_patients_dedup = set(df_rna_dedup.index)
common_patients = list(clinical_patients_dedup & rna_patients_dedup)
print(f"共同病人数（去重后）: {len(common_patients)}")

# 按共同病人列表排序
clinical_data_filtered = clinical_data_dedup[clinical_data_dedup['patient_id'].isin(common_patients)]
clinical_data_filtered = clinical_data_filtered.set_index('patient_id')
clinical_data_filtered = clinical_data_filtered.loc[common_patients].reset_index()

df_rna_filtered = df_rna_dedup.loc[common_patients].reset_index()  # 保留 patient_id 列

print(f"过滤后样本数: {len(df_rna_filtered)}")

# =====================
# 5. 合并数据
# =====================
print("\n正在合并数据...")

# df_rna_filtered 现在有 'case_id' 列和所有基因列
gene_cols = [c for c in df_rna_filtered.columns if c != 'case_id']

# 保存 patient_id 用于后续划分
patient_ids_all = df_rna_filtered['case_id'].values

# 合并：patient_id + 基因 + time
output_data_with_id = df_rna_filtered[['case_id'] + gene_cols].copy()
output_data_with_id['time'] = clinical_data_filtered['time'].values
output_data_with_id = output_data_with_id.rename(columns={'case_id': 'patient_id'})

# 输出版本：不包含 patient_id (保持 CGI 格式)
output_data = output_data_with_id[['patient_id'] + gene_cols + ['time']].copy()
# 移除 patient_id 用于最终输出
output_data_for_save = output_data.drop(columns=['patient_id'])

# 确保 time 是最后一列
cols = gene_cols + ['time']
output_data_for_save = output_data_for_save[cols]

print(f"输出数据维度: {output_data_for_save.shape}")

# =====================
# 6. 输出数据文件到 data 文件夹
# =====================
# 6.1 保存含表头的CSV（不含patient_id，保持CGI格式）
save_data_path = os.path.join(data_output_path, f'{cancer_type}_data.csv')
output_data_for_save.to_csv(save_data_path, index=False)
print(f"\n成功: 含表头数据已保存至 {save_data_path}")

# 6.2 保存含病人ID的CSV（用于后续特征提取验证）
save_data_with_id_path = os.path.join(data_output_path, f'{cancer_type}_data_with_id.csv')
output_data_with_id.to_csv(save_data_with_id_path, index=False)
print(f"成功: 含病人ID数据已保存至 {save_data_with_id_path}")

# 6.3 保存不含表头的纯数值矩阵
save_raw_path = os.path.join(data_output_path, f'{cancer_type}_data_raw.csv')
output_data_for_save.to_csv(save_raw_path, index=False, header=False)
print(f"成功: 纯数值矩阵已保存至 {save_raw_path}")

# 6.4 保存 .mat 文件
mat_path = os.path.join(data_output_path, f'{cancer_type}_data.mat')
scipy.io.savemat(mat_path, {'d': output_data_for_save.values})
print(f"成功: .mat 文件已保存至 {mat_path}")

# =====================
# 7. 5折交叉验证划分
# =====================
print(f"\n正在进行 {n_splits} 折交叉验证划分...")

# 使用 time 作为分层标签 (这里简化处理，用中位数分层)
times = output_data['time'].values
stratify_labels = (times > np.median(times)).astype(int)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cv_seed)

# 使用 patient_id 进行划分
patient_ids = patient_ids_all

for fold, (train_val_idx, test_idx) in enumerate(skf.split(patient_ids, stratify_labels)):
    print(f"\n Fold {fold}...")

    train_val_ids = patient_ids[train_val_idx]
    test_ids = patient_ids[test_idx]
    train_val_labels = stratify_labels[train_val_idx]

    # 划分 Train/Val (85% / 15%)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ids)),
        test_size=0.15,
        stratify=train_val_labels,
        random_state=cv_seed
    )

    train_ids = train_val_ids[train_idx]
    val_ids = train_val_ids[val_idx]

    print(f"   Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # 保存划分结果 CSV
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    train_col = list(train_ids) + [np.nan] * (max_len - len(train_ids))
    val_col = list(val_ids) + [np.nan] * (max_len - len(val_ids))
    test_col = list(test_ids) + [np.nan] * (max_len - len(test_ids))

    split_df = pd.DataFrame({
        'train': train_col,
        'val': val_col,
        'test': test_col
    })

    split_path = os.path.join(splits_output_path, f"nested_splits_{fold}.csv")
    split_df.to_csv(split_path, index=False)
    print(f"   保存划分至: {split_path}")

    # 输出训练集 .mat 文件 (不包含 patient_id)
    train_mask = output_data['patient_id'].isin(train_ids)
    train_data = output_data[train_mask].drop(columns=['patient_id'])

    train_mat_path = os.path.join(splits_output_path, f"train_fold{fold}.mat")
    scipy.io.savemat(train_mat_path, {'d': train_data.values})
    print(f"   保存训练集至: {train_mat_path}")

# 保存汇总文件
summary_data = []
for fold in range(n_splits):
    f_path = os.path.join(splits_output_path, f"nested_splits_{fold}.csv")
    if os.path.exists(f_path):
        d = pd.read_csv(f_path)
        summary_data.append({
            'fold': fold,
            'train': d['train'].notna().sum(),
            'val': d['val'].notna().sum(),
            'test': d['test'].notna().sum()
        })

summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(splits_output_path, 'summary.csv')
summary_df.to_csv(summary_path, index=False)

print(f"\n" + "="*50)
print("预处理完成！")
print("="*50)
print(f"保留样本数: {len(output_data_for_save)}")
print(f"保留基因数: {len(gene_cols)}")
print(f"\n数据文件 (data 文件夹):")
print(f"  - {save_data_path}")
print(f"  - {save_raw_path}")
print(f"  - {mat_path}")
print(f"\n划分文件 (splits/CGI_nested_cv/{cancer_type}/):")
print(f"  - {splits_output_path}/nested_splits_0~{n_splits-1}.csv")
print(f"  - {splits_output_path}/train_fold0~{n_splits-1}.mat")
print(f"  - {summary_path}")
