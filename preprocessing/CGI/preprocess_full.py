"""
CGI 数据预处理 - 全集版本（适配嵌套交叉验证）
1. 保留所有符合条件的样本（不过滤死亡状态）
2. 直接读取 nested_cv 目录下的分割文件
3. 输出每折训练集对应的 .mat 文件
"""
import os
import pandas as pd
import numpy as np
import scipy.io

# =====================
# 1. 参数设置
# =====================
cancer_type = 'brca'  # 可修改: 'blca', 'brca', 'coadread', 'hnsc', 'stad'
n_splits = 5

# 基础路径
base_path = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy'

# 输入文件路径
clinical_input_path = f'{base_path}/datasets_csv/metadata/tcga_{cancer_type}.csv'
rna_input_path = f'{base_path}/datasets_csv/raw_rna_data/combine/{cancer_type}/rna_clean.csv'

# 分割文件输入路径（从 nested_cv 读取）
splits_input_path = f'{base_path}/splits/nested_cv/{cancer_type}'

# 输出文件夹（保存每折训练集 .mat 文件）
splits_output_path = f'{base_path}/splits/nested_cv/{cancer_type}'

print(f"当前处理癌症类型: {cancer_type}")
print(f"分割文件输入路径: {splits_input_path}")
print(f"输出文件夹: {splits_output_path}")
print(f"交叉验证折数: {n_splits}")

# 确保输出目录存在
os.makedirs(splits_output_path, exist_ok=True)

# =====================
# 2. 处理临床数据（保留所有样本，不过滤死亡状态）
# =====================
print("\n正在处理临床数据...")
df_clinical_raw = pd.read_csv(clinical_input_path)

# 提取 case_id 和 DSS 生存数据
clinical_cols = ['case_id', 'survival_months_dss', 'censorship_dss']
clinical_data = df_clinical_raw[clinical_cols].copy()

# 【修改】移除死亡样本过滤逻辑，保留所有样本
# 原来: clinical_data = clinical_data[clinical_data['censorship_dss'] == 1].copy()
clinical_data = clinical_data[['case_id', 'survival_months_dss', 'censorship_dss']]
clinical_data.columns = ['patient_id', 'time', 'censorship']

print(f"临床数据样本数: {len(clinical_data)}")

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

# 重命名基因列（去除可能的 _rnaseq 后缀）
df_rna_raw.columns = [c.replace('_rnaseq', '') for c in df_rna_raw.columns]

print(f"基因数量: {len(df_rna_raw.columns)}")
print(f"基因数据病人数: {len(df_rna_raw)}")

# =====================
# 4. 对齐数据（保留所有样本）
# =====================
print("\n正在对齐数据...")

# 获取共同的病人ID
clinical_patients = set(clinical_data['patient_id'])
rna_patients = set(df_rna_raw.index)
common_patients = list(clinical_patients & rna_patients)

print(f"临床数据病人数: {len(clinical_patients)}")
print(f"基因数据病人数: {len(rna_patients)}")
print(f"共同病人数: {len(common_patients)}")

# 移除重复病人ID，保留第一条记录
clinical_data_dedup = clinical_data.drop_duplicates(subset=['patient_id'], keep='first')
df_rna_dedup = df_rna_raw[~df_rna_raw.index.duplicated(keep='first')]

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

gene_cols = [c for c in df_rna_filtered.columns if c != 'case_id']

# 保存 patient_id 用于后续匹配
patient_ids_all = df_rna_filtered['case_id'].values

# 合并：patient_id + 基因 + time + censorship
output_data_with_id = df_rna_filtered[['case_id'] + gene_cols].copy()
output_data_with_id['time'] = clinical_data_filtered['time'].values
output_data_with_id['censorship'] = clinical_data_filtered['censorship'].values
output_data_with_id = output_data_with_id.rename(columns={'case_id': 'patient_id'})

print(f"输出数据维度: {output_data_with_id.shape}")

# =====================
# 6. 读取 nested_cv 分割文件并输出训练集
# =====================
print(f"\n正在读取 nested_cv 分割文件并输出训练集...")

# 构建病人ID到数据行的映射
patient_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_all)}

summary_data = []

for fold in range(n_splits):
    print(f"\n Fold {fold}...")

    # 读取分割文件
    split_file = os.path.join(splits_input_path, f'nested_splits_{fold}.csv')

    if not os.path.exists(split_file):
        print(f"   警告: 分割文件不存在: {split_file}")
        continue

    split_df = pd.read_csv(split_file)

    # 获取训练集样本ID
    train_ids = split_df['train'].dropna().tolist()

    print(f"   分割文件中训练集样本数: {len(train_ids)}")

    # 过滤出训练集样本
    train_data_list = []
    matched_count = 0
    for pid in train_ids:
        if pid in patient_to_idx:
            idx = patient_to_idx[pid]
            train_data_list.append(output_data_with_id.iloc[idx])
            matched_count += 1

    print(f"   实际匹配到的训练集样本数: {matched_count}")

    if matched_count == 0:
        print(f"   警告: Fold {fold} 没有匹配到任何样本，跳过")
        continue

    # 构建训练集 DataFrame（不含 patient_id 和 censorship，仅基因 + time）
    train_data_df = pd.DataFrame(train_data_list)

    # 保存训练集 .mat 文件（变量名: d，格式：行为样本，列为基因+time）
    # 移除 patient_id 和 censorship，保留基因和 time
    train_data_for_save = train_data_df.drop(columns=['patient_id', 'censorship'])

    # 确保 time 是最后一列
    cols = gene_cols + ['time']
    train_data_for_save = train_data_for_save[cols]

    train_mat_path = os.path.join(splits_output_path, f'train_fold{fold}.mat')
    scipy.io.savemat(train_mat_path, {'d': train_data_for_save.values})
    print(f"   保存训练集至: {train_mat_path}")
    print(f"   训练集维度: {train_data_for_save.shape}")

    # 记录汇总信息
    summary_data.append({
        'fold': fold,
        'train': matched_count
    })

# 保存汇总文件
summary_df = pd.DataFrame(summary_data)
summary_path = os.path.join(splits_output_path, 'summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n保存汇总文件至: {summary_path}")

# =====================
# 7. 输出完成
# =====================
print(f"\n" + "="*50)
print("预处理完成！")
print("="*50)
print(f"总样本数: {len(output_data_with_id)}")
print(f"保留基因数: {len(gene_cols)}")
print(f"\n输出文件 (splits/CGI_nested_cv/{cancer_type}/):")
print(f"  - train_fold0~{n_splits-1}.mat")
print(f"  - summary.csv")
