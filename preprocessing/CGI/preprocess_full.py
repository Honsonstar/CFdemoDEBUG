"""
CGI 数据预处理 - 全集版本
只保留基因信息和DSS生存时间（死亡样本），生存时间在最后一列
输出格式：行为样本，列为基因
"""
import os
import pandas as pd
import numpy as np

# -----------------------
# 1. 参数设置
# -----------------------
cancer_type = 'blca'  # 可修改: 'blca', 'brca', 'coadread', 'hnsc', 'stad'

# 输入文件路径
clinical_input_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv/metadata/tcga_{cancer_type}.csv'
rna_input_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv/raw_rna_data/combine/{cancer_type}/rna_clean.csv'

# 输出文件夹
output_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/features/CGI_tcga_{cancer_type}_full'
print(f"当前处理癌症类型: {cancer_type}")
print(f"输出文件夹: {output_path}")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# -----------------------
# 2. 处理临床数据 (仅DSS + 只保留死亡样本)
# -----------------------
print("正在处理临床数据...")
df_clinical_raw = pd.read_csv(clinical_input_path)

# 只提取 case_id 用于对齐，以及 DSS 生存数据
clinical_cols = ['case_id', 'survival_months_dss', 'censorship_dss']
clinical_data = df_clinical_raw[clinical_cols].copy()

# 只保留死亡样本 (event=1)
clinical_data = clinical_data[clinical_data['censorship_dss'] == 1].copy()
clinical_data = clinical_data[['case_id', 'survival_months_dss']]
clinical_data.columns = ['patient_id', 'time']

print(f"死亡样本数: {len(clinical_data)}")

# -----------------------
# 3. 处理基因表达数据
# -----------------------
print("正在处理基因数据...")
# RNA数据第一列是病人ID但无列名，用header=None读取
df_rna_raw = pd.read_csv(rna_input_path, header=0)

# 将第一列命名为 case_id（与临床数据对齐）
first_col_name = df_rna_raw.columns[0]
df_rna_raw.rename(columns={first_col_name: 'case_id'}, inplace=True)

# 设置 case_id 为索引
df_rna_raw.set_index('case_id', inplace=True)

# 重命名基因列（去除可能的 _rnaseq 后缀）
df_rna_raw.columns = [c.replace('_rnaseq', '') for c in df_rna_raw.columns]

print(f"基因数量: {len(df_rna_raw.columns)}")
print(f"病人数量: {len(df_rna_raw)}")

# -----------------------
# 4. 对齐数据 (用patient_id)
# -----------------------
print("正在对齐数据...")

# 获取两个数据集中共同的病人ID
clinical_patients = set(clinical_data['patient_id'])
rna_patients = set(df_rna_raw.index)
common_patients = list(clinical_patients & rna_patients)  # 转为列表保持顺序

print(f"临床数据病人数: {len(clinical_patients)}")
print(f"基因数据病人数: {len(rna_patients)}")
print(f"共同病人数: {len(common_patients)}")

# 重新按 common_patients 排序临床数据
clinical_data_filtered = clinical_data[clinical_data['patient_id'].isin(common_patients)]
clinical_data_filtered = clinical_data_filtered.set_index('patient_id')
clinical_data_filtered = clinical_data_filtered.loc[common_patients].reset_index()

# 重新按 common_patients 排序基因数据
df_rna_filtered = df_rna_raw.loc[common_patients]

print(f"过滤后临床数据行数: {len(clinical_data_filtered)}")
print(f"过滤后基因数据行数: {len(df_rna_filtered)}")

# -----------------------
# 5. 合并数据并输出
# -----------------------
print("正在合并数据...")

# 合并：行为病人，列为基因 + time
gene_cols = df_rna_filtered.columns.tolist()
output_data = df_rna_filtered.reset_index(drop=True)
output_data['time'] = clinical_data_filtered['time'].values  # 现在长度一致

# 确保 time 是最后一列
cols = gene_cols + ['time']
output_data = output_data[cols]

print(f"输出数据维度: {output_data.shape}")
print(f"样本数: {len(output_data)}")
print(f"基因数: {len(gene_cols)}")

# 保存含表头的版本
save_data_path = os.path.join(output_path, 'data.csv')
output_data.to_csv(save_data_path, index=False)
print(f"成功: 含表头数据已保存至 {save_data_path}")

# 保存不含表头的纯数值矩阵
save_raw_path = os.path.join(output_path, 'data_raw.csv')
output_data.to_csv(save_raw_path, index=False, header=False)
print(f"成功: 纯数值矩阵已保存至 {save_raw_path}")

# 保存符合 CGI 要求的 .mat 文件 (变量名: d)
try:
    import scipy.io
    mat_path = os.path.join(output_path, 'data.mat')
    # 将数值矩阵保存为 MATLAB 格式，变量名为 'd'
    data_matrix = output_data.values
    scipy.io.savemat(mat_path, {'d': data_matrix})
    print(f"成功: CGI用 .mat 文件已保存至 {mat_path}")
    print(f"      变量名: d, 维度: {data_matrix.shape}")
except ImportError:
    print("警告: scipy 未安装，跳过 .mat 文件生成")

print(f"\n预处理完成！")
print(f"保留样本数: {len(output_data)}")
print(f"保留基因数: {len(gene_cols)}")
