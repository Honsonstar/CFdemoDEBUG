import os.path

import pandas as pd

# -----------------------
# 参数设置
# -----------------------
input_file = 'tcga_brca_all_clean.csv'


output_path = input_file.split('_')[0]+'_'+input_file.split('_')[1]
print("output_path: ", output_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 输出文件名
output_clinical = 'clinical.CSV'
output_data = 'data.csv'

# 根据你的文件格式设置分隔符，此处示例假设为制表符，若是逗号分隔请改为 sep=','
sep_char = ','

# -----------------------
# 读取原始数据，并进行去重
# -----------------------
df = pd.read_csv(input_file, sep=sep_char)
# 针对重复的病人（同一 case_id，只保留第一条），注意：这里假设病人的标识为 case_id
df_unique = df.drop_duplicates(subset='case_id', keep='first').copy()

# -----------------------
# 生成 clinical.CSV 文件
# -----------------------
# 选择临床数据的基本列
clinical_cols = [
    'case_id',  # 病人ID
    'site',  # 样本采集部位或病灶部位
    'is_female',  # 性别标识（1表示女性，0表示男性；具体含义请参考数据说明）
    'oncotree_code',  # 肿瘤分型代码
    'age',  # 诊断时年龄
    'survival_months',  # 生存时间（月）
    'censorship',  # 随访事件，1代表发生事件（例如死亡），0代表数据截尾
    'train'  # 数据集划分（例如训练集/测试集标识）
]
clinical_data = df_unique[clinical_cols].copy()

# 修改列名以匹配原项目预期的字段名称：
#   case_id         -> case_submitter_id
#   survival_months -> OS
#   censorship      -> Censor
clinical_data.rename(columns={
    'case_id': 'case_submitter_id',
    'survival_months': 'OS',
    # 'censorship': 'Censor'
}, inplace=True)

# 增加 vital_status 列，用于后续累积风险计算（例如 cum_hazard 中使用）
"""
tcga基因文件中censorship=0代表死亡，1代表存活，但为了符合项目一致，新增Censor列
这里定义 Censor == 1 表示事件发生（死亡），否则标记为 Alive，因此和Censorship刚好相反
"""
clinical_data['vital_status'] = clinical_data['censorship'].apply(lambda x: 'Dead' if x == 0 else 'Alive')
clinical_data['Censor'] = clinical_data['censorship'].apply(lambda x: 1 if x == 0 else 0)

# 保存 clinical 数据文件
clinical_data.to_csv(os.path.join(output_path,output_clinical), index=False)
print(f"Clinical data saved to {os.path.join(output_path,output_clinical)}")

# -----------------------
# 生成 data.csv 文件（基因表达数据）
# -----------------------
# 只选择后缀为 _rnaseq 的列，因为原项目识别时只使用了 RNA-seq 表达数据
rnaseq_cols = [col for col in df_unique.columns if col.endswith('_rnaseq')]
if not rnaseq_cols:
    raise ValueError("未能找到以 _rnaseq 结尾的基因数据列！")

# 提取病人标识（case_id）和 RNA-seq 数据列
gene_data = df_unique[['case_id'] + rnaseq_cols].copy()
# 以 case_id 为索引，确保与临床数据一致
gene_data.set_index('case_id', inplace=True)

# 去掉 _rnaseq 后缀，例如将 'NDUFC1_rnaseq' 变为 'NDUFC1'，便于后续与pc_symbol文件取交集
gene_data.columns = [col.replace('_rnaseq', '') for col in gene_data.columns]


# 根据原项目要求，转置数据：使得每一行对应一个基因，
# 第一列为 gene_name，其余每列为一个病人对应的表达值。
exp_data = gene_data.transpose().reset_index()
exp_data.rename(columns={'index': 'gene_name'}, inplace=True)

# 保存 gene expression 数据文件
exp_data.to_csv(os.path.join(output_path,output_data), index=False)
print(f"Gene expression data saved to {os.path.join(output_path,output_data)}")
