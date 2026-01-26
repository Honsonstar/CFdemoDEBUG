import os
import pandas as pd

# -----------------------
# 1. 参数设置 (可修改部分)
# -----------------------
# 在这里修改癌症类型，例如 'blca', 'brca', 'gbm' 等
cancer_type = 'coadread'

# 定义输入文件路径 (使用 f-string 自动拼接路径)
# 临床数据路径
clinical_input_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv/metadata/tcga_{cancer_type}.csv'
# 基因表达数据路径
rna_input_path = f'/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv/raw_rna_data/combine/{cancer_type}/rna_clean.csv'

# 定义输出文件夹路径 (例如: tcga_blca)
output_path = f'tcga_{cancer_type}'
print(f"当前处理癌症类型: {cancer_type}")
print(f"输出文件夹: {output_path}")

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 输出文件名
output_clinical = 'clinical.CSV'
output_data = 'data.csv'

# -----------------------
# 2. 处理临床数据 (Clinical Data)
# -----------------------
print("正在处理临床数据...")
try:
    # 读取新的临床数据源
    df_clinical_raw = pd.read_csv(clinical_input_path)

    # 需要保留的列 (与原代码保持一致)
    clinical_cols = [
        'case_id',          # 病人ID
        'site',             # 样本采集部位
        'is_female',        # 性别
        'oncotree_code',    # 肿瘤分型代码
        'age',              # 年龄
        'survival_months',  # 生存时间
        'censorship',       # 随访事件
        'train'             # 训练/测试集划分
    ]
    
    # 提取所需列
    clinical_data = df_clinical_raw[clinical_cols].copy()

    # 重命名列以匹配下游任务 (Stage 1) 的要求
    # case_id -> case_submitter_id
    # survival_months -> OS
    clinical_data.rename(columns={
        'case_id': 'case_submitter_id',
        'survival_months': 'OS',
    }, inplace=True)

    # 计算 vital_status 和 Censor
    # 逻辑：censorship=0 代表死亡(Event), 1 代表存活(Censored)
    # 我们需要生成 Censor 列：1 代表死亡 (Event happen), 0 代表存活
    clinical_data['vital_status'] = clinical_data['censorship'].apply(lambda x: 'Dead' if x == 0 else 'Alive')
    clinical_data['Censor'] = clinical_data['censorship'].apply(lambda x: 1 if x == 0 else 0)

    # 保存 clinical 数据文件
    save_clin_path = os.path.join(output_path, output_clinical)
    clinical_data.to_csv(save_clin_path, index=False)
    print(f"成功: 临床数据已保存至 {save_clin_path}")

except Exception as e:
    print(f"错误: 处理临床数据时出错 - {e}")
    exit()


# -----------------------
# 3. 处理基因表达数据 (RNA Data)
# -----------------------
print("正在处理基因数据...")
try:
    # 读取新的 RNA 数据源
    # 假设输入文件格式为：每一行是一个病人，每一列是一个基因 (需要转置)
    df_rna_raw = pd.read_csv(rna_input_path)

    # 检查 case_id 是否在列中
    if 'case_id' in df_rna_raw.columns:
        # 将 case_id 设为索引，以便转置时它变成列名
        df_rna_raw.set_index('case_id', inplace=True)
    elif 'Unnamed: 0' in df_rna_raw.columns:
        # 有些文件第一列可能是 Unnamed: 0，视情况而定
        df_rna_raw.set_index('Unnamed: 0', inplace=True)
    
    # 执行转置 (Transpose)
    # 转置前: Index=病人ID, Columns=基因名
    # 转置后: Index=基因名, Columns=病人ID
    exp_data = df_rna_raw.transpose()

    # 重置索引，将基因名从 Index 变成一列，并命名为 'gene_name'
    exp_data.reset_index(inplace=True)
    exp_data.rename(columns={'index': 'gene_name'}, inplace=True)

    # 注意：这里不需要再处理 _rnaseq 后缀，因为通常 rna_clean.csv 里的列名已经是纯净的基因名
    # 如果你的新文件里基因名还是带 _rnaseq，请取消下面这行的注释:
    # exp_data['gene_name'] = exp_data['gene_name'].str.replace('_rnaseq', '')

    # 保存 gene expression 数据文件
    save_data_path = os.path.join(output_path, output_data)
    exp_data.to_csv(save_data_path, index=False)
    print(f"成功: 基因数据(已转置)已保存至 {save_data_path}")

except Exception as e:
    print(f"错误: 处理基因数据时出错 - {e}")
    exit()

print("预处理全部完成！")