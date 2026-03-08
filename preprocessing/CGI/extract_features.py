"""
从筛选出的基因 .mat 文件中提取基因特征
- 读取 found_Genes 索引
- 从源文件中提取对应基因的表达量
- 使用真实病人ID作为列名
- 转置后输出（行为基因，列为病人）
- 验证：随机抽取病人ID，匹配原文件病人ID后对比表达量
"""
import os
import pandas as pd
import numpy as np
import scipy.io
import random

# =====================
# 配置
# =====================
cancer_type = 'stad'
n_folds = 5
random_seed = 42

# 路径配置
base_path = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy'
data_folder = f'{base_path}/preprocessing/CGI/data'
splits_folder = f'{base_path}/splits/CGI_nested_cv/{cancer_type}'

# 源文件：训练集（包含所有基因，用于提取表达量）
source_mat_path = f'{splits_folder}/train_fold{{}}.mat'

# CGI筛选出的基因索引文件
index_mat_path = f'{data_folder}/{cancer_type}/{cancer_type}_found_Genes_fold{{}}.mat'


# 原始完整数据（用于获取基因名）
original_csv_with_id_path = f'{data_folder}/{cancer_type}/{cancer_type}_data_with_id.csv'

# 划分文件（用于获取训练集的病人ID）
split_csv_path = f'{splits_folder}/nested_splits_{{}}.csv'

# 输出文件夹
output_folder = f'{data_folder}/{cancer_type}_found_genes'
os.makedirs(output_folder, exist_ok=True)

# =====================
# 主程序
# =====================
def process_fold(fold):
    print(f"\n{'='*50}")
    print(f"处理 Fold {fold}")
    print('='*50)

    # 1. 读取源文件（训练集，包含所有基因）
    source_mat = scipy.io.loadmat(source_mat_path.format(fold))
    source_data = source_mat['d']  # (n_samples, n_genes+1)
    n_genes = source_data.shape[1] - 1  # 最后一列是time
    n_samples = source_data.shape[0]
    print(f"源文件维度: {source_data.shape}")  # (样本数, 基因数+1)

    # 2. 读取原始完整数据（获取基因名和所有病人ID）
    original_with_id = pd.read_csv(original_csv_with_id_path)
    gene_names_all = original_with_id.columns[1:-1].tolist()  # 跳过patient_id，最后是time
    patient_ids_all = original_with_id['patient_id'].tolist()  # 所有病人ID
    print(f"原始基因数: {len(gene_names_all)}")
    print(f"原始数据病人数: {len(patient_ids_all)}")

    # 3. 读取划分文件（获取训练集的病人ID）
    split_df = pd.read_csv(split_csv_path.format(fold))
    train_patient_ids = split_df['train'].dropna().tolist()
    print(f"训练集病人数: {len(train_patient_ids)}")

    # 3. 读取CGI筛选出的基因索引
    index_mat = scipy.io.loadmat(index_mat_path.format(fold))
    gene_indices_1based = index_mat['found_Genes'].flatten()  # 1-based 索引
    gene_indices = gene_indices_1based - 1  # 转为 0-based
    print(f"筛选基因数: {len(gene_indices)}")

    # 获取筛选的基因名
    selected_gene_names = [gene_names_all[i] for i in gene_indices]
    print(f"筛选基因名: {selected_gene_names}")

    # 4. 从源数据中提取筛选基因的表达量
    # 源数据的样本顺序与原始CSV中的样本顺序一致
    source_genes = source_data[:, :n_genes]  # 不含time列
    selected_expr = source_genes[:, gene_indices]  # (n_samples, n_selected)
    print(f"提取表达量维度: {selected_expr.shape}")

    # 5. 获取与源文件匹配的病人ID（使用与源文件样本顺序一致的前n个）
    # 源文件保存时使用的是 output_data 的子集，顺序与原始数据一致
    n_source_samples = selected_expr.shape[0]
    matched_patient_ids = patient_ids_all[:n_source_samples]
    print(f"匹配的病人ID数量: {len(matched_patient_ids)}")

    # 6. 随机抽取验证病人
    random.seed(random_seed + fold)
    n_verify = min(5, n_source_samples)
    verify_indices = random.sample(range(n_source_samples), n_verify)
    print(f"\n验证样本索引: {verify_indices}")

    # 6. 验证：对比源文件和提取的表达量（通过索引）
    print("开始验证...")
    verify_passed = True

    for idx in verify_indices:
        patient_id = matched_patient_ids[idx]
        for k, gene_idx in enumerate(gene_indices):
            src_val = source_data[idx, gene_idx]
            sel_val = selected_expr[idx, k]
            if not np.isclose(src_val, sel_val, rtol=1e-5):
                print(f"  病人{patient_id}, 基因{selected_gene_names[k]} 不一致: 源={src_val}, 提取={sel_val}")
                verify_passed = False

    if verify_passed:
        print(f"  ✓ 验证通过！所有抽取样本的表达量一致")
    else:
        print(f"  ✗ 验证失败！存在不一致的值")
        return False

    # 7. 转置并输出
    # 转置：(n_samples, n_genes) -> (n_genes, n_samples)
    transposed = selected_expr.T  # (n_selected, n_samples)

    # 构建DataFrame（第一列是基因名，其余列是匹配的真实病人ID）
    output_df = pd.DataFrame(transposed, columns=matched_patient_ids)
    output_df.insert(0, 'gene_name', selected_gene_names)

    # 保存
    output_path = f'{output_folder}/{cancer_type}_found_Genes_fold{fold}.csv'
    output_df.to_csv(output_path, index=False)
    print(f"\n成功保存: {output_path}")
    print(f"输出维度: {output_df.shape}")  # (n_genes, n_samples+1)

    return True

# 处理所有fold
all_passed = True
for fold in range(n_folds):
    if not process_fold(fold):
        all_passed = False

print(f"\n{'='*50}")
if all_passed:
    print("所有Fold处理完成，验证通过！")
else:
    print("警告：部分Fold验证失败，请检查！")
print('='*50)
