"""
从筛选出的基因 .mat 文件中提取基因特征
- 读取 found_Genes 索引（CGI基因筛选只用训练集）
- 从所有样本中提取对应基因的表达量（用于训练和评估）
- 使用病人ID作为列名
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
cancer_type = 'coadread'
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
    print('===')

    # 1. 读取原始完整数据（获取基因名、所有病人ID和基因表达量）
    original_with_id = pd.read_csv(original_csv_with_id_path)
    gene_names_all = original_with_id.columns[1:-1].tolist()  # 跳过patient_id，最后是time
    patient_ids_all = original_with_id['patient_id'].tolist()  # 所有病人ID
    # 提取基因表达量（去掉patient_id和time列）
    all_gene_expr = original_with_id.iloc[:, 1:-1].values  # (259, 4999)
    print(f"原始基因数: {len(gene_names_all)}")
    print(f"原始数据病人数: {len(patient_ids_all)}")
    print(f"原始表达量维度: {all_gene_expr.shape}")

    # 2. 读取划分文件（获取所有样本的病人ID：train + val + test）
    # 注意：CGI基因筛选只用训练集，但特征提取需要包含所有样本用于评估
    split_df = pd.read_csv(split_csv_path.format(fold))
    train_patient_ids = split_df['train'].dropna().tolist()
    val_patient_ids = split_df['val'].dropna().tolist()
    test_patient_ids = split_df['test'].dropna().tolist()
    # 所有样本的ID（用于特征提取和模型评估）
    all_split_patient_ids = train_patient_ids + val_patient_ids + test_patient_ids
    print(f"训练集病人数: {len(train_patient_ids)}")
    print(f"验证集病人数: {len(val_patient_ids)}")
    print(f"测试集病人数: {len(test_patient_ids)}")
    print(f"总样本数: {len(all_split_patient_ids)}")

    # 3. 读取CGI筛选出的基因索引
    index_mat = scipy.io.loadmat(index_mat_path.format(fold))
    gene_indices_1based = index_mat['found_Genes'].flatten()  # 1-based 索引
    gene_indices = gene_indices_1based - 1  # 转为 0-based
    print(f"筛选基因数: {len(gene_indices)}")

    # 获取筛选的基因名
    selected_gene_names = [gene_names_all[i] for i in gene_indices]
    print(f"筛选基因名: {selected_gene_names}")

    # 4. 按划分文件中的所有样本ID顺序提取数据（train + val + test）
    # 创建病人ID到原始数据索引的映射
    patient_to_idx = {pid: idx for idx, pid in enumerate(patient_ids_all)}

    # 按照 all_split_patient_ids 的顺序从完整基因表达量中提取数据
    n_genes_selected = len(gene_indices)
    selected_expr = np.zeros((len(all_split_patient_ids), n_genes_selected))

    for i, patient_id in enumerate(all_split_patient_ids):
        original_idx = patient_to_idx[patient_id]
        selected_expr[i, :] = all_gene_expr[original_idx, gene_indices]

    print(f"提取表达量维度: {selected_expr.shape}")
    print(f"匹配的病人ID数量: {len(all_split_patient_ids)}")

    # 5. 验证：直接比较 selected_expr 和原始提取的数据是否一致
    print("开始验证...")
    verify_passed = True

    random.seed(random_seed + fold)
    n_verify = min(5, len(all_split_patient_ids))
    verify_indices = random.sample(range(len(all_split_patient_ids)), n_verify)
    print(f"验证样本索引: {verify_indices}")

    for idx in verify_indices:
        patient_id = all_split_patient_ids[idx]
        original_idx = patient_to_idx[patient_id]
        for k, gene_idx in enumerate(gene_indices):
            src_val = all_gene_expr[original_idx, gene_idx]
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

    # 构建DataFrame（第一列是基因名，其余列是所有样本的病人ID）
    output_df = pd.DataFrame(transposed, columns=all_split_patient_ids)
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
