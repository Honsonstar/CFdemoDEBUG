import pandas as pd
import argparse
import os
import sys

def merge_data(study):
    # 1. 定义文件路径
    base_dir = "/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/datasets_csv"
    clinical_path = os.path.join(base_dir, "clinical_data", f"tcga_{study}_clinical.csv")
    metadata_path = os.path.join(base_dir, "metadata", f"tcga_{study}.csv")

    print(f"[*] 正在处理 Study: {study}")
    print(f"    - Clinical文件: {clinical_path}")
    print(f"    - Metadata文件: {metadata_path}")

    # 2. 检查文件是否存在
    if not os.path.exists(clinical_path) or not os.path.exists(metadata_path):
        print("[!] 错误: 文件不存在，请检查路径。")
        return

    # 3. 读取数据
    df_clinical = pd.read_csv(clinical_path)
    df_meta = pd.read_csv(metadata_path)

    # 统一 case_id 格式
    if 'case_id' not in df_clinical.columns or 'case_id' not in df_meta.columns:
        print("[!] 错误: 两个文件都必须包含 'case_id' 列。")
        return

    df_clinical['case_id'] = df_clinical['case_id'].astype(str).str.strip()
    df_meta['case_id'] = df_meta['case_id'].astype(str).str.strip()

    print(f"[*] 读取成功:")
    print(f"    - Clinical 行数: {len(df_clinical)}")
    print(f"    - Metadata 行数: {len(df_meta)}")

    # 4. 验证重叠列
    common_cols = list(set(df_clinical.columns) & set(df_meta.columns))
    if 'case_id' in common_cols: common_cols.remove('case_id')

    if common_cols:
        print(f"[*] 检测到重叠列: {len(common_cols)} 个")
    else:
        print("[*] 没有发现重叠列。")

    # 5. 执行合并
    new_cols = [c for c in df_meta.columns if c not in df_clinical.columns]
    
    if not new_cols:
        print("[*] 提示: Metadata 中没有 Clinical 缺失的新列，跳过合并步骤，直接进入清洗检查。")
        df_merged = df_clinical
    else:
        print(f"[*] 准备合并下列新数据: {new_cols}")
        df_meta_subset = df_meta[['case_id'] + new_cols]
        # 使用 Left Join 保留所有 Clinical 中的病人
        df_merged = pd.merge(df_clinical, df_meta_subset, on='case_id', how='left')
    
    # ============================================================
    # 6. 数据清洗：保留样本，填充缺失的 slide_id
    # ============================================================
    if 'slide_id' in df_merged.columns:
        print("-" * 40)
        print(f"[*] 正在检查 slide_id 完整性...")
        
        original_count = len(df_merged)
        
        # 检查缺失情况
        missing_count = df_merged['slide_id'].isna().sum()
        
        if missing_count > 0:
            print(f"    ⚠️  发现 {missing_count} 名病人缺少 slide_id。")
            print(f"    -> [策略变更] 为了保留样本进行多模态训练(SNN)，我们不再剔除这些样本。")
            print(f"    -> [操作] 将缺失的 slide_id 填充为 'N/A'，防止 float 属性报错。")
            
            # 【关键修改】填充 NaN 为 "N/A"
            # 这样原本是 NaN (float) 的位置变成了 字符串，避免了 'float' object has no attribute 'values'
            df_merged['slide_id'] = df_merged['slide_id'].fillna("N/A")
            
        else:
            print("    [√] 所有样本均拥有对应的 slide_id。")
            
        print(f"    -> 最终保留样本数: {len(df_merged)} (原始: {original_count})")
        print("-" * 40)
    else:
        print("[!] 警告: 数据中未找到 'slide_id' 列！将自动添加全 'N/A' 列以防报错。")
        df_merged['slide_id'] = "N/A"

    # ============================================================

    # 7. 覆盖保存
    backup_path = clinical_path + ".bak"
    df_clinical.to_csv(backup_path, index=False)
    print(f"[*] 原文件已备份至: {backup_path}")

    df_merged.to_csv(clinical_path, index=False)
    print(f"[*] 成功！已覆盖保存至: {clinical_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Clinical and Metadata CSVs and Fill Missing Slides")
    parser.add_argument('--study', type=str, required=True, help="Study name (e.g., brca, blca, hnsc)")
    args = parser.parse_args()
    
    merge_data(args.study)