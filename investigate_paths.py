#!/usr/bin/env python3
"""
路径侦查脚本：检查 BRCA 和 BLCA 的 .pt 特征文件是否存在
完全复用 dataset_survival.py 中 _load_wsi_embs_from_path 的逻辑
"""

import os
import pandas as pd

# 测试样本
BRCA_SAMPLE = "TCGA-A8-A06Z"
BLCA_SAMPLE = "TCGA-DK-A3IM"

# 数据目录
DATA_ROOT_DIR = "data"

def check_sample_features(study, case_id):
    """检查某个样本的 WSI 特征文件是否存在"""

    print(f"\n{'='*80}")
    print(f"检查 {study.upper()} 癌种 - 病例: {case_id}")
    print(f"{'='*80}")

    # 读取临床数据获取 slide_ids
    clinical_file = f"datasets_csv/clinical_data/tcga_{study}_clinical.csv"

    if not os.path.exists(clinical_file):
        print(f"❌ 临床数据文件不存在: {clinical_file}")
        return

    df = pd.read_csv(clinical_file)

    if case_id not in df['case_id'].values:
        print(f"❌ 病例 ID 不在临床数据中: {case_id}")
        return

    slide_id = df[df['case_id'] == case_id]['slide_id'].values[0]
    print(f"   slide_id: {slide_id}")

    # 构造路径 - 完全复用 _load_wsi_embs_from_path 的逻辑
    data_dir = os.path.join(DATA_ROOT_DIR, study, "pt_files")
    wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))

    # 打印绝对路径
    abs_path = os.path.abspath(wsi_path)
    print(f"\n📁 构造的 .pt 文件路径:")
    print(f"   {abs_path}")

    # 检查是否存在
    exists = os.path.exists(abs_path)
    print(f"\n📊 文件存在性检查:")
    print(f"   存在: {'✅ 是' if exists else '❌ 否'}")

    if not exists:
        # 检查目录是否存在
        dir_exists = os.path.exists(data_dir)
        print(f"   目录存在: {'✅ 是' if dir_exists else '❌ 否'}")
        if dir_exists:
            # 列出目录中的文件数
            num_files = len(os.listdir(data_dir))
            print(f"   目录中的文件数: {num_files}")

            # 列出前5个文件作为参考
            files = os.listdir(data_dir)[:5]
            print(f"   前5个文件:")
            for f in files:
                print(f"      - {f}")

    return exists, abs_path

def main():
    print("="*80)
    print("路径侦查任务：检查 BRCA 和 BLCA 的 WSI 特征文件")
    print("="*80)

    # 检查 BRCA
    brca_exists, brca_path = check_sample_features('brca', BRCA_SAMPLE)

    # 检查 BLCA
    blca_exists, blca_path = check_sample_features('blca', BLCA_SAMPLE)

    # 总结
    print(f"\n{'='*80}")
    print("侦查结果总结")
    print(f"{'='*80}")
    print(f"BRCA ({BRCA_SAMPLE}): {'✅ 存在' if brca_exists else '❌ 缺失'} - {brca_path}")
    print(f"BLCA ({BLCA_SAMPLE}): {'✅ 存在' if blca_exists else '❌ 缺失'} - {blca_path}")
    print(f"{'='*80}")

    if not blca_exists:
        print("\n⚠️  BLCA 特征文件缺失!")
        print("建议: 运行全盘搜索或特征提取脚本")
    else:
        print("\n✅ 两个癌种的特征文件都存在")

if __name__ == "__main__":
    main()
