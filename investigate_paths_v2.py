#!/usr/bin/env python3
"""
路径侦查脚本 v2：检查 BRCA 和 BLCA 的 .pt 特征文件是否存在
检查多个可能的路径
"""

import os
import pandas as pd
import glob

# 测试样本
BRCA_SAMPLE = "TCGA-A8-A06Z"
BLCA_SAMPLE = "TCGA-DK-A3IM"

def find_all_pt_files(sample_id):
    """全盘搜索某个样本的 .pt 文件"""
    print(f"\n🔍 搜索 {sample_id} 相关的 .pt 文件...")

    # 方法1: 直接搜索包含 sample_id 的 .pt 文件
    pattern1 = f"**/*{sample_id}*.pt"
    files1 = glob.glob(pattern1, recursive=True)

    # 方法2: 搜索 slide_id 的前半部分
    pattern2 = f"**/*{sample_id[:15]}*.pt"
    files2 = glob.glob(pattern2, recursive=True)

    all_files = set(files1 + files2)
    return sorted(all_files)

def check_sample_features_v2(study, case_id):
    """检查某个样本的 WSI 特征文件是否存在 - 多路径检查"""

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
    slide_id_no_ext = slide_id.rstrip('.svs')

    # 可能的路径列表
    possible_paths = [
        # 路径1: data/{study}/pt_files/ (原始假设)
        f"data/{study}/pt_files/{slide_id_no_ext}.pt",
        # 路径2: results/ 目录中可能有
        f"results/**/{slide_id_no_ext}.pt",
        f"results/{study}/**/{slide_id_no_ext}.pt",
        # 路径3: data/ 根目录
        f"data/{slide_id_no_ext}.pt",
        # 路径4: 当前目录
        f"./{slide_id_no_ext}.pt",
    ]

    print(f"\n📁 检查可能的路径:")
    found_path = None
    for i, path in enumerate(possible_paths, 1):
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        status = "✅ 存在" if exists else "❌ 不存在"

        # 只打印前3个或者存在的
        if i <= 3 or exists:
            print(f"   {i}. {abs_path[:80]}...")
            print(f"      {status}")

        if exists and found_path is None:
            found_path = abs_path

    # 全盘搜索
    print(f"\n🔍 全盘搜索 {case_id} 相关的 .pt 文件...")
    search_results = find_all_pt_files(case_id)

    if search_results:
        print(f"   找到 {len(search_results)} 个文件:")
        for f in search_results[:10]:  # 只显示前10个
            print(f"      - {f}")
        if found_path is None and search_results:
            found_path = search_results[0]
    else:
        print(f"   ❌ 未找到任何相关文件")

    # 总结
    print(f"\n{'='*80}")
    print(f"检查结果: {'✅ 找到' if found_path else '❌ 完全缺失'}")
    if found_path:
        print(f"   路径: {found_path}")
    print(f"{'='*80}")

    return found_path

def main():
    print("="*80)
    print("路径侦查任务 v2：检查 BRCA 和 BLCA 的 WSI 特征文件")
    print("="*80)

    # 检查 BRCA
    print("\n" + "🔵"*40)
    brca_path = check_sample_features_v2('brca', BRCA_SAMPLE)

    # 检查 BLCA
    print("\n" + "🟠"*40)
    blca_path = check_sample_features_v2('blca', BLCA_SAMPLE)

    # 总结
    print(f"\n{'='*80}")
    print("侦查结果总结")
    print(f"{'='*80}")
    print(f"BRCA ({BRCA_SAMPLE}): {'✅ 存在' if brca_path else '❌ 缺失'}")
    if brca_path:
        print(f"   {brca_path}")
    print(f"BLCA ({BLCA_SAMPLE}): {'✅ 存在' if blca_path else '❌ 缺失'}")
    if blca_path:
        print(f"   {blca_path}")
    print(f"{'='*80}")

    if brca_path and not blca_path:
        print("\n⚠️  关键发现:")
        print("   BRCA 的特征文件存在，但 BLCA 的缺失!")
        print("   这解释了为什么 BRCA 能跑出 0.8，BLCA 只有 0.5")
    elif not brca_path and not blca_path:
        print("\n⚠️  两个癌种的特征文件都缺失!")
        print("   需要先运行特征提取脚本生成 .pt 文件")

if __name__ == "__main__":
    main()
