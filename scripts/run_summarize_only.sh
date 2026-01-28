#!/bin/bash
# ====================================================================
# 只运行汇总步骤（不重新训练）- 鲁棒递归查找版
# 支持任意嵌套深度的目录结构，包括双重 results/ 路径
# ====================================================================

if [ -z "$1" ]; then
    echo "❌ 用法: bash run_summarize_only.sh <癌种简称>"
    echo "   例如: bash run_summarize_only.sh blca"
    exit 1
fi

STUDY=$1
TODAY=$(date +%Y-%m-%d)
ABLRESULTS_DIR="results/ablation/${STUDY}"
export ABLRESULTS_DIR  # 导出变量
export STUDY           # 导出变量

# 【修复】检查是否存在双层路径
if [ -d "results/results/ablation/${STUDY}" ]; then
    echo "⚠️  检测到双层路径 results/results/ablation/"
    ABLRESULTS_DIR="results/results/ablation/${STUDY}"
    export ABLRESULTS_DIR
    echo "📁 已切换到: ${ABLRESULTS_DIR}"
fi

# 创建报告目录
REPORT_DIR="report"
mkdir -p "${REPORT_DIR}"

echo "🔄 只运行汇总步骤: ${STUDY}"
echo "=============================================="
echo "📁 结果目录: ${ABLRESULTS_DIR}"
echo "📋 报告将保存到: ${REPORT_DIR}/"
echo ""

cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 汇总 Gene Only 结果
echo "📊 汇总 Gene Only 结果..."
GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"
python3 << 'EOF_SUMMARY' | tee -a "${ABLRESULTS_DIR}/gene_summarize.log"
import pandas as pd
import glob
import os
import sys
import re

# 获取环境变量
study = os.environ.get('STUDY', '')
mode = 'gene'

print(f"🔍 搜索模式: */*/{study}/{mode}/**/summary_partial_*.csv")

# 【鲁棒搜索】从当前目录递归搜索
# 支持任意嵌套深度，包括双重 results/results/ 路径
search_pattern = f"./**/{study}/{mode}/**/summary_partial_*.csv"
partial_files = glob.glob(search_pattern, recursive=True)

print(f"📁 绝对搜索路径: {os.path.abspath('.')}")
print(f"📁 找到 {len(partial_files)} 个匹配文件")

if not partial_files:
    print("⚠️  未找到任何文件，扩大搜索范围...")
    # 备选：搜索所有 summary_partial_*.csv
    all_partials = glob.glob("./**/summary_partial_*.csv", recursive=True)
    print(f"   扩大搜索找到 {len(all_partials)} 个文件，列出前10个:")
    for f in all_partials[:10]:
        print(f"   - {f}")
    partial_files = all_partials

dfs = []
found_folds = set()

for file_path in partial_files:
    # 【路径过滤】确保路径包含正确的 study 和 mode
    if f"/{study}/" not in file_path or f"/{mode}/" not in file_path:
        print(f"  ⏭️  跳过不匹配的文件: {file_path}")
        continue

    print(f"  📄 匹配: {file_path}")

    # 【从文件名解析 Fold 编号】
    # 文件名格式: summary_partial_{k_start}_{k_end}.csv
    file_name = os.path.basename(file_path)
    match = re.search(r'summary_partial_(\d+)_(\d+)\.csv', file_name)

    if match:
        fold_num = int(match.group(1))  # k_start 作为 fold 编号
        found_folds.add(fold_num)

        try:
            df = pd.read_csv(file_path)
            df['fold'] = fold_num
            dfs.append(df)
            print(f"     ✓ Fold {fold_num}: 成功读取")
        except Exception as e:
            print(f"     ✗ Fold {fold_num}: 读取失败 - {e}")
    else:
        print(f"     ⚠️  无法解析文件名: {file_name}")

# 统计结果
total_expected = 5
found_count = len(found_folds)
missing_folds = set(range(total_expected)) - found_folds

print(f"\n📊 统计:")
print(f"   - 找到: {found_count}/5 折")
if missing_folds:
    print(f"   - 缺失: {sorted(missing_folds)}")
else:
    print(f"   - 完成度: 100%")

# 获取环境变量中的路径
gene_summary_path = os.environ.get('ABLRESULTS_DIR', '') + '/gene/summary.csv'

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(gene_summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'\n✅ Gene Only 汇总完成: {found_count}/5 折')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('\n❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv(gene_summary_path, index=False)
EOF_SUMMARY

# 汇总 Text Only 结果
echo ""
echo "📊 汇总 Text Only 结果..."
TEXT_SUMMARY="${ABLRESULTS_DIR}/text/summary.csv"
python3 << 'EOF_SUMMARY' | tee -a "${ABLRESULTS_DIR}/text_summarize.log"
import pandas as pd
import glob
import os
import sys
import re

study = os.environ.get('STUDY', '')
mode = 'text'

print(f"🔍 搜索模式: */*/{study}/{mode}/**/summary_partial_*.csv")

search_pattern = f"./**/{study}/{mode}/**/summary_partial_*.csv"
partial_files = glob.glob(search_pattern, recursive=True)

print(f"📁 绝对搜索路径: {os.path.abspath('.')}")
print(f"📁 找到 {len(partial_files)} 个匹配文件")

if not partial_files:
    print("⚠️  未找到任何文件，扩大搜索范围...")
    all_partials = glob.glob("./**/summary_partial_*.csv", recursive=True)
    print(f"   扩大搜索找到 {len(all_partials)} 个文件，列出前10个:")
    for f in all_partials[:10]:
        print(f"   - {f}")
    partial_files = all_partials

dfs = []
found_folds = set()

for file_path in partial_files:
    if f"/{study}/" not in file_path or f"/{mode}/" not in file_path:
        print(f"  ⏭️  跳过不匹配的文件: {file_path}")
        continue

    print(f"  📄 匹配: {file_path}")

    file_name = os.path.basename(file_path)
    match = re.search(r'summary_partial_(\d+)_(\d+)\.csv', file_name)

    if match:
        fold_num = int(match.group(1))
        found_folds.add(fold_num)

        try:
            df = pd.read_csv(file_path)
            df['fold'] = fold_num
            dfs.append(df)
            print(f"     ✓ Fold {fold_num}: 成功读取")
        except Exception as e:
            print(f"     ✗ Fold {fold_num}: 读取失败 - {e}")
    else:
        print(f"     ⚠️  无法解析文件名: {file_name}")

total_expected = 5
found_count = len(found_folds)
missing_folds = set(range(total_expected)) - found_folds

print(f"\n📊 统计:")
print(f"   - 找到: {found_count}/5 折")
if missing_folds:
    print(f"   - 缺失: {sorted(missing_folds)}")
else:
    print(f"   - 完成度: 100%")

# 获取环境变量中的路径
text_summary_path = os.environ.get('ABLRESULTS_DIR', '') + '/text/summary.csv'

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(text_summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'\n✅ Text Only 汇总完成: {found_count}/5 折')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('\n❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv(text_summary_path, index=False)
EOF_SUMMARY

# 汇总 Fusion 结果
echo ""
echo "📊 汇总 Fusion 结果..."
FUSION_SUMMARY="${ABLRESULTS_DIR}/fusion/summary.csv"
python3 << 'EOF_SUMMARY' | tee -a "${ABLRESULTS_DIR}/fusion_summarize.log"
import pandas as pd
import glob
import os
import sys
import re

study = os.environ.get('STUDY', '')
mode = 'fusion'

print(f"🔍 搜索模式: */*/{study}/{mode}/**/summary_partial_*.csv")

search_pattern = f"./**/{study}/{mode}/**/summary_partial_*.csv"
partial_files = glob.glob(search_pattern, recursive=True)

print(f"📁 绝对搜索路径: {os.path.abspath('.')}")
print(f"📁 找到 {len(partial_files)} 个匹配文件")

if not partial_files:
    print("⚠️  未找到任何文件，扩大搜索范围...")
    all_partials = glob.glob("./**/summary_partial_*.csv", recursive=True)
    print(f"   扩大搜索找到 {len(all_partials)} 个文件，列出前10个:")
    for f in all_partials[:10]:
        print(f"   - {f}")
    partial_files = all_partials

dfs = []
found_folds = set()

for file_path in partial_files:
    if f"/{study}/" not in file_path or f"/{mode}/" not in file_path:
        print(f"  ⏭️  跳过不匹配的文件: {file_path}")
        continue

    print(f"  📄 匹配: {file_path}")

    file_name = os.path.basename(file_path)
    match = re.search(r'summary_partial_(\d+)_(\d+)\.csv', file_name)

    if match:
        fold_num = int(match.group(1))
        found_folds.add(fold_num)

        try:
            df = pd.read_csv(file_path)
            df['fold'] = fold_num
            dfs.append(df)
            print(f"     ✓ Fold {fold_num}: 成功读取")
        except Exception as e:
            print(f"     ✗ Fold {fold_num}: 读取失败 - {e}")
    else:
        print(f"     ⚠️  无法解析文件名: {file_name}")

total_expected = 5
found_count = len(found_folds)
missing_folds = set(range(total_expected)) - found_folds

print(f"\n📊 统计:")
print(f"   - 找到: {found_count}/5 折")
if missing_folds:
    print(f"   - 缺失: {sorted(missing_folds)}")
else:
    print(f"   - 完成度: 100%")

# 获取环境变量中的路径
fusion_summary_path = os.environ.get('ABLRESULTS_DIR', '') + '/fusion/summary.csv'

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(fusion_summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'\n✅ Fusion 汇总完成: {found_count}/5 折')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('\n❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv(fusion_summary_path, index=False)
EOF_SUMMARY

echo ""
echo "✅ 汇总完成!"
echo "=============================================="
echo "📁 结果文件:"
echo "   - Gene: ${ABLRESULTS_DIR}/gene/summary.csv"
echo "   - Text: ${ABLRESULTS_DIR}/text/summary.csv"
echo "   - Fusion: ${ABLRESULTS_DIR}/fusion/summary.csv"
echo ""
echo "📋 报告文件 (复制到 report/):"
for mode in gene text fusion; do
    src="${ABLRESULTS_DIR}/${mode}/summary.csv"
    dst="report/${TODAY}_${STUDY}_${mode}_summary.csv"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "   ✓ ${dst}"
    fi
done
echo "=============================================="
