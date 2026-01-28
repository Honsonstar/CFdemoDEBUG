#!/bin/bash
# ====================================================================
# 只运行汇总步骤（不重新训练）
# ====================================================================

if [ -z "$1" ]; then
    echo "❌ 用法: bash run_summarize_only.sh <癌种简称>"
    echo "   例如: bash run_summarize_only.sh blca"
    exit 1
fi

STUDY=$1
TODAY=$(date +%Y-%m-%d)

# ==================== 数据路径配置 ====================
# 【重要】所有数据路径统一在此处配置
# 注意：必须放在 STUDY=$1 之后

# 1. 结果路径（只使用单层路径）
# ----------------
# 消融实验结果目录: 保存训练结果
ABLRESULTS_DIR="results/ablation/${STUDY}"

# 日志目录
LOG_DIR="log/${TODAY}/${STUDY}"

# 报告目录
REPORT_DIR="report"

# 2. 输出文件配置
# ----------------
# 各模式的汇总文件路径（根据癌症名称动态生成）
GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"
TEXT_SUMMARY="${ABLRESULTS_DIR}/text/summary.csv"
FUSION_SUMMARY="${ABLRESULTS_DIR}/fusion/summary.csv"
FINAL_CSV="${ABLRESULTS_DIR}/final_comparison.csv"
REPORT_CSV="report/${TODAY}_${STUDY}_ablation_comparison.csv"
# =========================================================

# 创建报告目录
mkdir -p "${REPORT_DIR}"

# 导出环境变量供Python使用
export GENE_SUMMARY      # Gene模式汇总文件路径
export TEXT_SUMMARY      # Text模式汇总文件路径
export FUSION_SUMMARY    # Fusion模式汇总文件路径
export FINAL_CSV         # 最终对比表格路径
export REPORT_CSV        # 报告文件路径

echo "🔄 只运行汇总步骤: ${STUDY}"
echo "=============================================="
echo "📁 结果目录: ${ABLRESULTS_DIR}"
echo "📋 报告将保存到: ${REPORT_DIR}/"
echo ""

cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 汇总 Gene Only 结果
echo "📊 汇总 Gene Only 结果..."

python3 << 'EOF_SUMMARY' | tee -a "${ABLRESULTS_DIR}/gene_summarize.log"
import pandas as pd
import glob
import os
import sys
import re

# 【环境变量法】获取变量
study = os.environ.get('STUDY', '')
gene_summary_path = os.environ.get('GENE_SUMMARY', '')
abresults_dir = os.environ.get('ABLRESULTS_DIR', '')

print(f"🔍 搜索模式: */*/{study}/gene/**/summary_partial_*.csv")

# 鲁棒搜索
search_pattern = f"./**/{study}/gene/**/summary_partial_*.csv"
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
    # 路径过滤
    if f"/{study}/" not in file_path or "/gene/" not in file_path:
        continue

    print(f"  📄 匹配: {file_path}")

    # 从文件名解析 fold 编号
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

# 统计
total_expected = 5
found_count = len(found_folds)
missing_folds = set(range(total_expected)) - found_folds

print(f"\n📊 统计:")
print(f"   - 找到: {found_count}/5 折")
if missing_folds:
    print(f"   - 缺失: {sorted(missing_folds)}")
else:
    print(f"   - 完成度: 100%")

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
export TEXT_SUMMARY

python3 << 'EOF_SUMMARY' | tee -a "${ABLRESULTS_DIR}/text_summarize.log"
import pandas as pd
import glob
import os
import sys
import re

study = os.environ.get('STUDY', '')
text_summary_path = os.environ.get('TEXT_SUMMARY', '')

print(f"🔍 搜索模式: */*/{study}/text/**/summary_partial_*.csv")

search_pattern = f"./**/{study}/text/**/summary_partial_*.csv"
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
    if f"/{study}/" not in file_path or "/text/" not in file_path:
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
export FUSION_SUMMARY

python3 << 'EOF_SUMMARY' | tee -a "${ABLRESULTS_DIR}/fusion_summarize.log"
import pandas as pd
import glob
import os
import sys
import re

study = os.environ.get('STUDY', '')
fusion_summary_path = os.environ.get('FUSION_SUMMARY', '')

print(f"🔍 搜索模式: */*/{study}/fusion/**/summary_partial_*.csv")

search_pattern = f"./**/{study}/fusion/**/summary_partial_*.csv"
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
    if f"/{study}/" not in file_path or "/fusion/" not in file_path:
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
