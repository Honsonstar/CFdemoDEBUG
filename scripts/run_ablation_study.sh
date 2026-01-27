#!/bin/bash

# ====================================================================
# 多模态消融实验脚本（并行优化版）
# 对比 Gene Only、Text Only、Fusion 三种模式的性能
# ====================================================================

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 用法: bash run_ablation_study.sh <癌种简称>"
    echo "   例如: bash run_ablation_study.sh blca"
    exit 1
fi

STUDY=$1
echo "🚀 开始多模态消融实验: ${STUDY}"
echo "=============================================="

# 创建结果根目录
ABLRESULTS_DIR="results/ablation/${STUDY}"
mkdir -p "${ABLRESULTS_DIR}"/{gene,text,fusion}

# 设置公共参数
SPLIT_DIR="splits/nested_cv/${STUDY}"
LABEL_FILE="datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv"
SEED=42
K_FOLDS=5
EPOCHS=20
LR=0.00005
MAX_JOBS=4  # 最大并行任务数

# 检查必要文件
if [ ! -d "${SPLIT_DIR}" ]; then
    echo "❌ 错误: 找不到划分文件目录 ${SPLIT_DIR}"
    echo "请先运行: bash create_nested_splits.sh ${STUDY}"
    exit 1
fi

if [ ! -f "${LABEL_FILE}" ]; then
    echo "❌ 错误: 找不到标签文件 ${LABEL_FILE}"
    exit 1
fi

# ====================================================================
# 辅助函数：运行单模式的多折训练（支持并行）
# ====================================================================
run_ablation_mode() {
    local mode_name=$1
    local ab_model=$2
    local results_subdir=$3
    local log_file=$4

    echo "" > "${log_file}"  # 清空日志文件

    echo "==============================================" | tee -a "${log_file}"
    echo "🧬 ${mode_name}" | tee -a "${log_file}"
    echo "==============================================" | tee -a "${log_file}"

    local fold=0
    local running_jobs=0

    for fold in $(seq 0 $((K_FOLDS-1))); do
        local RESULTS_DIR="${ABLRESULTS_DIR}/${results_subdir}/fold_${fold}"
        local fold_log="${RESULTS_DIR}/training.log"
        mkdir -p "${RESULTS_DIR}"

        echo "  └─ 启动 Fold ${fold}... (日志: ${fold_log})" | tee -a "${log_file}"

        # 后台运行训练任务
        python3 main.py \
            --study tcga_${STUDY} \
            --k_start ${fold} \
            --k_end $((fold + 1)) \
            --split_dir "${SPLIT_DIR}" \
            --results_dir "${RESULTS_DIR}" \
            --seed ${SEED} \
            --label_file "${LABEL_FILE}" \
            --task survival \
            --n_classes 4 \
            --modality snn \
            --omics_dir "datasets_csv/raw_rna_data/combine/${STUDY}" \
            --data_root_dir "data/${STUDY}/pt_files" \
            --label_col survival_months \
            --type_of_path combine \
            --max_epochs ${EPOCHS} \
            --lr ${LR} \
            --opt adam \
            --reg 0.00001 \
            --alpha_surv 0.5 \
            --weighted_sample \
            --batch_size 1 \
            --bag_loss nll_surv \
            --encoding_dim 256 \
            --num_patches 4096 \
            --wsi_projection_dim 256 \
            --encoding_layer_1_dim 8 \
            --encoding_layer_2_dim 16 \
            --encoder_dropout 0.25 \
            --ab_model ${ab_model} \
            > "${fold_log}" 2>&1 &

        local pid=$!
        echo "    PID: ${pid}" >> "${log_file}"

        ((running_jobs++))

        # 如果达到最大并发数，等待所有后台任务完成
        if [ ${running_jobs} -ge ${MAX_JOBS} ]; then
            echo "  └─ 达到最大并发数 ${MAX_JOBS}，等待任务完成..." | tee -a "${log_file}"
            wait  # 等待所有后台任务
            running_jobs=0
            echo "  └─ 任务完成，继续启动下一批" | tee -a "${log_file}"
        fi
    done

    # 等待剩余的后台任务
    if [ ${running_jobs} -gt 0 ]; then
        echo "  └─ 等待剩余 ${running_jobs} 个任务完成..." | tee -a "${log_file}"
        wait
    fi

    echo "  └─ ${mode_name} 所有 Fold 完成" | tee -a "${log_file}"
}

# ====================================================================
# 1. Gene Only 模式 (ab_model=2)
# ====================================================================
GENE_LOG="${ABLRESULTS_DIR}/gene_training.log"
run_ablation_mode "Gene Only (仅基因)" 2 "gene" "${GENE_LOG}"

# 汇总 Gene Only 结果
echo "" | tee -a "${GENE_LOG}"
echo "📊 汇总 Gene Only 结果..." | tee -a "${GENE_LOG}"
GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"

# 【加固】先等待所有后台任务完成（确保并行模式下文件已写入）
wait

python3 << 'EOF_SUMMARY' | tee -a "${GENE_LOG}"
import pandas as pd
import glob
import os
import re

results_dir = "${ABLRESULTS_DIR}/gene"
print(f"📁 搜索结果目录: {results_dir}")

dfs = []
missing_folds = []

# 遍历所有 fold 目录
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*")):
    fold_name = os.path.basename(fold_dir)
    fold_num = int(fold_name.split('_')[-1])

    # 【加固】优先查找 summary_partial_*.csv，再找 summary.csv
    summary_file = None
    partial_files = glob.glob(f"{fold_dir}/summary_partial_*.csv")
    if partial_files:
        # 取最新的 partial 文件（如果有多个）
        summary_file = max(partial_files, key=os.path.getmtime)
        print(f"  ✓ Fold {fold_num}: 使用 {os.path.basename(summary_file)}")
    elif os.path.exists(f"{fold_dir}/summary.csv"):
        summary_file = f"{fold_dir}/summary.csv"
        print(f"  ✓ Fold {fold_num}: 使用 summary.csv")
    else:
        print(f"  ✗ Fold {fold_num}: 文件缺失")
        missing_folds.append(fold_num)
        continue

    # 读取文件
    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except Exception as e:
        print(f"  ✗ Fold {fold_num}: 读取失败 - {e}")
        missing_folds.append(fold_num)

if missing_folds:
    print(f"⚠️  缺失折数: {missing_folds}")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv('${GENE_SUMMARY}', index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Gene Only 汇总完成: {len(dfs)}/{len(dfs) + len(missing_folds)} 折成功')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 没有任何折的结果文件可用')
    # 创建空文件避免后续错误
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv('${GENE_SUMMARY}', index=False)
EOF_SUMMARY

echo "  └─ 汇总完成: ${GENE_SUMMARY}" | tee -a "${GENE_LOG}"

# ====================================================================
# 2. Text Only 模式 (ab_model=1)
# ====================================================================
TEXT_LOG="${ABLRESULTS_DIR}/text_training.log"
run_ablation_mode "Text Only (仅文本)" 1 "text" "${TEXT_LOG}"

# 汇总 Text Only 结果
echo "" | tee -a "${TEXT_LOG}"
echo "📊 汇总 Text Only 结果..." | tee -a "${TEXT_LOG}"
TEXT_SUMMARY="${ABLRESULTS_DIR}/text/summary.csv"

# 【加固】先等待所有后台任务完成
wait

python3 << 'EOF_SUMMARY' | tee -a "${TEXT_LOG}"
import pandas as pd
import glob
import os
import re

results_dir = "${ABLRESULTS_DIR}/text"
print(f"📁 搜索结果目录: {results_dir}")

dfs = []
missing_folds = []

# 遍历所有 fold 目录
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*")):
    fold_name = os.path.basename(fold_dir)
    fold_num = int(fold_name.split('_')[-1])

    # 【加固】优先查找 summary_partial_*.csv，再找 summary.csv
    summary_file = None
    partial_files = glob.glob(f"{fold_dir}/summary_partial_*.csv")
    if partial_files:
        summary_file = max(partial_files, key=os.path.getmtime)
        print(f"  ✓ Fold {fold_num}: 使用 {os.path.basename(summary_file)}")
    elif os.path.exists(f"{fold_dir}/summary.csv"):
        summary_file = f"{fold_dir}/summary.csv"
        print(f"  ✓ Fold {fold_num}: 使用 summary.csv")
    else:
        print(f"  ✗ Fold {fold_num}: 文件缺失")
        missing_folds.append(fold_num)
        continue

    # 读取文件
    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except Exception as e:
        print(f"  ✗ Fold {fold_num}: 读取失败 - {e}")
        missing_folds.append(fold_num)

if missing_folds:
    print(f"⚠️  缺失折数: {missing_folds}")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv('${TEXT_SUMMARY}', index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Text Only 汇总完成: {len(dfs)}/{len(dfs) + len(missing_folds)} 折成功')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv('${TEXT_SUMMARY}', index=False)
EOF_SUMMARY

echo "  └─ 汇总完成: ${TEXT_SUMMARY}" | tee -a "${TEXT_LOG}"

# ====================================================================
# 3. Fusion 模式 (ab_model=3)
# ====================================================================
FUSION_LOG="${ABLRESULTS_DIR}/fusion_training.log"
run_ablation_mode "Fusion (多模态融合)" 3 "fusion" "${FUSION_LOG}"

# 汇总 Fusion 结果
echo "" | tee -a "${FUSION_LOG}"
echo "📊 汇总 Fusion 结果..." | tee -a "${FUSION_LOG}"
FUSION_SUMMARY="${ABLRESULTS_DIR}/fusion/summary.csv"

# 【加固】先等待所有后台任务完成
wait

python3 << 'EOF_SUMMARY' | tee -a "${FUSION_LOG}"
import pandas as pd
import glob
import os
import re

results_dir = "${ABLRESULTS_DIR}/fusion"
print(f"📁 搜索结果目录: {results_dir}")

dfs = []
missing_folds = []

# 遍历所有 fold 目录
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*")):
    fold_name = os.path.basename(fold_dir)
    fold_num = int(fold_name.split('_')[-1])

    # 【加固】优先查找 summary_partial_*.csv，再找 summary.csv
    summary_file = None
    partial_files = glob.glob(f"{fold_dir}/summary_partial_*.csv")
    if partial_files:
        summary_file = max(partial_files, key=os.path.getmtime)
        print(f"  ✓ Fold {fold_num}: 使用 {os.path.basename(summary_file)}")
    elif os.path.exists(f"{fold_dir}/summary.csv"):
        summary_file = f"{fold_dir}/summary.csv"
        print(f"  ✓ Fold {fold_num}: 使用 summary.csv")
    else:
        print(f"  ✗ Fold {fold_num}: 文件缺失")
        missing_folds.append(fold_num)
        continue

    # 读取文件
    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except Exception as e:
        print(f"  ✗ Fold {fold_num}: 读取失败 - {e}")
        missing_folds.append(fold_num)

if missing_folds:
    print(f"⚠️  缺失折数: {missing_folds}")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv('${FUSION_SUMMARY}', index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Fusion 汇总完成: {len(dfs)}/{len(dfs) + len(missing_folds)} 折成功')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv('${FUSION_SUMMARY}', index=False)
EOF_SUMMARY

echo "  └─ 汇总完成: ${FUSION_SUMMARY}" | tee -a "${FUSION_LOG}"

# ====================================================================
# 生成最终对比表格
# ====================================================================
echo ""
echo "=============================================="
echo "📈 生成最终对比表格"
echo "=============================================="

FINAL_CSV="${ABLRESULTS_DIR}/final_comparison.csv"

# 【加固】等待所有后台任务完成（理论上应该已经完成了，但以防万一）
wait

python3 << 'EOF_FINAL' | tee -a "${ABLRESULTS_DIR}/ablation_summary.log"
import pandas as pd
import numpy as np
import glob
import os

study = "${STUDY}"
ablation_dir = f"results/ablation/{study}"

# 读取三个模式的汇总结果
gene_dir = f"{ablation_dir}/gene"
text_dir = f"{ablation_dir}/text"
fusion_dir = f"{ablation_dir}/fusion"

print("📊 读取各模式汇总结果...")

def read_summary_csv(directory, mode_name):
    """读取单个模式的汇总CSV，兼容缺失情况"""
    summary_file = f"{directory}/summary.csv"
    if not os.path.exists(summary_file):
        print(f"  ⚠️  {mode_name}: {summary_file} 不存在")
        return None

    try:
        df = pd.read_csv(summary_file)
        print(f"  ✓ {mode_name}: 成功读取 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"  ✗ {mode_name}: 读取失败 - {e}")
        return None

# 读取三个模式的汇总
gene_summary = read_summary_csv(gene_dir, "Gene Only")
text_summary = read_summary_csv(text_dir, "Text Only")
fusion_summary = read_summary_csv(fusion_dir, "Fusion")

# 构建对比表格
comparison_data = []

# 获取所有 fold 编号
all_folds = set()
for summary in [gene_summary, text_summary, fusion_summary]:
    if summary is not None and 'fold' in summary.columns:
        all_folds.update(summary['fold'].tolist())

all_folds = sorted(all_folds)

print(f"\n📊 构建对比表格 (共 {len(all_folds)} 折)...")

for fold in all_folds:
    row = {'Fold': fold}

    # Gene Only
    if gene_summary is not None and 'fold' in gene_summary.columns:
        gene_row = gene_summary[gene_summary['fold'] == fold]
        if not gene_row.empty:
            row['Gene_C_Index'] = gene_row['val_cindex'].values[0]
        else:
            row['Gene_C_Index'] = np.nan
    else:
        row['Gene_C_Index'] = np.nan

    # Text Only
    if text_summary is not None and 'fold' in text_summary.columns:
        text_row = text_summary[text_summary['fold'] == fold]
        if not text_row.empty:
            row['Text_C_Index'] = text_row['val_cindex'].values[0]
        else:
            row['Text_C_Index'] = np.nan
    else:
        row['Text_C_Index'] = np.nan

    # Fusion
    if fusion_summary is not None and 'fold' in fusion_summary.columns:
        fusion_row = fusion_summary[fusion_summary['fold'] == fold]
        if not fusion_row.empty:
            row['Fusion_C_Index'] = fusion_row['val_cindex'].values[0]
        else:
            row['Fusion_C_Index'] = np.nan
    else:
        row['Fusion_C_Index'] = np.nan

    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("${FINAL_CSV}", index=False)

# 计算平均值（忽略 NaN）
gene_mean = comparison_df['Gene_C_Index'].mean()
text_mean = comparison_df['Text_C_Index'].mean()
fusion_mean = comparison_df['Fusion_C_Index'].mean()

# 打印结果
print("\n" + "="*60)
print("📊 多模态消融实验结果汇总")
print("="*60)
print(comparison_df.to_string(index=False))
print("="*60)
print(f"\n🎯 平均 C-Index:")
print(f"   • Gene Only (仅基因): {gene_mean:.4f}" if not np.isnan(gene_mean) else "   • Gene Only (仅基因): N/A")
print(f"   • Text Only (仅文本): {text_mean:.4f}" if not np.isnan(text_mean) else "   • Text Only (仅文本): N/A")
print(f"   • Fusion (多模态融合): {fusion_mean:.4f}" if not np.isnan(fusion_mean) else "   • Fusion (多模态融合): N/A")
print(f"\n📁 结果已保存到: ${FINAL_CSV}")
print("="*60)

# 计算提升百分比
if gene_mean and not np.isnan(gene_mean) and gene_mean > 0:
    fusion_improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
    print(f"\n📈 Fusion 相对于 Gene Only 的提升: {fusion_improvement:+.2f}%")
if text_mean and not np.isnan(text_mean) and text_mean > 0:
    fusion_vs_text = ((fusion_mean - text_mean) / text_mean) * 100
    print(f"📈 Fusion 相对于 Text Only 的提升: {fusion_vs_text:+.2f}%")

EOF_FINAL

echo ""
echo "✅ 消融实验完成！"
echo "=============================================="
echo "📁 结果目录: ${ABLRESULTS_DIR}"
echo "📊 对比表格: ${FINAL_CSV}"
echo "⚡ 并行任务数: ${MAX_JOBS}"
echo "=============================================="
