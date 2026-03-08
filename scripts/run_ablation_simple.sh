#!/bin/bash

# ====================================================================
# 简化消融实验脚本（CGI版）
# 只运行 Gene Only 和 Fusion 两种模式
# ====================================================================
#
# 运行方法:
#   bash scripts/run_ablation_simple.sh coadread
#
# 前置要求:
#   1. 运行 preprocess_test.py 生成 CGI 数据和划分
#   2. 运行 CGI 算法筛选基因
#   3. 运行 extract_features.py 生成基因特征文件
#
# ====================================================================

# 检查参数
if [ -z "$1" ]; then
    echo "用法: bash scripts/run_ablation_simple.sh <癌种简称>"
    echo "例如: bash scripts/run_ablation_simple.sh coadread"
    exit 1
fi

STUDY=$1
TODAY=$(date +%Y-%m-%d)

# ==================== 数据路径配置 ====================

# 临床标签文件
LABEL_FILE="datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv"

# 交叉验证划分文件（使用CGI重新划分的版本）
SPLIT_DIR="splits/CGI_nested_cv/${STUDY}"

# CGI筛选的基因特征文件
FEATURE_DIR="preprocessing/CGI/data/${STUDY}_found_genes"
FEATURE_FILE="${STUDY}_found_Genes_fold"

# RNA原始数据
OMICS_DIR="datasets_csv/raw_rna_data/combine/${STUDY}"

# PT数据文件
DATA_ROOT_DIR="data/${STUDY}/pt_files"

# 模型与输出路径
BIOBERT_DIR="biobert"
ABLRESULTS_DIR="results/ablation/${STUDY}"
LOG_DIR="log/${TODAY}/${STUDY}"
REPORT_DIR="report"

# 训练超参数
SEED=42
K_FOLDS=5
EPOCHS=20
LR=0.00005
MAX_JOBS=3  # 只运行两种模式，降低并发
# =========================================================

# 创建目录
mkdir -p "${LOG_DIR}" "${REPORT_DIR}" "${ABLRESULTS_DIR}"/{gene,fusion}

MAIN_LOG="${LOG_DIR}/ablation_simple.log"

echo "🚀 开始简化消融实验: ${STUDY}"
echo "📅 日期: ${TODAY}"
echo "📁 日志: ${MAIN_LOG}"
echo "=============================================="

export STUDY
export LABEL_FILE
export SPLIT_DIR
export ABLRESULTS_DIR

# 检查划分文件
if [ ! -d "${SPLIT_DIR}" ]; then
    echo "❌ 错误: 找不到划分文件目录 ${SPLIT_DIR}"
    echo "请先运行: python3 preprocessing/CGI/preprocess_test.py"
    exit 1
fi

# 检查特征文件
check_features() {
    local all_exist=true
    echo "🔍 检查 ${STUDY^^} 的 CGI 基因特征文件..."
    for fold in $(seq 0 $((K_FOLDS-1))); do
        local file="${FEATURE_DIR}/${FEATURE_FILE}${fold}.csv"
        if [ -f "$file" ]; then
            echo "   ✓ Fold ${fold}: $(basename $file) 存在"
        else
            echo "   ✗ Fold ${fold}: $(basename $file) 缺失!"
            all_exist=false
        fi
    done
    if [ "$all_exist" = false ]; then
        echo "❌ 错误: 特征文件不完整"
        echo "请先运行: python3 preprocessing/CGI/extract_features.py"
        exit 1
    fi
    echo "✅ CGI 特征文件检查通过"
}

check_features

# ==================== 辅助函数 ====================
run_mode() {
    local mode_name=$1
    local ab_model=$2
    local results_subdir=$3
    local log_file=$4

    echo "" | tee -a "${log_file}"
    echo "==============================================" | tee -a "${log_file}"
    echo "🧬 ${mode_name}" | tee -a "${log_file}"
    echo "==============================================" | tee -a "${log_file}"

    local fold=0
    local running_jobs=0

    for fold in $(seq 0 $((K_FOLDS-1))); do
        local RESULTS_DIR="${ABLRESULTS_DIR}/${results_subdir}/fold_${fold}"
        local fold_log="${RESULTS_DIR}/training.log"
        mkdir -p "${RESULTS_DIR}"

        echo "  └─ 启动 Fold ${fold}..." | tee -a "${log_file}"

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
            --omics_dir "${OMICS_DIR}" \
            --data_root_dir "${DATA_ROOT_DIR}" \
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

        if [ ${running_jobs} -ge ${MAX_JOBS} ]; then
            echo "  └─ 达到最大并发数 ${MAX_JOBS}，等待..." | tee -a "${log_file}"
            wait
            running_jobs=0
        fi
    done

    if [ ${running_jobs} -gt 0 ]; then
        wait
    fi

    echo "  └─ ${mode_name} 所有 Fold 完成" | tee -a "${log_file}"
}

# ==================== 1. Gene Only ====================
GENE_LOG="${ABLRESULTS_DIR}/gene_training.log"
run_mode "Gene Only (仅基因)" 2 "gene" "${GENE_LOG}"

# 汇总 Gene Only 结果
echo "" | tee -a "${GENE_LOG}"
echo "📊 汇总 Gene Only 结果..." | tee -a "${GENE_LOG}"
export GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"
wait

python3 << 'EOF' | tee -a "${GENE_LOG}"
import pandas as pd
import glob
import os
import time

base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'gene')
summary_path = os.environ.get('GENE_SUMMARY', '')

dfs = []
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    if not os.path.isdir(fold_dir):
        continue
    try:
        fold_num = int(fold_dir.split('_')[-1])
    except:
        continue

    # 优先查找 summary_partial_*.csv
    partial_files = glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)
    if partial_files:
        f = max(partial_files, key=os.path.getmtime)
        df = pd.read_csv(f)
        df['fold'] = fold_num
        dfs.append(df)
        print(f"  ✓ Fold {fold_num}: {os.path.basename(f)}")
    else:
        # 备选查找 summary.csv
        summary_files = glob.glob(f"{fold_dir}/**/summary.csv", recursive=True)
        if summary_files:
            f = max(summary_files, key=os.path.getmtime)
            df = pd.read_csv(f)
            # summary.csv 格式: 第一列是 folds
            if 'folds' in df.columns:
                df['fold'] = fold_num
            dfs.append(df)
            print(f"  ✓ Fold {fold_num}: {os.path.basename(f)} (from summary.csv)")
        else:
            print(f"  ✗ Fold {fold_num}: 结果文件缺失")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Gene Only 汇总: {len(dfs)}/5 折成功, 平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 无可用结果')
    pd.DataFrame(columns=['fold', 'val_cindex']).to_csv(summary_path, index=False)
EOF

# ==================== 2. Fusion ====================
FUSION_LOG="${ABLRESULTS_DIR}/fusion_training.log"
run_mode "Fusion (基因+文本)" 3 "fusion" "${FUSION_LOG}"

# 汇总 Fusion 结果
echo "" | tee -a "${FUSION_LOG}"
echo "📊 汇总 Fusion 结果..." | tee -a "${FUSION_LOG}"
export FUSION_SUMMARY="${ABLRESULTS_DIR}/fusion/summary.csv"
wait

python3 << 'EOF' | tee -a "${FUSION_LOG}"
import pandas as pd
import glob
import os

base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'fusion')
summary_path = os.environ.get('FUSION_SUMMARY', '')

dfs = []
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    if not os.path.isdir(fold_dir):
        continue
    try:
        fold_num = int(fold_dir.split('_')[-1])
    except:
        continue

    # 优先查找 summary_partial_*.csv
    partial_files = glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)
    if partial_files:
        f = max(partial_files, key=os.path.getmtime)
        df = pd.read_csv(f)
        df['fold'] = fold_num
        dfs.append(df)
        print(f"  ✓ Fold {fold_num}: {os.path.basename(f)}")
    else:
        # 备选查找 summary.csv
        summary_files = glob.glob(f"{fold_dir}/**/summary.csv", recursive=True)
        if summary_files:
            f = max(summary_files, key=os.path.getmtime)
            df = pd.read_csv(f)
            if 'folds' in df.columns:
                df['fold'] = fold_num
            dfs.append(df)
            print(f"  ✓ Fold {fold_num}: {os.path.basename(f)} (from summary.csv)")
        else:
            print(f"  ✗ Fold {fold_num}: 结果文件缺失")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Fusion 汇总: {len(dfs)}/5 折成功, 平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 无可用结果')
    pd.DataFrame(columns=['fold', 'val_cindex']).to_csv(summary_path, index=False)
EOF

# ==================== 生成对比表格 ====================
echo ""
echo "=============================================="
echo "📈 生成对比表格"
echo "=============================================="

export FINAL_CSV="${ABLRESULTS_DIR}/final_comparison.csv"
export REPORT_CSV="report/${TODAY}_${STUDY}_ablation_simple.csv"

wait

python3 << EOF | tee -a "${MAIN_LOG}"
import pandas as pd
import numpy as np
import os

study = os.environ.get('STUDY', '')
ablation_dir = f"results/ablation/{study}"
final_csv = os.environ.get('FINAL_CSV', '')
report_csv = os.environ.get('REPORT_CSV', '')

gene_summary = pd.read_csv(f"{ablation_dir}/gene/summary.csv") if os.path.exists(f"{ablation_dir}/gene/summary.csv") else None
fusion_summary = pd.read_csv(f"{ablation_dir}/fusion/summary.csv") if os.path.exists(f"{ablation_dir}/fusion/summary.csv") else None

comparison_data = []
for fold in range(5):
    row = {'Fold': fold}
    if gene_summary is not None:
        gene_val = gene_summary[gene_summary['fold'] == fold]['val_cindex'].values
        row['Gene_C_Index'] = gene_val[0] if len(gene_val) > 0 else np.nan
    if fusion_summary is not None:
        fusion_val = fusion_summary[fusion_summary['fold'] == fold]['val_cindex'].values
        row['Fusion_C_Index'] = fusion_val[0] if len(fusion_val) > 0 else np.nan
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
df.to_csv(final_csv, index=False)
df.to_csv(report_csv, index=False)

gene_mean = df['Gene_C_Index'].mean()
fusion_mean = df['Fusion_C_Index'].mean()

print("\n" + "="*50)
print("📊 简化消融实验结果汇总")
print("="*50)
print(df.to_string(index=False))
print("="*50)
print(f"\n🎯 平均 C-Index:")
print(f"   Gene Only: {gene_mean:.4f}")
print(f"   Fusion:     {fusion_mean:.4f}")
if gene_mean > 0:
    improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
    print(f"\n📈 Fusion 相对于 Gene Only: {improvement:+.2f}%")
print(f"\n📁 结果: ${FINAL_CSV}")
print("="*50)
EOF

echo ""
echo "✅ 简化消融实验完成！"
echo "📁 结果目录: ${ABLRESULTS_DIR}"
echo "📊 对比表格: ${FINAL_CSV}"
echo "📋 报告文件: ${REPORT_CSV}"
