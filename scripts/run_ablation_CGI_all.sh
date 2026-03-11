#!/bin/bash

# ====================================================================
# 完整消融实验脚本（CGI版）
# 运行三种模式：Text Only, Gene Only, Fusion
# 支持多种癌症种类
# ====================================================================
#
# 运行方法:
#   bash scripts/run_ablation_CGI_all.sh           # 运行所有癌症种类
#   bash scripts/run_ablation_CGI_all.sh brca       # 只运行 brca
#
# 前置要求:
#   1. 运行 preprocess_test.py 生成 CGI 数据和划分
#   2. 运行 CGI 算法筛选基因
#   3. 运行 extract_features.py 生成基因特征文件
#
# ====================================================================

# 癌症种类列表
ALL_CANCERS=("brca" "blca" "hnsc" "stad" "coadread")

# ==================== 数据路径配置 ====================

# 临床标签文件（动态生成）
get_label_file() {
    local study=$1
    echo "datasets_csv/clinical_data/tcga_${study}_clinical.csv"
}

# 交叉验证划分文件（使用CGI重新划分的版本）
get_split_dir() {
    local study=$1
    echo "splits/CGI_nested_cv/${study}"
}

# CGI筛选的基因特征文件
get_feature_dir() {
    local study=$1
    echo "preprocessing/CGI/data/${study}_found_genes"
}

get_feature_file() {
    local study=$1
    echo "${study}_found_Genes_fold"
}

# RNA原始数据
get_omics_dir() {
    local study=$1
    echo "datasets_csv/raw_rna_data/combine/${study}"
}

# PT数据文件
get_data_root_dir() {
    local study=$1
    echo "data/${study}/pt_files"
}

# 训练超参数
SEED=42
K_FOLDS=5
EPOCHS=20
LR=0.00005
MAX_JOBS=3  # 每种模式并发数

# ==================== 辅助函数 ====================

# 检查特征文件
check_features() {
    local study=$1
    local feature_dir=$(get_feature_dir $study)
    local feature_file=$(get_feature_file $study)
    local all_exist=true

    echo "🔍 检查 ${study^^} 的 CGI 基因特征文件..."
    for fold in $(seq 0 $((K_FOLDS-1))); do
        local file="${feature_dir}/${feature_file}${fold}.csv"
        if [ -f "$file" ]; then
            echo "   ✓ Fold ${fold}: $(basename $file) 存在"
        else
            echo "   ✗ Fold ${fold}: $(basename $file) 缺失!"
            all_exist=false
        fi
    done
    if [ "$all_exist" = false ]; then
        echo "❌ 错误: ${study} 特征文件不完整"
        return 1
    fi
    echo "✅ ${study} CGI 特征文件检查通过"
    return 0
}

# 运行单种模式
run_mode() {
    local study=$1
    local mode_name=$2
    local ab_model=$3
    local results_subdir=$4
    local log_file=$5

    local split_dir=$(get_split_dir $study)
    local label_file=$(get_label_file $study)
    local omics_dir=$(get_omics_dir $study)
    local data_root_dir=$(get_data_root_dir $study)
    local results_dir="results/ablation/${study}"

    echo "" | tee -a "${log_file}"
    echo "==============================================" | tee -a "${log_file}"
    echo "🧬 ${study^^} - ${mode_name}" | tee -a "${log_file}"
    echo "==============================================" | tee -a "${log_file}"

    local fold=0
    local running_jobs=0

    for fold in $(seq 0 $((K_FOLDS-1))); do
        local RESULTS_DIR="${results_dir}/${results_subdir}/fold_${fold}"
        local fold_log="${RESULTS_DIR}/training.log"
        mkdir -p "${RESULTS_DIR}"

        echo "  └─ 启动 Fold ${fold}..." | tee -a "${log_file}"

        python3 main.py \
            --study tcga_${study} \
            --k_start ${fold} \
            --k_end $((fold + 1)) \
            --split_dir "${split_dir}" \
            --results_dir "${RESULTS_DIR}" \
            --seed ${SEED} \
            --label_file "${label_file}" \
            --task survival \
            --n_classes 4 \
            --modality snn \
            --omics_dir "${omics_dir}" \
            --data_root_dir "${data_root_dir}" \
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

# 汇总结果
summarize_results() {
    local study=$1
    local mode_name=$2
    local results_subdir=$3
    local summary_env_var=$4

    local results_dir="results/ablation/${study}/${results_subdir}"
    local summary_path="${results_dir}/summary.csv"

    wait

    python3 << EOF
import pandas as pd
import glob
import os

results_dir = "${results_dir}"
summary_path = "${summary_path}"

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
    print(f'✅ ${mode_name} 汇总: {len(dfs)}/5 折成功, 平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 无可用结果')
    pd.DataFrame(columns=['fold', 'val_cindex']).to_csv(summary_path, index=False)
EOF
}

# 运行单个癌症种类
run_single_cancer() {
    local study=$1
    local today=$2
    local main_log=$3

    echo "" | tee -a "${main_log}"
    echo "################################################################" | tee -a "${main_log}"
    echo "### 开始处理癌症类型: ${study^^}" | tee -a "${main_log}"
    echo "################################################################"

    # 检查特征文件
    if ! check_features $study; then
        echo "❌ 跳过 ${study}，特征文件缺失" | tee -a "${main_log}"
        return 1
    fi

    local results_dir="results/ablation/${study}"
    local log_dir="log/${today}/${study}"

    # 创建目录
    mkdir -p "${log_dir}" "report" "${results_dir}"/{text,gene,fusion}

    echo "🚀 开始完整消融实验: ${study}" | tee -a "${main_log}"
    echo "📅 日期: ${today}" | tee -a "${main_log}"
    echo "📁 日志: ${main_log}" | tee -a "${main_log}"
    echo "==============================================" | tee -a "${main_log}"

    # ==================== 1. Text Only ====================
    local text_log="${results_dir}/text_training.log"
    run_mode $study "Text Only (仅文本)" 1 "text" "${text_log}"
    summarize_results $study "Text Only" "text"

    # ==================== 2. Gene Only ====================
    local gene_log="${results_dir}/gene_training.log"
    run_mode $study "Gene Only (仅基因)" 2 "gene" "${gene_log}"
    summarize_results $study "Gene Only" "gene"

    # ==================== 3. Fusion ====================
    local fusion_log="${results_dir}/fusion_training.log"
    run_mode $study "Fusion (基因+文本)" 3 "fusion" "${fusion_log}"
    summarize_results $study "Fusion" "fusion"

    # ==================== 生成对比表格 ====================
    echo ""
    echo "=============================================="
    echo "📈 生成对比表格"
    echo "=============================================="

    local final_csv="${results_dir}/final_comparison.csv"
    local report_csv="report/${today}_${study}_ablation_all.csv"

    wait

    python3 << EOF
import pandas as pd
import numpy as np
import os

study = "${study}"
ablation_dir = f"results/ablation/{study}"
final_csv = "${final_csv}"
report_csv = "${report_csv}"

text_summary = pd.read_csv(f"{ablation_dir}/text/summary.csv") if os.path.exists(f"{ablation_dir}/text/summary.csv") else None
gene_summary = pd.read_csv(f"{ablation_dir}/gene/summary.csv") if os.path.exists(f"{ablation_dir}/gene/summary.csv") else None
fusion_summary = pd.read_csv(f"{ablation_dir}/fusion/summary.csv") if os.path.exists(f"{ablation_dir}/fusion/summary.csv") else None

comparison_data = []
for fold in range(5):
    row = {'Fold': fold}
    if text_summary is not None:
        text_val = text_summary[text_summary['fold'] == fold]['val_cindex'].values
        row['Text_C_Index'] = text_val[0] if len(text_val) > 0 else np.nan
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

text_mean = df['Text_C_Index'].mean() if 'Text_C_Index' in df.columns else 0
gene_mean = df['Gene_C_Index'].mean() if 'Gene_C_Index' in df.columns else 0
fusion_mean = df['Fusion_C_Index'].mean() if 'Fusion_C_Index' in df.columns else 0

print("\n" + "="*60)
print("📊 完整消融实验结果汇总 - " + study.upper())
print("="*60)
print(df.to_string(index=False))
print("="*60)
print(f"\n🎯 平均 C-Index:")
print(f"   Text Only: {text_mean:.4f}")
print(f"   Gene Only: {gene_mean:.4f}")
print(f"   Fusion:    {fusion_mean:.4f}")
print(f"\n📁 结果: {final_csv}")
print("="*60)
EOF

    echo "" | tee -a "${main_log}"
    echo "✅ ${study} 消融实验完成！" | tee -a "${main_log}"
}

# ==================== 主程序 ====================

TODAY=$(date +%Y-%m-%d)
LOG_DIR="log/${TODAY}"

# 检查参数 - 如果没有参数则运行所有癌症种类
if [ -z "$1" ]; then
    echo "未指定癌症种类，将运行所有癌症种类..."
    CANCERS_TO_RUN=("${ALL_CANCERS[@]}")
else
    # 解析参数
    CANCERS_TO_RUN=("$@")
fi

echo "🚀 开始完整消融实验" | tee -a "${LOG_DIR}/ablation_all.log"
echo "📅 日期: ${TODAY}" | tee -a "${LOG_DIR}/ablation_all.log"
echo "📋 癌症种类: ${CANCERS_TO_RUN[*]}" | tee -a "${LOG_DIR}/ablation_all.log"
echo "==============================================" | tee -a "${LOG_DIR}/ablation_all.log"

# 验证癌症种类
for study in "${CANCERS_TO_RUN[@]}"; do
    valid=false
    for valid_cancer in "${ALL_CANCERS[@]}"; do
        if [ "$study" = "$valid_cancer" ]; then
            valid=true
            break
        fi
    done
    if [ "$valid" = false ]; then
        echo "❌ 错误: 未知癌症种类 '$study'"
        echo "支持的癌症种类: ${ALL_CANCERS[*]}"
        exit 1
    fi
done

# 运行每种癌症
LOG_DIR="log/${TODAY}"
for study in "${CANCERS_TO_RUN[@]}"; do
    MAIN_LOG="${LOG_DIR}/${study}/ablation_all.log"
    run_single_cancer $study $TODAY "${MAIN_LOG}"
done

echo ""
echo "################################################################"
echo "### ✅ 所有癌症种类消融实验完成！"
echo "################################################################"
echo "📁 结果目录: results/ablation/"
echo "📋 报告目录: report/"

# 保存完整日志
echo ""
echo "################################################################"
echo "### ✅ 所有癌症种类消融实验完成！" | tee -a "${LOG_DIR}/ablation_all.log"
echo "################################################################" | tee -a "${LOG_DIR}/ablation_all.log"
echo "📁 结果目录: results/ablation/" | tee -a "${LOG_DIR}/ablation_all.log"
echo "📋 报告目录: report/" | tee -a "${LOG_DIR}/ablation_all.log"
