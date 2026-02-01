#!/bin/bash

# ====================================================================
# 多模态消融实验脚本（使用 mRMR+Stage2 精炼基因特征）
# 对比 Gene Only、Text Only、Fusion 三种模式的性能
# ====================================================================

# 检查参数
if [ -z "$1" ]; then
    echo "❌ 用法: bash run_ablation_study_mrmr_stage2.sh <癌种简称>"
    echo "   例如: bash run_ablation_study_mrmr_stage2.sh blca"
    echo ""
    echo "📝 说明: 本脚本使用 mRMR+Stage2 (PC算法) 精炼后的基因特征"
    echo "   特征路径: features/mrmr_stage2_{study}/fold_{fold}_genes.csv"
    exit 1
fi

STUDY=$1
TODAY=$(date +%Y-%m-%d)

# ==================== 数据路径配置 ====================
# 【重要】所有数据路径统一在此处配置，请勿在代码中硬编码路径
# 注意：必须放在 STUDY=$1 之后，确保变量已定义

# 1. 训练数据路径
# ----------------
# 临床标签文件: 包含患者生存信息、文本报告等
LABEL_FILE="datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv"

# 交叉验证划分文件: 5折嵌套交叉验证的划分
SPLIT_DIR="splits/nested_cv/${STUDY}"

# 【关键修改】mRMR+Stage2 精炼基因特征文件
# 路径格式: features/mrmr_stage2_{study}/fold_{fold}_genes.csv
FEATURE_DIR="features/mrmr_stage2_${STUDY}"

# RNA原始数据: 用于加载完整的RNA表达数据（备选）
OMICS_DIR="datasets_csv/raw_rna_data/combine/${STUDY}"

# PT数据文件: WSI图像的预提取特征
DATA_ROOT_DIR="data/${STUDY}/pt_files"

# 2. 模型与输出路径
# ----------------
# BioBERT预训练模型: 文本编码器（预训练，无需下载）
BIOBERT_DIR="biobert"

# 【修改】消融实验结果目录: 添加 mrmr_stage2 标识以区分
ABLRESULTS_DIR="results/ablation_mrmr_stage2/${STUDY}"

# 日志目录
LOG_DIR="log/${TODAY}/${STUDY}_mrmr_stage2"

# 报告目录
REPORT_DIR="report"

# 3. 训练超参数
# ----------------
SEED=42                    # 随机种子（保证可复现性）
K_FOLDS=5                  # 交叉验证折数
EPOCHS=20                  # 训练轮数
LR=0.00005                 # 学习率
MAX_JOBS=3                 # 最大并行任务数（避免GPU内存不足）
# =========================================================

# 创建日志和报告目录
mkdir -p "${LOG_DIR}" "${REPORT_DIR}"

# 配置日志文件
MAIN_LOG="${LOG_DIR}/ablation_study.log"

echo "🚀 开始多模态消融实验: ${STUDY} (mRMR+Stage2)"
echo "📁 特征路径: ${FEATURE_DIR}"
echo "📁 日志目录: ${LOG_DIR}"
echo "📁 结果将保存到: ${ABLRESULTS_DIR}"
echo "=============================================="
echo "" | tee -a "${MAIN_LOG}"
echo "==============================================" | tee -a "${MAIN_LOG}"
echo "🚀 开始多模态消融实验: ${STUDY} (mRMR+Stage2)" | tee -a "${MAIN_LOG}"
echo "📁 特征路径: ${FEATURE_DIR}" | tee -a "${MAIN_LOG}"
echo "📁 日志目录: ${LOG_DIR}" | tee -a "${MAIN_LOG}"
echo "==============================================" | tee -a "${MAIN_LOG}"

export ABLRESULTS_DIR  # 导出变量供Python子进程使用

# 创建结果目录结构
mkdir -p "${ABLRESULTS_DIR}"/{gene,text,fusion}

echo "📁 结果目录: ${ABLRESULTS_DIR}"

export STUDY           # 导出STUDY变量
export LABEL_FILE      # 导出标签文件路径（供Python使用）
export SPLIT_DIR       # 导出划分文件路径（供Python使用）
export ABLRESULTS_DIR  # 导出结果目录路径（供Python使用）

# 【修改】检查 mRMR+Stage2 特征文件是否存在
check_features() {
    local study=$1
    local all_exist=true
    echo "🔍 检查 ${study^^} 的 mRMR+Stage2 特征文件..."
    for fold in $(seq 0 $((K_FOLDS-1))); do
        local file="${FEATURE_DIR}/fold_${fold}_genes.csv"
        if [ -f "$file" ]; then
            echo "   ✓ Fold ${fold}: $(basename $file) 存在"
        else
            echo "   ✗ Fold ${fold}: $(basename $file) 缺失!"
            all_exist=false
        fi
    done
    if [ "$all_exist" = false ]; then
        echo "❌ 错误: ${study} mRMR+Stage2 特征文件不完整，跳过训练"
        echo "请先运行:"
        echo "  1. python preprocessing/CPCG_algo/stage0/run_mrmr.py --study ${study} --fold all ..."
        echo "  2. bash scripts/quick_stage2_refine.sh ${study}"
        return 1
    fi
    echo "✅ ${study} mRMR+Stage2 特征文件检查通过"
    return 0
}

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

# 【新增】先检查特征文件
if ! check_features "${STUDY}"; then
    echo "⚠️  跳过 ${STUDY} 的消融实验"
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
run_ablation_mode "Gene Only (仅基因 - mRMR+Stage2)" 2 "gene" "${GENE_LOG}"

# 汇总 Gene Only 结果
echo "" | tee -a "${GENE_LOG}"
echo "📊 汇总 Gene Only 结果..." | tee -a "${GENE_LOG}"
GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"
export GENE_SUMMARY

# 【加固】先等待所有后台任务完成（确保并行模式下文件已写入）
wait

# 【修复】使用 os.environ 获取环境变量
python3 << 'EOF_SUMMARY' | tee -a "${GENE_LOG}"
import pandas as pd
import glob
import os
import sys

# 【修复】只使用单层路径
base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'gene')
gene_summary_path = os.environ.get('GENE_SUMMARY', '')

print(f"📁 搜索结果目录: {results_dir}")

dfs = []
missing_folds = []

# 【修复】递归查找所有子目录下的summary文件
# main.py会在fold_X目录下创建更深的子文件夹
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    # 只处理目录
    if not os.path.isdir(fold_dir):
        continue

    fold_name = os.path.basename(fold_dir)
    # 提取fold编号
    try:
        fold_num = int(fold_name.split('_')[-1])
    except (ValueError, IndexError):
        continue

    print(f"  📂 检查 Fold {fold_num} 目录: {fold_dir}")

    # 【修复】递归查找summary文件
    summary_file = None
    partial_files = glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)

    if partial_files:
        # 取最新的partial文件
        summary_file = max(partial_files, key=os.path.getmtime)
        print(f"     ✓ Fold {fold_num}: 使用 {os.path.basename(summary_file)}")
    else:
        # 尝试根目录的summary.csv
        root_summary = f"{fold_dir}/summary.csv"
        if os.path.exists(root_summary):
            summary_file = root_summary
            print(f"     ✓ Fold {fold_num}: 使用 summary.csv")
        else:
            print(f"     ✗ Fold {fold_num}: 文件缺失")
            missing_folds.append(fold_num)
            continue

    # 读取文件
    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except Exception as e:
        print(f"     ✗ Fold {fold_num}: 读取失败 - {e}")
        missing_folds.append(fold_num)

if missing_folds:
    print(f"⚠️  缺失折数: {missing_folds}")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(gene_summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Gene Only (mRMR+Stage2) 汇总完成: {len(dfs)}/{len(dfs) + len(missing_folds)} 折成功')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 没有任何折的结果文件可用')
    # 创建空文件避免后续错误
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv(gene_summary_path, index=False)
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
export TEXT_SUMMARY

# 【加固】先等待所有后台任务完成
wait

python3 << 'EOF_SUMMARY' | tee -a "${TEXT_LOG}"
import pandas as pd
import glob
import os
import sys

# 【修复】只使用单层路径
base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'text')
text_summary_path = os.environ.get('TEXT_SUMMARY', '')

print(f"📁 搜索结果目录: {results_dir}")

dfs = []
missing_folds = []

for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    if not os.path.isdir(fold_dir):
        continue

    fold_name = os.path.basename(fold_dir)
    try:
        fold_num = int(fold_name.split('_')[-1])
    except (ValueError, IndexError):
        continue

    print(f"  📂 检查 Fold {fold_num} 目录: {fold_dir}")

    summary_file = None
    partial_files = glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)

    if partial_files:
        summary_file = max(partial_files, key=os.path.getmtime)
        print(f"     ✓ Fold {fold_num}: 使用 {os.path.basename(summary_file)}")
    else:
        root_summary = f"{fold_dir}/summary.csv"
        if os.path.exists(root_summary):
            summary_file = root_summary
            print(f"     ✓ Fold {fold_num}: 使用 summary.csv")
        else:
            print(f"     ✗ Fold {fold_num}: 文件缺失")
            missing_folds.append(fold_num)
            continue

    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except Exception as e:
        print(f"     ✗ Fold {fold_num}: 读取失败 - {e}")
        missing_folds.append(fold_num)

if missing_folds:
    print(f"⚠️  缺失折数: {missing_folds}")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(text_summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Text Only 汇总完成: {len(dfs)}/{len(dfs) + len(missing_folds)} 折成功')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv(text_summary_path, index=False)
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
export FUSION_SUMMARY

# 【加固】先等待所有后台任务完成
wait

python3 << 'EOF_SUMMARY' | tee -a "${FUSION_LOG}"
import pandas as pd
import glob
import os
import sys

# 【修复】只使用单层路径
base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'fusion')
fusion_summary_path = os.environ.get('FUSION_SUMMARY', '')

print(f"📁 搜索结果目录: {results_dir}")

dfs = []
missing_folds = []

for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    if not os.path.isdir(fold_dir):
        continue

    fold_name = os.path.basename(fold_dir)
    try:
        fold_num = int(fold_name.split('_')[-1])
    except (ValueError, IndexError):
        continue

    print(f"  📂 检查 Fold {fold_num} 目录: {fold_dir}")

    summary_file = None
    partial_files = glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)

    if partial_files:
        summary_file = max(partial_files, key=os.path.getmtime)
        print(f"     ✓ Fold {fold_num}: 使用 {os.path.basename(summary_file)}")
    else:
        root_summary = f"{fold_dir}/summary.csv"
        if os.path.exists(root_summary):
            summary_file = root_summary
            print(f"     ✓ Fold {fold_num}: 使用 summary.csv")
        else:
            print(f"     ✗ Fold {fold_num}: 文件缺失")
            missing_folds.append(fold_num)
            continue

    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except Exception as e:
        print(f"     ✗ Fold {fold_num}: 读取失败 - {e}")
        missing_folds.append(fold_num)

if missing_folds:
    print(f"⚠️  缺失折数: {missing_folds}")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(fusion_summary_path, index=False)
    mean_cindex = result['val_cindex'].mean()
    print(f'✅ Fusion 汇总完成: {len(dfs)}/{len(dfs) + len(missing_folds)} 折成功')
    print(f'   平均 C-Index: {mean_cindex:.4f}')
else:
    print('❌ 错误: 没有任何折的结果文件可用')
    pd.DataFrame(columns=['folds', 'val_cindex']).to_csv(fusion_summary_path, index=False)
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
export FINAL_CSV
REPORT_CSV="report/${TODAY}_${STUDY}_ablation_mrmr_stage2_comparison.csv"

# 【加固】等待所有后台任务完成
wait

python3 << 'EOF_FINAL' | tee -a "${ABLRESULTS_DIR}/ablation_summary.log"
import pandas as pd
import numpy as np
import glob
import os

ablation_dir = os.environ.get('ABLRESULTS_DIR', '')

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
gene_summary = read_summary_csv(gene_dir, "Gene Only (mRMR+Stage2)")
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

    # Gene Only (mRMR+Stage2)
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
final_csv_path = os.environ.get('FINAL_CSV', '')
comparison_df.to_csv(final_csv_path, index=False)

# 计算平均值（忽略 NaN）
gene_mean = comparison_df['Gene_C_Index'].mean()
text_mean = comparison_df['Text_C_Index'].mean()
fusion_mean = comparison_df['Fusion_C_Index'].mean()

# 打印结果
print("\n" + "="*60)
print("📊 多模态消融实验结果汇总 (mRMR+Stage2)")
print("="*60)
print(comparison_df.to_string(index=False))
print("="*60)
print(f"\n🎯 平均 C-Index:")
print(f"   • Gene Only (mRMR+Stage2): {gene_mean:.4f}" if not np.isnan(gene_mean) else "   • Gene Only (mRMR+Stage2): N/A")
print(f"   • Text Only (仅文本): {text_mean:.4f}" if not np.isnan(text_mean) else "   • Text Only (仅文本): N/A")
print(f"   • Fusion (多模态融合): {fusion_mean:.4f}" if not np.isnan(fusion_mean) else "   • Fusion (多模态融合): N/A")
print(f"\n📁 结果已保存到: {final_csv_path}")
print("="*60)

# 计算提升百分比
if gene_mean and not np.isnan(gene_mean) and gene_mean > 0:
    fusion_improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
    print(f"\n📈 Fusion 相对于 Gene Only (mRMR+Stage2) 的提升: {fusion_improvement:+.2f}%")
if text_mean and not np.isnan(text_mean) and text_mean > 0:
    fusion_vs_text = ((fusion_mean - text_mean) / text_mean) * 100
    print(f"📈 Fusion 相对于 Text Only 的提升: {fusion_vs_text:+.2f}%")

EOF_FINAL

echo ""
echo "✅ 消融实验完成！"
echo "=============================================="
echo "📁 结果目录: ${ABLRESULTS_DIR}"
echo "📊 对比表格: ${FINAL_CSV}"
echo "📋 报告文件: ${REPORT_CSV}"
echo "⚡ 并行任务数: ${MAX_JOBS}"
echo "🧬 特征类型: mRMR+Stage2 (PC算法精炼)"
echo "=============================================="

# 【新增】复制最终结果到 report 目录
if [ -f "${FINAL_CSV}" ]; then
    cp "${FINAL_CSV}" "${REPORT_CSV}"
    echo "📁 已复制结果到: ${REPORT_CSV}"
fi
