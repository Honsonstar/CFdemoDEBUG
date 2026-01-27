#!/bin/bash

# ====================================================================
# 多模态消融实验脚本
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
# 1. Gene Only 模式 (ab_model=2)
# ====================================================================
echo ""
echo "=============================================="
echo "🧬 模式1/3: Gene Only (仅基因)"
echo "=============================================="

for fold in $(seq 0 $((K_FOLDS-1))); do
    echo "  └─ Fold ${fold}..."

    RESULTS_DIR="${ABLRESULTS_DIR}/gene/fold_${fold}"
    mkdir -p "${RESULTS_DIR}"

    python3 main.py \
        --study tcga_${STUDY} \
        --k_start ${fold} \
        --k_end ${fold} \
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
        --ab_model 2  # 仅基因模式

    echo "  └─ Fold ${fold} 完成"
done

# 汇总 Gene Only 结果
echo ""
echo "📊 汇总 Gene Only 结果..."
GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"
python3 -c "
import pandas as pd
import glob
import os

dfs = []
for f in glob.glob('${ABLRESULTS_DIR}/gene/fold_*/summary.csv'):
    df = pd.read_csv(f)
    fold = int(f.split('/')[-2].split('_')[-1])
    df['fold'] = fold
    dfs.append(df)

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv('${GENE_SUMMARY}', index=False)
    print(f'Gene Only 平均 C-Index: {result[\"val_cindex\"].mean():.4f}')
else:
    print('警告: 找不到 Gene Only 结果文件')
"
echo "  └─ 汇总完成: ${GENE_SUMMARY}"

# ====================================================================
# 2. Text Only 模式 (ab_model=1)
# ====================================================================
echo ""
echo "=============================================="
echo "📝 模式2/3: Text Only (仅文本)"
echo "=============================================="

for fold in $(seq 0 $((K_FOLDS-1))); do
    echo "  └─ Fold ${fold}..."

    RESULTS_DIR="${ABLRESULTS_DIR}/text/fold_${fold}"
    mkdir -p "${RESULTS_DIR}"

    python3 main.py \
        --study tcga_${STUDY} \
        --k_start ${fold} \
        --k_end ${fold} \
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
        --ab_model 1  # 仅文本模式

    echo "  └─ Fold ${fold} 完成"
done

# 汇总 Text Only 结果
echo ""
echo "📊 汇总 Text Only 结果..."
TEXT_SUMMARY="${ABLRESULTS_DIR}/text/summary.csv"
python3 -c "
import pandas as pd
import glob
import os

dfs = []
for f in glob.glob('${ABLRESULTS_DIR}/text/fold_*/summary.csv'):
    df = pd.read_csv(f)
    fold = int(f.split('/')[-2].split('_')[-1])
    df['fold'] = fold
    dfs.append(df)

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv('${TEXT_SUMMARY}', index=False)
    print(f'Text Only 平均 C-Index: {result[\"val_cindex\"].mean():.4f}')
else:
    print('警告: 找不到 Text Only 结果文件')
"
echo "  └─ 汇总完成: ${TEXT_SUMMARY}"

# ====================================================================
# 3. Fusion 模式 (ab_model=3)
# ====================================================================
echo ""
echo "=============================================="
echo "🔗 模式3/3: Fusion (多模态融合)"
echo "=============================================="

for fold in $(seq 0 $((K_FOLDS-1))); do
    echo "  └─ Fold ${fold}..."

    RESULTS_DIR="${ABLRESULTS_DIR}/fusion/fold_${fold}"
    mkdir -p "${RESULTS_DIR}"

    python3 main.py \
        --study tcga_${STUDY} \
        --k_start ${fold} \
        --k_end ${fold} \
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
        --ab_model 3  # 多模态融合模式

    echo "  └─ Fold ${fold} 完成"
done

# 汇总 Fusion 结果
echo ""
echo "📊 汇总 Fusion 结果..."
FUSION_SUMMARY="${ABLRESULTS_DIR}/fusion/summary.csv"
python3 -c "
import pandas as pd
import glob
import os

dfs = []
for f in glob.glob('${ABLRESULTS_DIR}/fusion/fold_*/summary.csv'):
    df = pd.read_csv(f)
    fold = int(f.split('/')[-2].split('_')[-1])
    df['fold'] = fold
    dfs.append(df)

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv('${FUSION_SUMMARY}', index=False)
    print(f'Fusion 平均 C-Index: {result[\"val_cindex\"].mean():.4f}')
else:
    print('警告: 找不到 Fusion 结果文件')
"
echo "  └─ 汇总完成: ${FUSION_SUMMARY}"

# ====================================================================
# 生成最终对比表格
# ====================================================================
echo ""
echo "=============================================="
echo "📈 生成最终对比表格"
echo "=============================================="

FINAL_CSV="${ABLRESULTS_DIR}/final_comparison.csv"

python3 << 'EOF'
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

# 收集各折结果
gene_results = {}
text_results = {}
fusion_results = {}

# Gene Only
for f in glob.glob(f"{gene_dir}/fold_*/summary.csv"):
    df = pd.read_csv(f)
    fold = int(f.split('/')[-2].split('_')[-1])
    gene_results[fold] = df['val_cindex'].values[0]

# Text Only
for f in glob.glob(f"{text_dir}/fold_*/summary.csv"):
    df = pd.read_csv(f)
    fold = int(f.split('/')[-2].split('_')[-1])
    text_results[fold] = df['val_cindex'].values[0]

# Fusion
for f in glob.glob(f"{fusion_dir}/fold_*/summary.csv"):
    df = pd.read_csv(f)
    fold = int(f.split('/')[-2].split('_')[-1])
    fusion_results[fold] = df['val_cindex'].values[0]

# 构建对比表格
comparison_data = []
for fold in sorted(set(gene_results.keys()) | set(text_results.keys()) | set(fusion_results.keys())):
    comparison_data.append({
        'Fold': fold,
        'Gene_C_Index': gene_results.get(fold, np.nan),
        'Text_C_Index': text_results.get(fold, np.nan),
        'Fusion_C_Index': fusion_results.get(fold, np.nan)
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("${FINAL_CSV}", index=False)

# 计算平均值
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
print(f"   • Gene Only (仅基因): {gene_mean:.4f}")
print(f"   • Text Only (仅文本): {text_mean:.4f}")
print(f"   • Fusion (多模态融合): {fusion_mean:.4f}")
print(f"\n📁 结果已保存到: ${FINAL_CSV}")
print("="*60)

# 计算提升百分比
if gene_mean > 0:
    fusion_improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
    print(f"\n📈 Fusion 相对于 Gene Only 的提升: {fusion_improvement:+.2f}%")
if text_mean > 0:
    fusion_vs_text = ((fusion_mean - text_mean) / text_mean) * 100
    print(f"📈 Fusion 相对于 Text Only 的提升: {fusion_vs_text:+.2f}%")

EOF

echo ""
echo "✅ 消融实验完成！"
echo "=============================================="
echo "📁 结果目录: ${ABLRESULTS_DIR}"
echo "📊 对比表格: ${FINAL_CSV}"
echo "=============================================="
