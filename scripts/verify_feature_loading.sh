#!/bin/bash
# ====================================================================
# 快速验证：检查CPCG特征文件是否被正确加载
# 【前台运行，只跑1个batch，立刻退出】
# ====================================================================

echo "🔍 验证CPCG特征文件加载逻辑"
echo "=============================================="
echo ""
echo "📝 任务：运行1个batch，验证是否读取 fold_0_genes.csv"
echo "⚠️  盯着屏幕，如果出现 🔍 [Data Loading] 就成功了！"
echo "=============================================="

cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 设置环境变量加速
export OMP_NUM_THREADS=4

# 直接前台运行，不后台，不写log
# 使用timeout防止卡死（2分钟）
timeout 120 python3 main.py \
    --study tcga_blca \
    --k_start 0 \
    --k_end 1 \
    --split_dir "splits/nested_cv/blca" \
    --results_dir "results/verify_features/fold_0" \
    --seed 42 \
    --label_file "datasets_csv/clinical_data/tcga_blca_clinical.csv" \
    --task survival \
    --n_classes 4 \
    --modality snn \
    --omics_dir "datasets_csv/raw_rna_data/combine/blca" \
    --data_root_dir "data/blca/pt_files" \
    --label_col survival_months \
    --type_of_path combine \
    --max_epochs 1 \
    --lr 0.00005 \
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
    --ab_model 2

# 检查退出状态
EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试完成!"
    echo ""
    echo "📝 请在上方日志中搜索以下关键信息："
    echo "   🔍 [Data Loading] Loading gene features from:"
    echo ""
    echo "预期看到："
    echo "   🔍 [Data Loading] Loading gene features from: features/blca/fold_0_genes.csv"
    echo ""
    echo "如果看到了这行，说明CPCG特征文件被正确加载了！"
else
    echo "❌ 测试失败 (退出码: $EXIT_CODE)"
    echo ""
    echo "请将完整的错误信息发送给我。"
fi
echo "=============================================="
