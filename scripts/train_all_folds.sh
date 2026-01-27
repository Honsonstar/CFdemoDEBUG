#!/bin/bash
# 训练所有折的脚本 (并行版本)

STUDY=$1

# ============================================================
# 【新增】GPU性能优化配置
# ============================================================
# 最大并发任务数 - 根据GPU显存调整
# 推荐值: 3-4 (基于22% GPU利用率观察)
# 如果遇到OOM，尝试减少到2或1
MAX_JOBS=4

# GPU显存安全阈值 (可选配置)
# GPU_MEMORY_FRACTION=0.8  # 使用80%的GPU显存

if [ -z "$STUDY" ]; then
    echo "=========================================="
    echo "用法: bash train_all_folds.sh <study>"
    echo ""
    echo "示例:"
    echo "  bash train_all_folds.sh blca"
    echo ""
    echo "配置:"
    echo "  最大并发任务数: MAX_JOBS=$MAX_JOBS"
    echo "  调整方法: MAX_JOBS=3 bash train_all_folds.sh blca"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "训练所有折 (并行版本 - 嵌套CV)"
echo "=========================================="
echo "   癌种: $STUDY"
echo "   最大并发任务数: $MAX_JOBS"
echo "=========================================="

# ============================================================
# 【修改】使用外部提供的5折划分
# ============================================================
SPLIT_DIR="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_${STUDY}"

# 检查必要文件
echo "\n🔍 检查必要文件..."

MISSING_FILES=0

# 检查外部划分文件
for fold in {0..4}; do
    if [ ! -f "${SPLIT_DIR}/splits_${fold}.csv" ]; then
        echo "❌ 缺少: ${SPLIT_DIR}/splits_${fold}.csv"
        MISSING_FILES=1
    fi
done

# 检查CPCG特征文件
for fold in {0..4}; do
    if [ ! -f "features/${STUDY}/fold_${fold}_genes.csv" ]; then
        echo "❌ 缺少: features/${STUDY}/fold_${fold}_genes.csv"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo "\n⚠️  缺少必要文件，请先运行:"
    echo "   # 确保外部划分文件位于:"
    echo "   #   ${SPLIT_DIR}/"
    echo "   # 然后运行CPCG筛选:"
    echo "   bash run_all_cpog.sh $STUDY"
    exit 1
fi

echo "✅ 所有必要文件检查通过"

# 创建结果目录
RESULTS_DIR="results/nested_cv/${STUDY}"
mkdir -p "$RESULTS_DIR"

echo "\n📁 结果目录: $RESULTS_DIR"

# 训练所有折 (并行版本)
echo "\n🚀 开始并行训练所有折..."
echo "=========================================="
echo "   最大并发任务数: $MAX_JOBS"
echo "   日志模式: 文件写入 (安静模式)"
echo "=========================================="

# 并行训练变量
declare -a JOB_PIDS=()
declare -a JOB_START_TIMES=()
declare -a JOB_FOLDS=()
job_count=0

# 并行启动所有训练任务
for fold in {0..4}; do
    # 等待直到有可用的GPU槽位
    while [ ${#JOB_PIDS[@]} -ge $MAX_JOBS ]; do
        # 检查是否有已完成的任务
        for i in "${!JOB_PIDS[@]}"; do
            pid=${JOB_PIDS[i]}
            if ! kill -0 "$pid" 2>/dev/null; then
                # 任务已完成
                wait "$pid"
                end_time=$(date +%s)
                duration=$((end_time - JOB_START_TIMES[i]))
                echo ""
                echo "✅ Fold ${JOB_FOLDS[$i]} 完成 (PID: $pid, 耗时: ${duration}s)"

                # 从数组中移除
                unset 'JOB_PIDS[i]'
                unset 'JOB_START_TIMES[i]'
                unset 'JOB_FOLDS[i]'

                # 重新索引数组
                JOB_PIDS=("${JOB_PIDS[@]}")
                JOB_START_TIMES=("${JOB_START_TIMES[@]}")
                JOB_FOLDS=("${JOB_FOLDS[@]}")
                break
            fi
        done

        # 如果仍在满负载状态，等待1秒后重试
        if [ ${#JOB_PIDS[@]} -ge $MAX_JOBS ]; then
            sleep 1
        fi
    done

    # 启动新任务
    echo ""
    echo "🚀 启动 Fold $fold (当前并发: ${#JOB_PIDS[@]}/$MAX_JOBS)"

    start_time=$(date +%s)

    # 静默模式：日志只写入文件，不输出到终端
    python3 main.py \
        --study tcga_${STUDY} \
        --k_start $fold \
        --k_end $((fold + 1)) \
        --split_dir "${SPLIT_DIR}" \
        --results_dir "$RESULTS_DIR/fold_${fold}" \
        --seed $((42 + fold)) \
        --label_file datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv \
        --task survival \
        --n_classes 4 \
        --modality snn \
        --omics_dir "datasets_csv/raw_rna_data/combine/${STUDY}" \
        --data_root_dir "data/${STUDY}/pt_files" \
        --label_col survival_months \
        --type_of_path combine \
        --max_epochs 20 \
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
        >> "$RESULTS_DIR/fold_${fold}.log" 2>&1 &

    pid=$!
    JOB_PIDS+=($pid)
    JOB_START_TIMES+=($start_time)
    JOB_FOLDS+=($fold)
done

# 等待所有后台任务完成
echo ""
echo "⏳ 等待所有训练任务完成..."
wait

echo ""
echo "✅ 所有训练任务已完成!"

# 汇总结果
echo ""
echo "=================================================="
echo "📊 汇总所有折的结果"
echo "=================================================="

python3 << PYTHON
import pandas as pd
import numpy as np
import os
import glob

study = '$STUDY'
results_dir = 'results/nested_cv/${STUDY}'

print(f"\n癌种: {study}")
print(f"结果目录: {results_dir}")

# 读取所有折的结果
cindex_scores = []
cindex_ipcw = []
bs_scores = []
ibs_scores = []
iauc_scores = []

for fold in range(5):
    summary_file = f'{results_dir}/fold_{fold}/summary.csv'
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file)
        cindex_scores.append(df['val_cindex'].iloc[0])
        if 'val_cindex_ipcw' in df.columns:
            cindex_ipcw.append(df['val_cindex_ipcw'].iloc[0])
        if 'val_BS' in df.columns:
            bs_scores.append(df['val_BS'].iloc[0])
        if 'val_IBS' in df.columns:
            ibs_scores.append(df['val_IBS'].iloc[0])
        if 'val_iauc' in df.columns:
            iauc_scores.append(df['val_iauc'].iloc[0])
        print(f"Fold {fold}: C-index = {df['val_cindex'].iloc[0]:.4f}")
    else:
        print(f"⚠️  Fold {fold}: 结果文件不存在")

if cindex_scores:
    # 计算统计量
    mean_cindex = np.mean(cindex_scores)
    std_cindex = np.std(cindex_scores)
    
    print(f"\n{'='*50}")
    print("最终结果 (嵌套CV)")
    print(f"{'='*50}")
    print(f"C-index:     {mean_cindex:.4f} ± {std_cindex:.4f}")
    
    if cindex_ipcw:
        mean_ipcw = np.mean(cindex_ipcw)
        std_ipcw = np.std(cindex_ipcw)
        print(f"C-index IPCW: {mean_ipcw:.4f} ± {std_ipcw:.4f}")
    
    if bs_scores:
        mean_bs = np.mean(bs_scores)
        std_bs = np.std(bs_scores)
        print(f"Brier Score: {mean_bs:.4f} ± {std_bs:.4f}")
    
    if ibs_scores:
        mean_ibs = np.mean(ibs_scores)
        std_ibs = np.std(ibs_scores)
        print(f"IBS:         {mean_ibs:.4f} ± {std_ibs:.4f}")
    
    if iauc_scores:
        mean_iauc = np.mean(iauc_scores)
        std_iauc = np.std(iauc_scores)
        print(f"IAUC:        {mean_iauc:.4f} ± {std_iauc:.4f}")
    
    # 保存汇总
    summary = {
        'fold': list(range(5)),
        'val_cindex': cindex_scores
    }
    
    if cindex_ipcw:
        summary['val_cindex_ipcw'] = cindex_ipcw
    if bs_scores:
        summary['val_BS'] = bs_scores
    if ibs_scores:
        summary['val_IBS'] = ibs_scores
    if iauc_scores:
        summary['val_iauc'] = iauc_scores
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{results_dir}/summary.csv', index=False)
    
    print(f"\n✅ 汇总结果保存到: {results_dir}/summary.csv")
else:
    print("\n❌ 没有找到任何结果文件!")

PYTHON

echo ""
echo "=========================================="
echo "✅ 所有折训练完成!"
echo "=========================================="
echo "   结果目录: $RESULTS_DIR"
echo "   汇总文件: $RESULTS_DIR/summary.csv"
echo ""
echo "📊 查看结果:"
echo "   cat $RESULTS_DIR/summary.csv"
echo "=========================================="
