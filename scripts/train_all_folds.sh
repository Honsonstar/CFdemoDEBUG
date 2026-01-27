#!/bin/bash
STUDY=$1
MAX_JOBS=4  # 并行数量

if [ -z "$STUDY" ]; then echo "Usage: bash scripts/train_all_folds.sh <study>"; exit 1; fi

SPLIT_DIR="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_${STUDY}"
RESULTS_DIR="results/nested_cv/${STUDY}"
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/summary*.csv # 清理旧汇总

echo "🚀 [Parallel Training] Study: $STUDY | Max Jobs: $MAX_JOBS"

declare -a PIDS=()

for fold in {0..4}; do
    # 控流
    while [ ${#PIDS[@]} -ge $MAX_JOBS ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
                unset 'PIDS[i]'; PIDS=("${PIDS[@]}")
                break
            fi
        done
        sleep 1
    done

    echo "▶️  Starting Fold $fold..."
    python3 main.py \
        --study tcga_${STUDY} --k_start $fold --k_end $((fold + 1)) \
        --split_dir "${SPLIT_DIR}" --results_dir "$RESULTS_DIR/fold_${fold}" \
        --seed $((42 + fold)) --label_file datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv \
        --task survival --n_classes 4 --modality snn \
        --omics_dir "datasets_csv/raw_rna_data/combine/${STUDY}" \
        --data_root_dir "data/${STUDY}/pt_files" --label_col survival_months \
        --type_of_path combine --max_epochs 20 --lr 0.00005 --opt adam --reg 0.00001 \
        --alpha_surv 0.5 --weighted_sample --batch_size 1 --bag_loss nll_surv \
        --encoding_dim 256 --num_patches 4096 --wsi_projection_dim 256 \
        --encoding_layer_1_dim 8 --encoding_layer_2_dim 16 --encoder_dropout 0.25 \
        > "$RESULTS_DIR/fold_${fold}.log" 2>&1 &
    
    PIDS+=($!)
done

echo "⏳ Waiting for all folds to finish..."
wait
echo "✅ All folds finished."

# --- 最终汇总逻辑 ---
echo ""
echo "📊 FINAL 5-FOLD SUMMARY"
echo "======================="
python3 << PYTHON
import pandas as pd
import glob
import numpy as np
import os

res_dir = "$RESULTS_DIR"
scores = []
print(f"{'Fold':<6} | {'C-Index':<10}")
print("-" * 20)

for fold in range(5):
    # 查找 fold_X 目录下的 summary 文件
    pattern = os.path.join(res_dir, f"fold_{fold}", "summary_partial_*.csv")
    files = glob.glob(pattern)
    if files:
        try:
            df = pd.read_csv(files[0])
            val = df['val_cindex'].iloc[0]
            scores.append(val)
            print(f"{fold:<6} | {val:.4f}")
        except:
            print(f"{fold:<6} | Error")
    else:
        print(f"{fold:<6} | Missing")

print("-" * 20)
if scores:
    print(f"AVG    | {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    pd.DataFrame({'c_index': scores}).to_csv(os.path.join(res_dir, 'final_summary.csv'))
else:
    print("No valid results found.")
PYTHON
