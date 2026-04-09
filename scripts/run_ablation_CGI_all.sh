#!/bin/bash

# ============================================================
# 简化消融实验脚本（CGI版）
# 运行模式：Text Only（ab_model=1）、Gene Only（ab_model=2）、Fusion（ab_model=3）
# ============================================================

set -u

SCRIPT_START_TS=$(date +%s)

# -------------------- 癌种参数 --------------------
ALL_CANCERS=("brca" "blca" "hnsc" "stad" "coadread")
if [ $# -eq 0 ]; then
    echo "未指定癌种，将运行全部癌种：${ALL_CANCERS[*]}"
    CANCERS_TO_RUN=("${ALL_CANCERS[@]}")
else
    CANCERS_TO_RUN=("$@")
fi

# -------------------- 训练配置区 --------------------
SEED=42
K_FOLDS=5
MAX_EPOCHS=20
LR=0.00005
REG=0.0001
ENCODER_DROPOUT=0.35
MAX_JOBS=3
GENE_TOPK=100

# -------------------- 早停配置区 --------------------
EARLY_STOP_ENABLE=0
EARLY_STOP_MONITOR="val_cindex_ipcw"
EARLY_STOP_MODE="max"
EARLY_STOP_PATIENCE=5
EARLY_STOP_MIN_DELTA=0.001

# -------------------- 两阶段训练配置区 --------------------
TWO_STAGE_ENABLE=0
FREEZE_TEXT_EPOCHS=8
TWO_STAGE_FUSION_ONLY=1

# -------------------- 公共变量 --------------------
TODAY=$(date +%Y-%m-%d)

supports_early_stop() {
    local args_file="utils/process_args.py"
    if [ ! -f "$args_file" ]; then
        return 1
    fi
    if grep -q "add_argument('--early_stop" "$args_file"; then
        return 0
    fi
    return 1
}

supports_two_stage() {
    local args_file="utils/process_args.py"
    if [ ! -f "$args_file" ]; then
        return 1
    fi
    if grep -q "add_argument('--two_stage_train" "$args_file"; then
        return 0
    fi
    return 1
}

print_run_config() {
    local study="$1"
    local feature_dir="$2"
    local sample_feature_file="${feature_dir}/fold_0_genes.csv"
    local feature_info="未知"

    if [ -f "$sample_feature_file" ]; then
        feature_info=$(python3 - << 'PY'
import csv
import os
fp = os.environ.get("SAMPLE_FEATURE_FILE", "")
try:
    with open(fp, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    sample_cols = max(len(header) - 1, 0)
    print(f"样本列数={sample_cols}")
except Exception:
    print("未知")
PY
)
    fi

    echo "=============================================================="
    echo "当前癌种：${study^^}"
    echo "随机种子：${SEED}"
    echo "折数：${K_FOLDS}"
    echo "最大训练轮次：${MAX_EPOCHS}"
    echo "学习率（lr）：${LR}"
    echo "权重衰减（reg）：${REG}"
    echo "Dropout（encoder_dropout）：${ENCODER_DROPOUT}"
    echo "目标基因数（GENE_TOPK）：${GENE_TOPK}"
    echo "特征目录：${feature_dir}"
    echo "特征规模检查：${feature_info}"
    echo "早停开关：${EARLY_STOP_ENABLE}"
    echo "早停监控指标：${EARLY_STOP_MONITOR}"
    echo "早停模式：${EARLY_STOP_MODE}"
    echo "早停耐心轮次：${EARLY_STOP_PATIENCE}"
    echo "早停最小提升：${EARLY_STOP_MIN_DELTA}"
    echo "两阶段训练开关：${TWO_STAGE_ENABLE}"
    echo "第1阶段冻结epoch：${FREEZE_TEXT_EPOCHS}"
    echo "仅Fusion启用两阶段：${TWO_STAGE_FUSION_ONLY}"
    echo "=============================================================="
}

check_required_paths() {
    local split_dir="$1"
    local feature_dir="$2"
    local study="$3"

    if [ ! -d "$split_dir" ]; then
        echo "错误：找不到划分目录：$split_dir"
        echo "请先执行：python3 preprocessing/CGI/preprocess_test.py"
        return 1
    fi

    local missing=0
    echo "检查 ${study^^} 的稳定基因特征文件..."
    for fold in $(seq 0 $((K_FOLDS - 1))); do
        local f="${feature_dir}/fold_${fold}_genes.csv"
        if [ -f "$f" ]; then
            echo "  Fold ${fold}：存在（$(basename "$f")）"
        else
            echo "  Fold ${fold}：缺失（$(basename "$f")）"
            missing=1
        fi
    done
    if [ "$missing" -ne 0 ]; then
        echo "错误：稳定特征文件不完整，终止运行。"
        return 1
    fi
    return 0
}

build_early_stop_args() {
    EXTRA_ARGS=()
    if [ "$EARLY_STOP_ENABLE" -eq 1 ]; then
        if supports_early_stop; then
            EXTRA_ARGS+=("--early_stop")
            EXTRA_ARGS+=("--early_stop_monitor" "$EARLY_STOP_MONITOR")
            EXTRA_ARGS+=("--early_stop_mode" "$EARLY_STOP_MODE")
            EXTRA_ARGS+=("--early_stop_patience" "$EARLY_STOP_PATIENCE")
            EXTRA_ARGS+=("--early_stop_min_delta" "$EARLY_STOP_MIN_DELTA")
            echo "已启用早停：训练入口支持早停参数。"
        else
            echo "警告：当前训练入口未检测到早停参数定义，早停配置仅打印，不会生效。"
        fi
    else
        echo "早停已关闭。"
    fi
}

build_two_stage_args() {
    if [ "$TWO_STAGE_ENABLE" -eq 1 ]; then
        if supports_two_stage; then
            EXTRA_ARGS+=("--two_stage_train")
            EXTRA_ARGS+=("--freeze_text_epochs" "$FREEZE_TEXT_EPOCHS")
            if [ "$TWO_STAGE_FUSION_ONLY" -eq 1 ]; then
                EXTRA_ARGS+=("--two_stage_fusion_only")
            fi
            echo "已启用两阶段训练：冻结文本${FREEZE_TEXT_EPOCHS}个epoch，随后解冻联合训练。"
        else
            echo "警告：当前训练入口未检测到两阶段参数定义，两阶段配置仅打印，不会生效。"
        fi
    else
        echo "两阶段训练已关闭。"
    fi
}

run_mode() {
    local study="$1"
    local mode_name="$2"
    local ab_model="$3"
    local results_subdir="$4"
    local label_file="$5"
    local split_dir="$6"
    local omics_dir="$7"
    local feature_dir="$8"
    local data_root_dir="$9"
    local abresults_dir="${10}"

    local mode_dir="${abresults_dir}/${results_subdir}"
    local mode_log="${abresults_dir}/${results_subdir}_training.log"
    mkdir -p "$mode_dir"

    echo "" | tee -a "$mode_log"
    echo "==============================================================" | tee -a "$mode_log"
    echo "开始训练模式：${mode_name}" | tee -a "$mode_log"
    echo "输出目录：${mode_dir}" | tee -a "$mode_log"
    echo "==============================================================" | tee -a "$mode_log"

    local failed_folds=()
    local launched_folds=()

    if [ "$MAX_JOBS" -lt 1 ]; then
        echo "警告：MAX_JOBS=${MAX_JOBS} 非法，自动修正为 1" | tee -a "$mode_log"
        MAX_JOBS=1
    fi

    for fold in $(seq 0 $((K_FOLDS - 1))); do
        local fold_dir="${mode_dir}/fold_${fold}"
        local fold_log="${fold_dir}/training.log"
        local fold_exit="${fold_dir}/exit_code.txt"
        mkdir -p "$fold_dir"

        rm -f "$fold_exit"

        echo "  └─ 启动 Fold ${fold}..." | tee -a "$mode_log"
        launched_folds+=("$fold")

        (
            python -u main.py \
                --study "tcga_${study}" \
                --k_start "$fold" \
                --k_end "$((fold + 1))" \
                --split_dir "$split_dir" \
                --results_dir "$fold_dir" \
                --seed "$SEED" \
                --label_file "$label_file" \
                --task survival \
                --n_classes 4 \
                --modality snn \
                --omics_dir "$omics_dir" \
                --fold_feature_dir "$feature_dir" \
                --data_root_dir "$data_root_dir" \
                --label_col survival_months \
                --type_of_path combine \
                --max_epochs "$MAX_EPOCHS" \
                --lr "$LR" \
                --opt adamW \
                --reg "$REG" \
                --alpha_surv 0.5 \
                --weighted_sample \
                --batch_size 1 \
                --bag_loss nll_surv \
                --encoding_dim 256 \
                --num_patches 4096 \
                --wsi_projection_dim 256 \
                --encoding_layer_1_dim 8 \
                --encoding_layer_2_dim 16 \
                --encoder_dropout "$ENCODER_DROPOUT" \
                --ab_model "$ab_model" \
                "${EXTRA_ARGS[@]}" \
                > "$fold_log" 2>&1
            echo $? > "$fold_exit"
        ) &

        while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do
            echo "  └─ 达到最大并发数 ${MAX_JOBS}，等待..." | tee -a "$mode_log"
            wait -n
        done
    done

    wait
    echo "  └─ ${mode_name} 所有 Fold 完成，开始校验结果..." | tee -a "$mode_log"

    for fold in "${launched_folds[@]}"; do
        local fold_dir="${mode_dir}/fold_${fold}"
        local fold_log="${fold_dir}/training.log"
        local fold_exit="${fold_dir}/exit_code.txt"
        local code=999

        if [ -f "$fold_exit" ]; then
            code=$(cat "$fold_exit" | tr -d ' ')
        fi

        if [ "$code" -ne 0 ]; then
            echo "错误：Fold ${fold} 训练失败，退出码=${code}，日志：${fold_log}" | tee -a "$mode_log"
            failed_folds+=("$fold")
            continue
        fi

        local partial_count
        local summary_count
        partial_count=$(find "$fold_dir" -type f -name "summary_partial_*.csv" | wc -l | tr -d ' ')
        summary_count=$(find "$fold_dir" -type f -name "summary.csv" | wc -l | tr -d ' ')
        if [ "$partial_count" -eq 0 ] && [ "$summary_count" -eq 0 ]; then
            echo "错误：Fold ${fold} 未生成 summary_partial_*.csv 或 summary.csv，判定为失败。" | tee -a "$mode_log"
            failed_folds+=("$fold")
            continue
        fi

        echo "Fold ${fold} 完成。" | tee -a "$mode_log"
    done

    if [ "${#failed_folds[@]}" -gt 0 ]; then
        echo "模式 ${mode_name} 存在失败折：${failed_folds[*]}" | tee -a "$mode_log"
    else
        echo "模式 ${mode_name} 全部 Fold 训练成功。" | tee -a "$mode_log"
    fi
}

merge_mode_summary() {
    local mode_dir="$1"
    local out_csv="$2"
    local mode_name="$3"

    echo ""
    echo "📊 汇总 ${mode_name} 结果..."

    python3 - << 'PY'
import glob
import os
import pandas as pd

mode_dir = os.environ["MODE_DIR"]
out_csv = os.environ["OUT_CSV"]
mode_name = os.environ["MODE_NAME"]
k_folds = int(os.environ["K_FOLDS"])

rows = []
for fold in range(k_folds):
    fold_dir = os.path.join(mode_dir, f"fold_{fold}")
    if not os.path.isdir(fold_dir):
        print(f"{mode_name} Fold {fold}: 目录不存在，跳过")
        continue

    partials = sorted(glob.glob(os.path.join(fold_dir, "**", "summary_partial_*.csv"), recursive=True))
    if partials:
        f = partials[-1]
        from_summary = False
    else:
        summaries = sorted(glob.glob(os.path.join(fold_dir, "**", "summary.csv"), recursive=True))
        if not summaries:
            print(f"{mode_name} Fold {fold}: 未找到 summary_partial_*.csv 或 summary.csv，跳过")
            continue
        f = summaries[-1]
        from_summary = True

    try:
        df = pd.read_csv(f)
        if "val_cindex" not in df.columns:
            print(f"{mode_name} Fold {fold}: 文件缺少 val_cindex 列，跳过")
            continue
        r = df.iloc[-1].copy()
        r["fold"] = fold
        rows.append(r)
        if from_summary:
            print(f"  ✓ Fold {fold}: {os.path.basename(f)} (from summary.csv)")
        else:
            print(f"  ✓ Fold {fold}: {os.path.basename(f)}")
    except Exception as e:
        print(f"{mode_name} Fold {fold}: 读取失败 {e}")

if not rows:
    print(f"❌ {mode_name}: 没有可汇总数据")
    pd.DataFrame().to_csv(out_csv, index=False)
else:
    out_df = pd.DataFrame(rows)
    ordered = ["fold"] + [c for c in out_df.columns if c != "fold"]
    out_df = out_df[ordered].sort_values("fold")
    out_df.to_csv(out_csv, index=False)
    mean_ci = out_df["val_cindex"].mean() if "val_cindex" in out_df.columns else float("nan")
    print(f"✅ {mode_name} 汇总: {len(out_df)}/{k_folds} 折成功, 平均 C-Index: {mean_ci:.4f}")
PY
}

make_final_report() {
    local ab_dir="$1"
    local final_csv="$2"
    local report_csv="$3"

    python3 - << 'PY'
import os
import pandas as pd

ab_dir = os.environ["AB_DIR"]
final_csv = os.environ["FINAL_CSV"]
report_csv = os.environ["REPORT_CSV"]

text_csv = os.path.join(ab_dir, "text", "summary.csv")
gene_csv = os.path.join(ab_dir, "gene", "summary.csv")
fusion_csv = os.path.join(ab_dir, "fusion", "summary.csv")

text_df = pd.read_csv(text_csv) if os.path.exists(text_csv) else pd.DataFrame()
gene_df = pd.read_csv(gene_csv) if os.path.exists(gene_csv) else pd.DataFrame()
fusion_df = pd.read_csv(fusion_csv) if os.path.exists(fusion_csv) else pd.DataFrame()

folds = sorted(set(text_df.get("fold", pd.Series(dtype=int)).tolist()
                   + gene_df.get("fold", pd.Series(dtype=int)).tolist()
                   + fusion_df.get("fold", pd.Series(dtype=int)).tolist()))

rows = []
for f in folds:
    row = {"fold": f}
    row["text_val_cindex"] = float(text_df.loc[text_df["fold"] == f, "val_cindex"].iloc[0]) if (not text_df.empty and (text_df["fold"] == f).any()) else float("nan")
    row["gene_val_cindex"] = float(gene_df.loc[gene_df["fold"] == f, "val_cindex"].iloc[0]) if (not gene_df.empty and (gene_df["fold"] == f).any()) else float("nan")
    row["fusion_val_cindex"] = float(fusion_df.loc[fusion_df["fold"] == f, "val_cindex"].iloc[0]) if (not fusion_df.empty and (fusion_df["fold"] == f).any()) else float("nan")
    if pd.notna(row["text_val_cindex"]) and pd.notna(row["fusion_val_cindex"]):
        row["fusion_minus_text"] = row["fusion_val_cindex"] - row["text_val_cindex"]
    if pd.notna(row["gene_val_cindex"]) and pd.notna(row["fusion_val_cindex"]):
        row["fusion_minus_gene"] = row["fusion_val_cindex"] - row["gene_val_cindex"]
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(final_csv, index=False)

text_mean = df["text_val_cindex"].mean() if "text_val_cindex" in df.columns else float("nan")
gene_mean = df["gene_val_cindex"].mean() if "gene_val_cindex" in df.columns else float("nan")
fusion_mean = df["fusion_val_cindex"].mean() if "fusion_val_cindex" in df.columns else float("nan")

report = pd.DataFrame([{
    "text_mean_cindex": text_mean,
    "gene_mean_cindex": gene_mean,
    "fusion_mean_cindex": fusion_mean,
    "fusion_minus_text_mean": (fusion_mean - text_mean) if (pd.notna(fusion_mean) and pd.notna(text_mean)) else float("nan"),
    "fusion_minus_gene_mean": (fusion_mean - gene_mean) if (pd.notna(fusion_mean) and pd.notna(gene_mean)) else float("nan"),
}])
report.to_csv(report_csv, index=False)

print("\n==================================================")
print("简化消融结果汇总")
print("==================================================")
if not df.empty:
    print(df.to_string(index=False))
else:
    print("无可用折结果")
print("==================================================")
print(f"Text Only 平均 C-Index: {text_mean:.4f}" if pd.notna(text_mean) else "Text Only 平均 C-Index: nan")
print(f"Gene Only 平均 C-Index: {gene_mean:.4f}" if pd.notna(gene_mean) else "Gene Only 平均 C-Index: nan")
print(f"Fusion    平均 C-Index: {fusion_mean:.4f}" if pd.notna(fusion_mean) else "Fusion    平均 C-Index: nan")
print(f"最终表格: {final_csv}")
print(f"报告文件: {report_csv}")
PY
}

for STUDY in "${CANCERS_TO_RUN[@]}"; do
    STUDY_START_TS=$(date +%s)
    echo ""
    echo "################################################################"
    echo "开始处理癌种：${STUDY^^}"
    echo "################################################################"

    LABEL_FILE="datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv"
    SPLIT_DIR="splits/nested_cv/${STUDY}"
    FEATURE_DIR="preprocessing/CGI_py/features/stable/${STUDY}"
    OMICS_DIR="datasets_csv/raw_rna_data/combine/${STUDY}"
    DATA_ROOT_DIR="data/${STUDY}/pt_files"
    ABLRESULTS_DIR="results/ablation/${STUDY}"
    LOG_DIR="log/${TODAY}/${STUDY}"
    REPORT_DIR="report"
    MAIN_LOG="${LOG_DIR}/ablation_simple.log"

    mkdir -p "$LOG_DIR" "$REPORT_DIR" "$ABLRESULTS_DIR/text" "$ABLRESULTS_DIR/gene" "$ABLRESULTS_DIR/fusion"

    export SAMPLE_FEATURE_FILE="${FEATURE_DIR}/fold_0_genes.csv"
    print_run_config "$STUDY" "$FEATURE_DIR" | tee -a "$MAIN_LOG"

    if ! check_required_paths "$SPLIT_DIR" "$FEATURE_DIR" "$STUDY"; then
        exit 1
    fi

    build_early_stop_args
    build_two_stage_args
    echo "附加参数：${EXTRA_ARGS[*]:-无}" | tee -a "$MAIN_LOG"

    run_mode "$STUDY" "Text Only（仅文本）" 1 "text" \
        "$LABEL_FILE" "$SPLIT_DIR" "$OMICS_DIR" "$FEATURE_DIR" "$DATA_ROOT_DIR" "$ABLRESULTS_DIR"
    export MODE_DIR="${ABLRESULTS_DIR}/text"
    export OUT_CSV="${ABLRESULTS_DIR}/text/summary.csv"
    export MODE_NAME="Text Only"
    export K_FOLDS
    merge_mode_summary "$MODE_DIR" "$OUT_CSV" "$MODE_NAME" | tee -a "$MAIN_LOG"

    run_mode "$STUDY" "Gene Only（仅基因）" 2 "gene" \
        "$LABEL_FILE" "$SPLIT_DIR" "$OMICS_DIR" "$FEATURE_DIR" "$DATA_ROOT_DIR" "$ABLRESULTS_DIR"
    export MODE_DIR="${ABLRESULTS_DIR}/gene"
    export OUT_CSV="${ABLRESULTS_DIR}/gene/summary.csv"
    export MODE_NAME="Gene Only"
    merge_mode_summary "$MODE_DIR" "$OUT_CSV" "$MODE_NAME" | tee -a "$MAIN_LOG"

    run_mode "$STUDY" "Fusion（基因+文本）" 3 "fusion" \
        "$LABEL_FILE" "$SPLIT_DIR" "$OMICS_DIR" "$FEATURE_DIR" "$DATA_ROOT_DIR" "$ABLRESULTS_DIR"
    export MODE_DIR="${ABLRESULTS_DIR}/fusion"
    export OUT_CSV="${ABLRESULTS_DIR}/fusion/summary.csv"
    export MODE_NAME="Fusion"
    merge_mode_summary "$MODE_DIR" "$OUT_CSV" "$MODE_NAME" | tee -a "$MAIN_LOG"

    export AB_DIR="$ABLRESULTS_DIR"
    export FINAL_CSV="${ABLRESULTS_DIR}/final_comparison.csv"
    export REPORT_CSV="${REPORT_DIR}/${TODAY}_${STUDY}_ablation_simple.csv"
    make_final_report "$AB_DIR" "$FINAL_CSV" "$REPORT_CSV" | tee -a "$MAIN_LOG"

    STUDY_END_TS=$(date +%s)
    STUDY_TOTAL_SECONDS=$((STUDY_END_TS - STUDY_START_TS))
    STUDY_HOURS=$((STUDY_TOTAL_SECONDS / 3600))
    STUDY_MINUTES=$(((STUDY_TOTAL_SECONDS % 3600) / 60))
    STUDY_REMAIN_SECONDS=$((STUDY_TOTAL_SECONDS % 60))

    echo "癌种 ${STUDY^^} 运行完成。"
    echo "结果目录：${ABLRESULTS_DIR}"
    echo "比较表：${FINAL_CSV}"
    echo "报告：${REPORT_CSV}"
    printf "癌种 %s 耗时：%02d:%02d:%02d\n" "${STUDY^^}" "$STUDY_HOURS" "$STUDY_MINUTES" "$STUDY_REMAIN_SECONDS"
done

echo ""
echo "################################################################"
echo "全部癌种运行完成。"
echo "结果总目录：results/ablation/"
echo "报告目录：report/"
SCRIPT_END_TS=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TS - SCRIPT_START_TS))
TOTAL_HOURS=$((TOTAL_SECONDS / 3600))
TOTAL_MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
TOTAL_REMAIN_SECONDS=$((TOTAL_SECONDS % 60))
printf "总耗时：%02d:%02d:%02d\n" "$TOTAL_HOURS" "$TOTAL_MINUTES" "$TOTAL_REMAIN_SECONDS"
echo "################################################################"
