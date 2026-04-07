#!/bin/bash
#
# 文件说明：
#   该脚本用于批量扫描 Fusion 模式下的正则项（REG）和编码器 Dropout（ENCODER_DROPOUT）组合。
#   逻辑参考 run_ablation_simple.sh，但这里只保留 Fusion（ab_model=3）训练与汇总流程，
#   不运行 Gene Only 模式。
#
# 使用方式：
#   1. 不传癌种参数时，默认依次运行 5 个癌种：
#        bash scripts/run_fusion_reg_dropout_sweep.sh
#   2. 传入一个或多个癌种时，只运行指定癌种：
#        bash scripts/run_fusion_reg_dropout_sweep.sh hnsc
#        bash scripts/run_fusion_reg_dropout_sweep.sh brca blca
#
# 输出说明：
#   1. 训练结果仍复用项目默认目录：results/ablation/<study>/fusion
#   2. 参数扫描摘要写入：log/<日期>/<study>_sweeps/results.txt
#   3. results.txt 仅记录参数组合、固定记录参数和 Fusion 平均 C-Index

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

TODAY="$(date +%Y-%m-%d)"
SCRIPT_START_TS=$(date +%s)

# -------------------------------------------------------------------
# 扫描对象配置
#   传参时使用命令行中的癌种列表；不传参时默认扫描 5 个癌种。
# -------------------------------------------------------------------

if [ $# -ge 1 ]; then
    STUDIES=("$@")
else
    STUDIES=("brca" "blca" "hnsc" "stad" "coadread")
fi

# -------------------------------------------------------------------
# 全局目录配置
#   尽量将目录前缀统一放在前面，便于后续迁移或集中调整。
# -------------------------------------------------------------------
CLINICAL_ROOT="datasets_csv/clinical_data"
SPLIT_ROOT="splits/CGI_nested_cv"
FEATURE_ROOT="preprocessing/CGI_py/features/stable"
OMICS_ROOT="datasets_csv/raw_rna_data/combine"
PT_ROOT="data"
ABLATION_ROOT="results/ablation"
LOG_ROOT="log"
REPORT_ROOT="report"

# -------------------------------------------------------------------
# 训练基础参数
#   这一组参数在本脚本中对所有癌种、所有组合保持一致。
# -------------------------------------------------------------------
SEED=42
K_FOLDS=5
MAX_EPOCHS=20
LR=0.00005
MAX_JOBS=2
GENE_TOPK=100

# -------------------------------------------------------------------
# 可选训练策略开关
#   默认关闭；保留这些配置是为了和主训练脚本行为保持兼容。
# -------------------------------------------------------------------
EARLY_STOP_ENABLE=0
EARLY_STOP_MONITOR="val_cindex_ipcw"
EARLY_STOP_MODE="max"
EARLY_STOP_PATIENCE=5
EARLY_STOP_MIN_DELTA=0.001

TWO_STAGE_ENABLE=0
FREEZE_TEXT_EPOCHS=8
TWO_STAGE_FUSION_ONLY=1

# -------------------------------------------------------------------
# 固定记录参数
#   当前仅用于结果记录与实验说明，不作为实际命令行参数传入训练。
# -------------------------------------------------------------------
MARKOV_BLANKET="0"
TOP_K="50"

# -------------------------------------------------------------------
# 参数扫描区
#   通过遍历 REG_VALUES 和 DROPOUT_VALUES 形成网格搜索。
# -------------------------------------------------------------------
REG_VALUES=(
    "0.00001"
    "0.00005"
    "0.0001"
    "0.0005"
)

DROPOUT_VALUES=(
    "0.25"
    "0.30"
    "0.35"
)

export K_FOLDS

build_study_paths() {
    local study="$1"

    LABEL_FILE="${CLINICAL_ROOT}/tcga_${study}_clinical.csv"
    SPLIT_DIR="${SPLIT_ROOT}/${study}"
    FEATURE_DIR="${FEATURE_ROOT}/${study}"
    OMICS_DIR="${OMICS_ROOT}/${study}"
    DATA_ROOT_DIR="${PT_ROOT}/${study}/pt_files"
    ABLRESULTS_DIR="${ABLATION_ROOT}/${study}"
    LOG_DIR="${LOG_ROOT}/${TODAY}/${study}"
    RESULTS_TXT="${LOG_ROOT}/${TODAY}/${study}_sweeps/results.txt"
    MAIN_LOG="${LOG_DIR}/fusion_only_sweep.log"
}

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

print_run_config() {
    local study="$1"
    local feature_dir="$2"
    local sample_feature_file="${feature_dir}/fold_0_genes.csv"
    local gene_dim="未知"

    if [ -f "$sample_feature_file" ]; then
        gene_dim=$(python - << 'PY'
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
    echo "特征规模检查：${gene_dim}"
    echo "固定参数：Markov blanket=${MARKOV_BLANKET}, TOP_K=${TOP_K}"
    echo "=============================================================="
}

check_required_paths() {
    local split_dir="$1"
    local feature_dir="$2"
    local study="$3"

    if [ ! -d "$split_dir" ]; then
        echo "错误：找不到划分目录：$split_dir"
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
        fi
    fi
}

build_two_stage_args() {
    if [ "$TWO_STAGE_ENABLE" -eq 1 ]; then
        EXTRA_ARGS+=("--two_stage_train")
        EXTRA_ARGS+=("--freeze_text_epochs" "$FREEZE_TEXT_EPOCHS")
        if [ "$TWO_STAGE_FUSION_ONLY" -eq 1 ]; then
            EXTRA_ARGS+=("--two_stage_fusion_only")
        fi
    fi
}

run_fusion_mode() {
    local study="$1"
    local label_file="$2"
    local split_dir="$3"
    local omics_dir="$4"
    local feature_dir="$5"
    local data_root_dir="$6"
    local abresults_dir="$7"

    local mode_dir="${abresults_dir}/fusion"
    local mode_log="${abresults_dir}/fusion_training.log"
    mkdir -p "$mode_dir"

    echo "" | tee -a "$mode_log"
    echo "==============================================================" | tee -a "$mode_log"
    echo "开始训练模式：Fusion（基因+文本）" | tee -a "$mode_log"
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
                --ab_model 3 \
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
    echo "  └─ Fusion 所有 Fold 完成，开始校验结果..." | tee -a "$mode_log"

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
        echo "模式 Fusion 存在失败折：${failed_folds[*]}" | tee -a "$mode_log"
    else
        echo "模式 Fusion 全部 Fold 训练成功。" | tee -a "$mode_log"
    fi
}

merge_mode_summary() {
    local mode_dir="$1"
    local out_csv="$2"
    local mode_name="$3"

    echo ""
    echo "📊 汇总 ${mode_name} 结果..."

    python - << 'PY'
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
    else:
        summaries = sorted(glob.glob(os.path.join(fold_dir, "**", "summary.csv"), recursive=True))
        if not summaries:
            print(f"{mode_name} Fold {fold}: 未找到 summary_partial_*.csv 或 summary.csv，跳过")
            continue
        f = summaries[-1]
    try:
        df = pd.read_csv(f)
        if "val_cindex" not in df.columns:
            print(f"{mode_name} Fold {fold}: 文件缺少 val_cindex 列，跳过")
            continue
        val = float(df["val_cindex"].iloc[-1])
        rows.append({"fold": fold, "val_cindex": val, "source_file": os.path.basename(f)})
    except Exception as e:
        print(f"{mode_name} Fold {fold}: 读取失败，原因：{e}")

if rows:
    out_df = pd.DataFrame(rows).sort_values("fold")
    out_df.to_csv(out_csv, index=False)
    mean_cindex = out_df["val_cindex"].mean()
    print(f"✅ {mode_name} 汇总: {len(rows)}/{k_folds} 折成功, 平均 C-Index: {mean_cindex:.4f}")
else:
    pd.DataFrame(columns=["fold", "val_cindex", "source_file"]).to_csv(out_csv, index=False)
    print(f"❌ {mode_name} 汇总失败: 无可用结果")
PY
}

echo "开始 Fusion 参数扫描"
echo "项目目录：${PROJECT_ROOT}"
echo "癌种：${STUDIES[*]}"
echo "REG 组合：${REG_VALUES[*]}"
echo "DROPOUT 组合：${DROPOUT_VALUES[*]}"
echo "固定参数：Markov blanket=${MARKOV_BLANKET}, TOP_K=${TOP_K}"
echo "结果汇总根目录：${LOG_ROOT}/${TODAY}"

for STUDY in "${STUDIES[@]}"; do
    build_study_paths "${STUDY}"
    mkdir -p "$(dirname "${RESULTS_TXT}")"

    {
        echo "=============================================================="
        echo "${STUDY^^} fusion sweep started at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "REG 组合：${REG_VALUES[*]}"
        echo "DROPOUT 组合：${DROPOUT_VALUES[*]}"
        echo "固定参数：Markov blanket=${MARKOV_BLANKET}, TOP_K=${TOP_K}"
        echo "=============================================================="
    } >> "${RESULTS_TXT}"

    for REG in "${REG_VALUES[@]}"; do
        for ENCODER_DROPOUT in "${DROPOUT_VALUES[@]}"; do
            STUDY_START_TS=$(date +%s)

            echo ""
            echo "=============================================================="
            echo "当前模式：Fusion Only"
            echo "开始运行癌种：${STUDY}，参数组合：REG=${REG}, DROPOUT=${ENCODER_DROPOUT}"
            echo "=============================================================="

            build_study_paths "${STUDY}"

            mkdir -p "$LOG_DIR" "$REPORT_ROOT" "$ABLRESULTS_DIR/fusion"

            export SAMPLE_FEATURE_FILE="${FEATURE_DIR}/fold_0_genes.csv"
            print_run_config "$STUDY" "$FEATURE_DIR" | tee -a "$MAIN_LOG"

            if ! check_required_paths "$SPLIT_DIR" "$FEATURE_DIR" "$STUDY"; then
                {
                    echo "参数组合：REG=${REG}, DROPOUT=${ENCODER_DROPOUT}"
                    echo "固定参数：Markov blanket=${MARKOV_BLANKET}, TOP_K=${TOP_K}"
                    echo "Fusion    平均 C-Index: FAILED"
                    echo "说明：路径检查失败"
                    echo ""
                } >> "${RESULTS_TXT}"
                continue
            fi

            build_early_stop_args
            build_two_stage_args

            run_fusion_mode "$STUDY" "$LABEL_FILE" "$SPLIT_DIR" "$OMICS_DIR" "$FEATURE_DIR" "$DATA_ROOT_DIR" "$ABLRESULTS_DIR"

            export MODE_DIR="${ABLRESULTS_DIR}/fusion"
            export OUT_CSV="${ABLRESULTS_DIR}/fusion/summary.csv"
            export MODE_NAME="Fusion"
            merge_mode_summary "$MODE_DIR" "$OUT_CSV" "$MODE_NAME" | tee -a "$MAIN_LOG"

            FUSION_MEAN="N/A"
            if [ -f "${OUT_CSV}" ]; then
                FUSION_MEAN=$(python - << 'PY' "${OUT_CSV}"
import pandas as pd
import sys
fp = sys.argv[1]
df = pd.read_csv(fp)
if "val_cindex" in df.columns and len(df) > 0:
    print(f"{df['val_cindex'].mean():.6f}")
else:
    print("N/A")
PY
)
            fi

            {
                echo "参数组合：REG=${REG}, DROPOUT=${ENCODER_DROPOUT}"
                echo "固定参数：Markov blanket=${MARKOV_BLANKET}, TOP_K=${TOP_K}"
                echo "Fusion    平均 C-Index: ${FUSION_MEAN}"
                echo ""
            } >> "${RESULTS_TXT}"

            STUDY_END_TS=$(date +%s)
            STUDY_TOTAL_SECONDS=$((STUDY_END_TS - STUDY_START_TS))
            STUDY_HOURS=$((STUDY_TOTAL_SECONDS / 3600))
            STUDY_MINUTES=$(((STUDY_TOTAL_SECONDS % 3600) / 60))
            STUDY_REMAIN_SECONDS=$((STUDY_TOTAL_SECONDS % 60))
            printf "参数组合耗时：%02d:%02d:%02d\n" "$STUDY_HOURS" "$STUDY_MINUTES" "$STUDY_REMAIN_SECONDS" | tee -a "$MAIN_LOG"
        done
    done
done

SCRIPT_END_TS=$(date +%s)
TOTAL_SECONDS=$((SCRIPT_END_TS - SCRIPT_START_TS))
TOTAL_HOURS=$((TOTAL_SECONDS / 3600))
TOTAL_MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
TOTAL_REMAIN_SECONDS=$((TOTAL_SECONDS % 60))

echo ""
echo "全部参数组合运行完成。"
printf "总耗时：%02d:%02d:%02d\n" "$TOTAL_HOURS" "$TOTAL_MINUTES" "$TOTAL_REMAIN_SECONDS"
