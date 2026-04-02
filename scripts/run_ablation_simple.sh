#!/bin/bash

# Simplified ablation runner:
# - fixed stable feature path
# - only Gene Only and Fusion
# - fail fast if training does not start or does not finish successfully
# - never reuse stale summaries from previous runs

ALL_CANCERS=("brca" "blca" "hnsc" "stad" "coadread")

if [ -z "$1" ]; then
    echo "No cancer type specified, running all supported studies."
    CANCERS_TO_RUN=("${ALL_CANCERS[@]}")
else
    CANCERS_TO_RUN=("$@")
fi

TODAY=$(date +%Y-%m-%d)

for STUDY in "${CANCERS_TO_RUN[@]}"; do
    echo ""
    echo "################################################################"
    echo "### Running study: ${STUDY^^}"
    echo "################################################################"

    LABEL_FILE="datasets_csv/clinical_data/tcga_${STUDY}_clinical.csv"
    SPLIT_DIR="splits/CGI_nested_cv/${STUDY}"
    FEATURE_DIR="preprocessing/CGI_py/features/stable/${STUDY}"
    OMICS_DIR="datasets_csv/raw_rna_data/combine/${STUDY}"
    DATA_ROOT_DIR="data/${STUDY}/pt_files"

    ABLRESULTS_DIR="results/ablation/${STUDY}"
    LOG_DIR="log/${TODAY}/${STUDY}"
    REPORT_DIR="report"
    MAIN_LOG="${LOG_DIR}/ablation_simple.log"

    SEED=42
    K_FOLDS=5
    EPOCHS=20
    LR=0.00005
    MAX_JOBS=3

    mkdir -p "${LOG_DIR}" "${REPORT_DIR}" "${ABLRESULTS_DIR}"/{gene,fusion}

    echo "Start simplified ablation: ${STUDY}"
    echo "Date: ${TODAY}"
    echo "Log: ${MAIN_LOG}"
    echo "=============================================="

    export STUDY
    export LABEL_FILE
    export SPLIT_DIR
    export ABLRESULTS_DIR

    if [ ! -d "${SPLIT_DIR}" ]; then
        echo "ERROR: split directory not found: ${SPLIT_DIR}"
        echo "Run first: python3 preprocessing/CGI/preprocess_test.py"
        exit 1
    fi

    check_features() {
        local all_exist=true
        echo "Checking stable feature files for ${STUDY^^}..."
        for fold in $(seq 0 $((K_FOLDS-1))); do
            local file="${FEATURE_DIR}/fold_${fold}_genes.csv"
            if [ -f "${file}" ]; then
                echo "  OK fold ${fold}: $(basename "${file}")"
            else
                echo "  MISSING fold ${fold}: $(basename "${file}")"
                all_exist=false
            fi
        done

        if [ "${all_exist}" = false ]; then
            echo "ERROR: stable feature files are incomplete."
            echo "Run first: python3 preprocessing/CGI_py/find_genes_stable.py"
            exit 1
        fi
    }

    check_features

    run_batch_wait() {
        local log_file=$1
        shift
        local -a pids=("$@")
        local status=0
        local idx=0
        for pid in "${pids[@]}"; do
            local fold="${BATCH_FOLDS[$idx]}"
            local results_dir="${BATCH_DIRS[$idx]}"
            if ! wait "${pid}"; then
                status=1
                echo "  FAIL fold ${fold}: training process exited with error" | tee -a "${log_file}"
                if [ -f "${results_dir}/.failure_reason" ]; then
                    echo "  Failure reason for fold ${fold}:" | tee -a "${log_file}"
                    sed 's/^/    /' "${results_dir}/.failure_reason" | tee -a "${log_file}"
                fi
            fi
            ((idx++))
        done
        return ${status}
    }

    run_mode() {
        local mode_name=$1
        local ab_model=$2
        local results_subdir=$3
        local log_file=$4

        echo "" | tee -a "${log_file}"
        echo "==============================================" | tee -a "${log_file}"
        echo "${mode_name}" | tee -a "${log_file}"
        echo "==============================================" | tee -a "${log_file}"

        local mode_failed=0
        local running_jobs=0
        local -a pids=()
        local -a folds_seen=()
        BATCH_FOLDS=()
        BATCH_DIRS=()

        for fold in $(seq 0 $((K_FOLDS-1))); do
            local results_dir="${ABLRESULTS_DIR}/${results_subdir}/fold_${fold}"
            local fold_log="${results_dir}/training.log"
            local run_marker="${results_dir}/.run_started"
            local exit_code_file="${results_dir}/.exit_code"
            local success_marker="${results_dir}/.run_succeeded"
            local failed_marker="${results_dir}/.run_failed"
            local reason_file="${results_dir}/.failure_reason"

            mkdir -p "${results_dir}"
            rm -f "${run_marker}" "${exit_code_file}" "${success_marker}" "${failed_marker}" "${reason_file}"
            date +%s > "${run_marker}"

            echo "  Launch fold ${fold}..." | tee -a "${log_file}"

            (
                python3 main.py \
                    --study tcga_${STUDY} \
                    --k_start ${fold} \
                    --k_end $((fold + 1)) \
                    --split_dir "${SPLIT_DIR}" \
                    --results_dir "${results_dir}" \
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
                    > "${fold_log}" 2>&1

                exit_code=$?
                echo "${exit_code}" > "${exit_code_file}"
                if [ ${exit_code} -eq 0 ]; then
                    touch "${success_marker}"
                else
                    touch "${failed_marker}"
                    tail -n 40 "${fold_log}" > "${reason_file}" 2>/dev/null || true
                fi
                exit ${exit_code}
            ) &

            local pid=$!
            echo "    PID: ${pid}" >> "${log_file}"

            pids+=("${pid}")
            BATCH_FOLDS+=("${fold}")
            BATCH_DIRS+=("${results_dir}")
            folds_seen+=("${fold}")
            ((running_jobs++))

            if [ ${running_jobs} -ge ${MAX_JOBS} ]; then
                echo "  Reached max parallel jobs ${MAX_JOBS}, waiting..." | tee -a "${log_file}"
                if ! run_batch_wait "${log_file}" "${pids[@]}"; then
                    mode_failed=1
                fi
                pids=()
                BATCH_FOLDS=()
                BATCH_DIRS=()
                running_jobs=0
            fi
        done

        if [ ${running_jobs} -gt 0 ]; then
            if ! run_batch_wait "${log_file}" "${pids[@]}"; then
                mode_failed=1
            fi
        fi

        for fold in "${folds_seen[@]}"; do
            local results_dir="${ABLRESULTS_DIR}/${results_subdir}/fold_${fold}"
            if [ ! -f "${results_dir}/.run_succeeded" ]; then
                echo "  FAIL fold ${fold}: no success marker for current run" | tee -a "${log_file}"
                if [ -f "${results_dir}/.failure_reason" ]; then
                    echo "  Failure reason for fold ${fold}:" | tee -a "${log_file}"
                    sed 's/^/    /' "${results_dir}/.failure_reason" | tee -a "${log_file}"
                fi
                mode_failed=1
            fi
        done

        echo "  Completed mode: ${mode_name}" | tee -a "${log_file}"
        return ${mode_failed}
    }

    GENE_LOG="${ABLRESULTS_DIR}/gene_training.log"
    if ! run_mode "Gene Only" 2 "gene" "${GENE_LOG}"; then
        echo "ERROR: Gene Only training failed. Stop before summary to avoid reading stale results." | tee -a "${GENE_LOG}" "${MAIN_LOG}"
        exit 1
    fi

    echo "" | tee -a "${GENE_LOG}"
    echo "Summarizing Gene Only results..." | tee -a "${GENE_LOG}"
    export GENE_SUMMARY="${ABLRESULTS_DIR}/gene/summary.csv"

    python3 << 'EOF' | tee -a "${GENE_LOG}"
import glob
import os
import pandas as pd

base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'gene')
summary_path = os.environ.get('GENE_SUMMARY', '')

dfs = []
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    if not os.path.isdir(fold_dir):
        continue
    try:
        fold_num = int(fold_dir.split('_')[-1])
    except Exception:
        continue

    success_marker = os.path.join(fold_dir, '.run_succeeded')
    run_started = os.path.join(fold_dir, '.run_started')
    if not os.path.exists(success_marker):
        print(f"  x Fold {fold_num}: current run did not succeed, skip stale files")
        continue

    run_started_ts = os.path.getmtime(run_started) if os.path.exists(run_started) else 0
    partial_files = [p for p in glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)
                     if os.path.getmtime(p) >= run_started_ts]

    if partial_files:
        f = max(partial_files, key=os.path.getmtime)
        df = pd.read_csv(f)
        df['fold'] = fold_num
        dfs.append(df)
        print(f"  ok Fold {fold_num}: {os.path.basename(f)}")
        continue

    summary_files = [p for p in glob.glob(f"{fold_dir}/**/summary.csv", recursive=True)
                     if os.path.getmtime(p) >= run_started_ts]
    if summary_files:
        f = max(summary_files, key=os.path.getmtime)
        df = pd.read_csv(f)
        if 'folds' in df.columns:
            df['fold'] = fold_num
        dfs.append(df)
        print(f"  ok Fold {fold_num}: {os.path.basename(f)}")
    else:
        print(f"  x Fold {fold_num}: no fresh summary generated in current run")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(summary_path, index=False)
    print(f"OK Gene Only summary: {len(dfs)}/5 folds")
    print(f"Mean C-Index: {result['val_cindex'].mean():.4f}")
else:
    print("ERROR: no usable Gene Only results from current run")
    pd.DataFrame(columns=['fold', 'val_cindex']).to_csv(summary_path, index=False)
    raise SystemExit(1)
EOF

    FUSION_LOG="${ABLRESULTS_DIR}/fusion_training.log"
    if ! run_mode "Fusion" 3 "fusion" "${FUSION_LOG}"; then
        echo "ERROR: Fusion training failed. Stop before summary to avoid reading stale results." | tee -a "${FUSION_LOG}" "${MAIN_LOG}"
        exit 1
    fi

    echo "" | tee -a "${FUSION_LOG}"
    echo "Summarizing Fusion results..." | tee -a "${FUSION_LOG}"
    export FUSION_SUMMARY="${ABLRESULTS_DIR}/fusion/summary.csv"

    python3 << 'EOF' | tee -a "${FUSION_LOG}"
import glob
import os
import pandas as pd

base_path = os.environ.get('ABLRESULTS_DIR', '')
results_dir = os.path.join(base_path, 'fusion')
summary_path = os.environ.get('FUSION_SUMMARY', '')

dfs = []
for fold_dir in sorted(glob.glob(f"{results_dir}/fold_*", recursive=True)):
    if not os.path.isdir(fold_dir):
        continue
    try:
        fold_num = int(fold_dir.split('_')[-1])
    except Exception:
        continue

    success_marker = os.path.join(fold_dir, '.run_succeeded')
    run_started = os.path.join(fold_dir, '.run_started')
    if not os.path.exists(success_marker):
        print(f"  x Fold {fold_num}: current run did not succeed, skip stale files")
        continue

    run_started_ts = os.path.getmtime(run_started) if os.path.exists(run_started) else 0
    partial_files = [p for p in glob.glob(f"{fold_dir}/**/summary_partial_*.csv", recursive=True)
                     if os.path.getmtime(p) >= run_started_ts]

    if partial_files:
        f = max(partial_files, key=os.path.getmtime)
        df = pd.read_csv(f)
        df['fold'] = fold_num
        dfs.append(df)
        print(f"  ok Fold {fold_num}: {os.path.basename(f)}")
        continue

    summary_files = [p for p in glob.glob(f"{fold_dir}/**/summary.csv", recursive=True)
                     if os.path.getmtime(p) >= run_started_ts]
    if summary_files:
        f = max(summary_files, key=os.path.getmtime)
        df = pd.read_csv(f)
        if 'folds' in df.columns:
            df['fold'] = fold_num
        dfs.append(df)
        print(f"  ok Fold {fold_num}: {os.path.basename(f)}")
    else:
        print(f"  x Fold {fold_num}: no fresh summary generated in current run")

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(summary_path, index=False)
    print(f"OK Fusion summary: {len(dfs)}/5 folds")
    print(f"Mean C-Index: {result['val_cindex'].mean():.4f}")
else:
    print("ERROR: no usable Fusion results from current run")
    pd.DataFrame(columns=['fold', 'val_cindex']).to_csv(summary_path, index=False)
    raise SystemExit(1)
EOF

    export FINAL_CSV="${ABLRESULTS_DIR}/final_comparison.csv"
    export REPORT_CSV="report/${TODAY}_${STUDY}_ablation_simple.csv"

    python3 << 'EOF' | tee -a "${MAIN_LOG}"
import numpy as np
import os
import pandas as pd

study = os.environ.get('STUDY', '')
ablation_dir = f"results/ablation/{study}"
final_csv = os.environ.get('FINAL_CSV', '')
report_csv = os.environ.get('REPORT_CSV', '')

gene_summary_path = f"{ablation_dir}/gene/summary.csv"
fusion_summary_path = f"{ablation_dir}/fusion/summary.csv"

if not os.path.exists(gene_summary_path) or not os.path.exists(fusion_summary_path):
    raise SystemExit("Missing mode summaries; abort final comparison.")

gene_summary = pd.read_csv(gene_summary_path)
fusion_summary = pd.read_csv(fusion_summary_path)

comparison_data = []
for fold in range(5):
    row = {'Fold': fold}
    gene_val = gene_summary[gene_summary['fold'] == fold]['val_cindex'].values
    fusion_val = fusion_summary[fusion_summary['fold'] == fold]['val_cindex'].values
    row['Gene_C_Index'] = gene_val[0] if len(gene_val) > 0 else np.nan
    row['Fusion_C_Index'] = fusion_val[0] if len(fusion_val) > 0 else np.nan
    comparison_data.append(row)

df = pd.DataFrame(comparison_data)
df.to_csv(final_csv, index=False)
df.to_csv(report_csv, index=False)

gene_mean = df['Gene_C_Index'].mean()
fusion_mean = df['Fusion_C_Index'].mean()

print("=" * 50)
print("Simplified ablation summary")
print(df.to_string(index=False))
print("=" * 50)
print(f"Gene Only mean C-Index: {gene_mean:.4f}")
print(f"Fusion mean C-Index:    {fusion_mean:.4f}")
if gene_mean > 0:
    improvement = ((fusion_mean - gene_mean) / gene_mean) * 100
    print(f"Fusion vs Gene Only: {improvement:+.2f}%")
print(f"Saved comparison: {final_csv}")
EOF

    echo ""
    echo "Study completed: ${STUDY}"
    echo "Results directory: ${ABLRESULTS_DIR}"
    echo "Comparison table: ${FINAL_CSV}"
    echo "Report file: ${REPORT_CSV}"
done

echo ""
echo "################################################################"
echo "### All simplified ablation runs completed"
echo "################################################################"
echo "Results directory: results/ablation/"
echo "Report directory: report/"
