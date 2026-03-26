#!/bin/bash

# ====================================================================
# 汇总消融实验结果脚本
# 只汇总结果，不重新训练
# ====================================================================
#
# 使用方法:
#   bash scripts/summarize_results.sh blca              # 汇总单个癌种 (默认mRMR+Stage2)
#   bash scripts/summarize_results.sh cpcg blca       # 汇总CPCG结果
#   bash scripts/summarize_results.sh                  # 汇总所有
#
# 结果路径:
#   - mRMR+Stage2: results/ablation_mrmr_stage2/{study}/
#   - CPCG:       results/ablation_cpcg/{study}/
#   - 日志: log/summarize_{mode}_{date}.log
# ====================================================================

MODE="mrmr_stage2"
STUDY=${2:-}

# 日志文件
TODAY=$(date +%Y-%m-%d)
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# 解析参数
if [ "$1" = "cpcg" ]; then
    MODE="cpcg"
    STUDY=${2:-}
elif [ -n "$1" ]; then
    STUDY=$1
fi

LOG_FILE="${LOG_DIR}/summarize_${MODE}_${TODAY}.log"
echo "日志文件: $LOG_FILE"

# 汇总单个结果
summarize_one() {
    local mode=$1
    local study=$2

    if [ "$mode" = "cpcg" ]; then
        local results_dir="results/ablation_cpcg/${study}"
        local mode_name="CPCG"
    else
        local results_dir="results/ablation_mrmr_stage2/${study}"
        local mode_name="mRMR+Stage2"
    fi

    echo "📊 汇总 ${study} (${mode_name})..." | tee -a "$LOG_FILE"

    if [ ! -d "$results_dir" ]; then
        echo "❌ 结果目录不存在: $results_dir" | tee -a "$LOG_FILE"
        return 1
    fi

    # 汇总各模式
    for subdir in gene text fusion; do
        local sub_path="${results_dir}/${subdir}"
        if [ ! -d "$sub_path" ]; then
            echo "  ⚠️  ${subdir}: 目录不存在" | tee -a "$LOG_FILE"
            continue
        fi

        python3 -c "
import pandas as pd
import glob
import os

results_dir = '${sub_path}'
dfs = []

for fold_dir in sorted(glob.glob(f'{results_dir}/fold_*', recursive=True)):
    if not os.path.isdir(fold_dir):
        continue

    fold_name = os.path.basename(fold_dir)
    try:
        fold_num = int(fold_name.split('_')[-1])
    except:
        continue

    # 找summary文件
    partial_files = glob.glob(f'{fold_dir}/**/summary_partial_*.csv', recursive=True)
    if partial_files:
        summary_file = max(partial_files, key=os.path.getmtime)
    else:
        summary_files = glob.glob(f'{fold_dir}/**/summary.csv', recursive=True)
        if summary_files:
            summary_file = max(summary_files, key=os.path.getmtime)
        else:
            continue

    try:
        df = pd.read_csv(summary_file)
        df['fold'] = fold_num
        dfs.append(df)
    except:
        continue

if dfs:
    result = pd.concat(dfs).sort_values('fold')
    result.to_csv(f'{results_dir}/summary.csv', index=False)
    mean_ci = result['val_cindex'].mean()
    print(f'  ✓ ${subdir}: {len(dfs)} folds, C-index: {mean_ci:.4f}')
else:
    print(f'  ✗ ${subdir}: 无结果')
" 2>&1 | tee -a "$LOG_FILE"
    done

    # 生成对比表
    python3 -c "
import pandas as pd
import os

results_dir = '${results_dir}'
comparison_data = []

for fold in range(5):
    row = {'Fold': fold}

    for subdir in ['gene', 'text', 'fusion']:
        summary_file = f'{results_dir}/{subdir}/summary.csv'
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            fold_row = df[df['fold'] == fold]
            if not fold_row.empty:
                row[f'{subdir.capitalize()}_C_Index'] = fold_row['val_cindex'].values[0]

    comparison_data.append(row)

if comparison_data:
    df = pd.DataFrame(comparison_data)
    df.to_csv(f'{results_dir}/final_comparison.csv', index=False)
    print(f'\n📈 最终对比:')
    print(f'   Gene:  {df[\"Gene_C_Index\"].mean():.4f}')
    print(f'   Text:  {df[\"Text_C_Index\"].mean():.4f}')
    print(f'   Fusion: {df[\"Fusion_C_Index\"].mean():.4f}')
" 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# 主逻辑
if [ -n "$STUDY" ]; then
    summarize_one "$MODE" "$STUDY"
else
    # 汇总所有癌种
    for study in blca brca coadread hnsc stad; do
        summarize_one "$MODE" "$study"
    done
fi

echo "✅ 汇总完成，日志: $LOG_FILE"
