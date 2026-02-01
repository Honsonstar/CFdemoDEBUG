# 基因比对脚本 - 模式切换说明

## 📋 概述

`compare_mrmr_gene_signatures.py` 和 `quick_mrmr_compare.sh` 现在支持**两种模式**：

1. **mRMR 模式** (默认): 比对 mRMR 原始筛选的基因
2. **Stage2 模式**: 比对经过 PC 算法精炼后的基因

## 🔄 两种模式的区别

| 特性 | mRMR 模式 | Stage2 模式 |
|------|----------|------------|
| **输入路径** | `features/mrmr_{study}/` | `features/mrmr_stage2_{study}/` |
| **基因来源** | mRMR 原始筛选（200个） | PC算法精炼后（150-180个） |
| **筛选标准** | 最大相关性+最小冗余 | 与OS直接因果关联 |
| **热图配色** | 橙色 (Oranges) | 紫色 (Purples) |
| **输出文件后缀** | `_mrmr_` | `_stage2_` |

## 🚀 使用方法

### 方式1: 使用快捷脚本 (推荐)

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 比对 mRMR 原始筛选的基因（默认）
bash scripts/quick_mrmr_compare.sh brca

# 比对 Stage2 精炼后的基因
bash scripts/quick_mrmr_compare.sh brca stage2
```

### 方式2: 直接使用 Python 脚本

```bash
# 比对 mRMR 原始筛选的基因
python scripts/compare_mrmr_gene_signatures.py --study brca

# 比对 Stage2 精炼后的基因
python scripts/compare_mrmr_gene_signatures.py --study brca --stage2
```

## 📊 输出文件对比

### mRMR 模式输出

```
results/
├── brca_mrmr_overlap_stats.csv          # 折间重合度统计
├── brca_mrmr_all_genes.csv              # 所有基因列表
└── mrmr_gene_overlap_heatmap_brca.png   # 热图 (橙色)
```

### Stage2 模式输出

```
results/
├── brca_stage2_overlap_stats.csv         # 折间重合度统计
├── brca_stage2_all_genes.csv             # 所有基因列表
└── stage2_gene_overlap_heatmap_brca.png  # 热图 (紫色)
```

## 📈 使用场景

### 场景1: 分析 mRMR 筛选效果

**目的**: 评估 mRMR 算法在不同折间的一致性

```bash
bash scripts/quick_mrmr_compare.sh brca
```

**关注指标**:
- 折间重合率
- 与全局 CPCG 基因的对比

### 场景2: 分析 Stage2 精炼效果

**目的**: 评估 PC 算法精炼后的基因稳定性

```bash
bash scripts/quick_mrmr_compare.sh brca stage2
```

**关注指标**:
- 精炼后的重合率是否提高
- 基因数量减少了多少

### 场景3: 对比两种模式

**步骤**:

```bash
# 1. 比对 mRMR 原始基因
bash scripts/quick_mrmr_compare.sh brca

# 2. 比对 Stage2 精炼基因
bash scripts/quick_mrmr_compare.sh brca stage2

# 3. 对比结果
echo "mRMR 平均一致性:"
grep "平均一致性" results/brca_mrmr_overlap_stats.csv

echo "Stage2 平均一致性:"
grep "平均一致性" results/brca_stage2_overlap_stats.csv
```

## 🔍 详细示例

### 示例1: brca mRMR 模式

```bash
$ bash scripts/quick_mrmr_compare.sh brca

==========================================
基因签名快速比对
==========================================
   癌种: brca
   模式: MRMR
==========================================

📊 使用 mRMR 原始筛选的基因 (路径: features/mrmr_brca)
✓ MRMR筛选 Fold 0: 100 基因, 598 训练样本
✓ MRMR筛选 Fold 1: 100 基因, 598 训练样本
...

📊 1. MRMR筛选基因的嵌套CV内部稳定性 (Fold间重合度)
------------------------------------------------------------
Folds      | 交集数      | 重合率(%)    
--------------------------------------------------
0 vs 1    | 22       | 22.0%
0 vs 2    | 20       | 20.0%
...
--------------------------------------------------
👉 平均一致性 (重合率): 0.2630

输出文件:
- results/brca_mrmr_overlap_stats.csv
- results/brca_mrmr_all_genes.csv
- results/mrmr_gene_overlap_heatmap_brca.png (橙色热图)
```

### 示例2: brca Stage2 模式

```bash
$ bash scripts/quick_mrmr_compare.sh brca stage2

==========================================
基因签名快速比对
==========================================
   癌种: brca
   模式: MRMR + Stage2 (PC算法)
==========================================

📊 使用 Stage2 精炼后的基因 (路径: features/mrmr_stage2_brca)
✓ Stage2精炼 Fold 0: 99 基因, 598 训练样本
✓ Stage2精炼 Fold 1: 95 基因, 598 训练样本
...

📊 1. Stage2精炼基因的嵌套CV内部稳定性 (Fold间重合度)
------------------------------------------------------------
Folds      | 交集数      | 重合率(%)    
--------------------------------------------------
0 vs 1    | 24       | 25.3%
0 vs 2    | 22       | 23.2%
...
--------------------------------------------------
👉 平均一致性 (重合率): 0.2850

输出文件:
- results/brca_stage2_overlap_stats.csv
- results/brca_stage2_all_genes.csv
- results/stage2_gene_overlap_heatmap_brca.png (紫色热图)
```

## 🔧 前置条件

### mRMR 模式

需要先运行 mRMR 特征选择：

```bash
python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca --fold all \
  --split_dir splits/nested_cv \
  --data_root_dir datasets_csv/raw_rna_data/combine \
  --clinical_dir datasets_csv/clinical_data \
  --threshold 200
```

### Stage2 模式

需要先运行 Stage2 特征精炼：

```bash
# 方式1: 使用快捷脚本
bash scripts/quick_stage2_refine.sh brca

# 方式2: 使用 Python 脚本
python preprocessing/CPCG_algo/stage0/run_stage2_refinement.py \
  --study brca --fold all \
  --clinical_dir datasets_csv/clinical_data
```

## 📊 结果解读

### 重合率分析

- **mRMR 重合率**: 通常 20-30%
  - 原因: mRMR 基于相关性，不同训练集可能选出不同的高相关基因
  
- **Stage2 重合率**: 通常 25-35% (略高)
  - 原因: PC 算法筛选因果关联，更稳定

### 基因数量变化

典型情况（k=200）：
1. **mRMR**: 200 基因/fold
2. **Stage2**: 150-180 基因/fold (减少 10-25%)

### 热图配色

- **橙色** (mRMR): 温暖色调，表示相关性筛选
- **紫色** (Stage2): 冷静色调，表示因果筛选

## 🐛 常见问题

### Q1: 提示"缺少必要文件"

**A**: 根据提示运行相应的前置步骤：

```bash
# mRMR 模式缺少文件
python preprocessing/CPCG_algo/stage0/run_mrmr.py --study brca --fold all ...

# Stage2 模式缺少文件
bash scripts/quick_stage2_refine.sh brca
```

### Q2: 如何同时比对两种模式？

**A**: 依次运行两个命令：

```bash
bash scripts/quick_mrmr_compare.sh brca        # mRMR 模式
bash scripts/quick_mrmr_compare.sh brca stage2 # Stage2 模式
```

### Q3: 输出文件会覆盖吗？

**A**: 不会！两种模式的输出文件名不同：
- mRMR: `brca_mrmr_*.csv`、`mrmr_gene_overlap_heatmap_brca.png`
- Stage2: `brca_stage2_*.csv`、`stage2_gene_overlap_heatmap_brca.png`

### Q4: 如何恢复默认模式？

**A**: 默认就是 mRMR 模式，不加 `stage2` 参数即可：

```bash
bash scripts/quick_mrmr_compare.sh brca  # 自动使用 mRMR 模式
```

## 💡 最佳实践

### 1. 完整工作流程

```bash
# Step 1: mRMR 特征选择 (k=200)
python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca --fold all ... --threshold 200

# Step 2: 比对 mRMR 基因
bash scripts/quick_mrmr_compare.sh brca

# Step 3: Stage2 特征精炼
bash scripts/quick_stage2_refine.sh brca

# Step 4: 比对 Stage2 基因
bash scripts/quick_mrmr_compare.sh brca stage2

# Step 5: 对比两种模式的结果
```

### 2. 批量处理多个癌种

```bash
#!/bin/bash

STUDIES="brca blca luad stad hnsc"

for study in $STUDIES; do
    echo "Processing $study..."
    
    # mRMR 模式
    bash scripts/quick_mrmr_compare.sh $study
    
    # Stage2 模式（如果文件存在）
    if [ -d "features/mrmr_stage2_$study" ]; then
        bash scripts/quick_mrmr_compare.sh $study stage2
    fi
done
```

### 3. 生成对比报告

```bash
# 对比 mRMR 和 Stage2 的平均重合率
echo "Cancer\tmRMR_Overlap\tStage2_Overlap"
for study in brca blca luad; do
    mrmr_rate=$(python -c "import pandas as pd; df=pd.read_csv('results/${study}_mrmr_overlap_stats.csv'); print(f'{df[\"Overlap_Rate\"].mean():.4f}')")
    stage2_rate=$(python -c "import pandas as pd; df=pd.read_csv('results/${study}_stage2_overlap_stats.csv'); print(f'{df[\"Overlap_Rate\"].mean():.4f}')")
    echo "$study\t$mrmr_rate\t$stage2_rate"
done
```

## 📚 相关文档

- `compare_mrmr_gene_signatures.py` - 核心比对脚本
- `quick_mrmr_compare.sh` - 快捷运行脚本
- `run_mrmr.py` - mRMR 特征选择
- `run_stage2_refinement.py` - Stage2 特征精炼
- `README_stage2_refinement.md` - Stage2 详细说明

## 🎯 总结

现在你可以轻松切换两种模式：

| 命令 | 模式 | 用途 |
|------|------|------|
| `bash scripts/quick_mrmr_compare.sh brca` | mRMR | 评估相关性筛选 |
| `bash scripts/quick_mrmr_compare.sh brca stage2` | Stage2 | 评估因果筛选 |

通过对比两种模式的结果，你可以：
1. ✅ 了解 mRMR 和 Stage2 的筛选效果
2. ✅ 评估基因选择的稳定性
3. ✅ 选择最适合你研究的特征集

祝使用愉快！🎉
