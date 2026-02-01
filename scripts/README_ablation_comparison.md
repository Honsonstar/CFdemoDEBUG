# 消融实验脚本对比说明

## 📋 概述

项目中现在有**两个消融实验脚本**，主要区别在于使用的**基因特征类型**：

1. **run_ablation_study.sh** - 使用 CPCG 原始特征
2. **run_ablation_study_mrmr_stage2.sh** - 使用 mRMR+Stage2 精炼特征 ✨

## 🔄 两个脚本的对比

| 特性 | run_ablation_study.sh | run_ablation_study_mrmr_stage2.sh |
|------|----------------------|-----------------------------------|
| **特征路径** | `features/{study}/` | `features/mrmr_stage2_{study}/` |
| **特征来源** | CPCG 算法筛选 | mRMR + PC 算法精炼 |
| **特征数量** | CPCG 筛选结果 | ~150-180 基因/fold |
| **结果目录** | `results/ablation/{study}/` | `results/ablation_mrmr_stage2/{study}/` |
| **日志目录** | `log/{date}/{study}/` | `log/{date}/{study}_mrmr_stage2/` |
| **报告文件** | `{date}_{study}_ablation_comparison.csv` | `{date}_{study}_ablation_mrmr_stage2_comparison.csv` |
| **特征类型** | 全流程 CPCG | 相关性 + 因果性双重筛选 |

## 🚀 使用方法

### 方式1: 使用 CPCG 原始特征

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 运行消融实验（CPCG特征）
bash scripts/run_ablation_study.sh brca
```

**前置条件**:
- ✅ 需要先运行完整的 CPCG 流程生成特征

### 方式2: 使用 mRMR+Stage2 精炼特征

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 运行消融实验（mRMR+Stage2特征）
bash scripts/run_ablation_study_mrmr_stage2.sh brca
```

**前置条件**:
1. ✅ 运行 mRMR 特征选择
2. ✅ 运行 Stage2 特征精炼

## 📊 主要修改点

### 1. 特征路径修改

```bash
# 原版
FEATURE_DIR="features/${STUDY}"

# 新版（第32行）
FEATURE_DIR="features/mrmr_stage2_${STUDY}"
```

### 2. 结果目录修改

```bash
# 原版
ABLRESULTS_DIR="results/ablation/${STUDY}"

# 新版（第47行）
ABLRESULTS_DIR="results/ablation_mrmr_stage2/${STUDY}"
```

### 3. 日志目录修改

```bash
# 原版
LOG_DIR="log/${TODAY}/${STUDY}"

# 新版（第50行）
LOG_DIR="log/${TODAY}/${STUDY}_mrmr_stage2"
```

### 4. 特征检查函数修改

```bash
# 新版（第95-115行）添加了更详细的提示
echo "🔍 检查 ${study^^} 的 mRMR+Stage2 特征文件..."
...
echo "请先运行:"
echo "  1. python preprocessing/CPCG_algo/stage0/run_mrmr.py --study ${study} --fold all ..."
echo "  2. bash scripts/quick_stage2_refine.sh ${study}"
```

### 5. 日志标识修改

```bash
# 新版添加了特征类型标识
echo "🚀 开始多模态消融实验: ${STUDY} (mRMR+Stage2)"
echo "📁 特征路径: ${FEATURE_DIR}"
```

## 🎯 使用场景

### 场景1: 评估 CPCG 原始特征效果

**目的**: 使用传统 CPCG 全流程筛选的特征

```bash
bash scripts/run_ablation_study.sh brca
```

**适用于**:
- 已有完整 CPCG 流程结果
- 想使用传统方法的基准测试
- 对比不同特征选择方法

### 场景2: 评估 mRMR+Stage2 精炼特征效果

**目的**: 使用相关性+因果性双重筛选的特征

```bash
bash scripts/run_ablation_study_mrmr_stage2.sh brca
```

**适用于**:
- 想使用更精炼的特征集
- 评估因果关联基因的预测效果
- 对比不同特征数量的影响

### 场景3: 对比两种特征选择方法

**步骤**:

```bash
# 1. 运行 CPCG 版本
bash scripts/run_ablation_study.sh brca

# 2. 运行 mRMR+Stage2 版本
bash scripts/run_ablation_study_mrmr_stage2.sh brca

# 3. 对比结果
echo "CPCG 结果:"
cat results/ablation/brca/final_comparison.csv

echo "mRMR+Stage2 结果:"
cat results/ablation_mrmr_stage2/brca/final_comparison.csv
```

## 📁 输出目录结构对比

### CPCG 版本

```
results/ablation/brca/
├── gene/
│   ├── fold_0/
│   ├── fold_1/
│   ├── ...
│   └── summary.csv
├── text/
│   └── summary.csv
├── fusion/
│   └── summary.csv
└── final_comparison.csv
```

### mRMR+Stage2 版本

```
results/ablation_mrmr_stage2/brca/
├── gene/
│   ├── fold_0/
│   ├── fold_1/
│   ├── ...
│   └── summary.csv
├── text/
│   └── summary.csv
├── fusion/
│   └── summary.csv
└── final_comparison.csv
```

## 🔍 前置条件详解

### CPCG 版本前置条件

1. **数据划分**:
   ```bash
   bash create_nested_splits.sh brca
   ```

2. **CPCG 特征选择**:
   ```bash
   bash run_all_cpog.sh brca
   ```
   生成: `features/brca/fold_{0-4}_genes.csv`

### mRMR+Stage2 版本前置条件

1. **数据划分**:
   ```bash
   bash create_nested_splits.sh brca
   ```

2. **mRMR 特征选择**:
   ```bash
   python preprocessing/CPCG_algo/stage0/run_mrmr.py \
     --study brca --fold all \
     --split_dir splits/nested_cv \
     --data_root_dir datasets_csv/raw_rna_data/combine \
     --clinical_dir datasets_csv/clinical_data \
     --threshold 200
   ```
   生成: `features/mrmr_brca/fold_{0-4}_genes.csv`

3. **Stage2 特征精炼**:
   ```bash
   bash scripts/quick_stage2_refine.sh brca
   ```
   生成: `features/mrmr_stage2_brca/fold_{0-4}_genes.csv`

## 📈 预期结果对比

### 特征数量

| 版本 | 典型基因数量 |
|------|-------------|
| CPCG | 根据 CPCG 算法筛选结果 |
| mRMR+Stage2 | ~150-180 基因/fold |

### 性能预期

- **CPCG**: 使用传统方法，基准性能
- **mRMR+Stage2**: 
  - 特征更精炼
  - 因果关联更强
  - 可能提升模型性能
  - 计算效率更高

## 🐛 常见问题

### Q1: 提示"mRMR+Stage2 特征文件不完整"

**A**: 按顺序运行前置步骤：

```bash
# 步骤1: mRMR
python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca --fold all ...

# 步骤2: Stage2
bash scripts/quick_stage2_refine.sh brca

# 步骤3: 验证
ls -lh features/mrmr_stage2_brca/
```

### Q2: 两个版本的结果会冲突吗？

**A**: 不会！输出目录完全分离：
- CPCG: `results/ablation/{study}/`
- mRMR+Stage2: `results/ablation_mrmr_stage2/{study}/`

### Q3: 应该用哪个版本？

**A**: 根据研究目的选择：
- **追求基准对比**: 使用 CPCG 版本
- **追求特征精炼**: 使用 mRMR+Stage2 版本
- **完整研究**: 两个都运行，对比分析

### Q4: 如何确认使用的是哪个版本？

**A**: 查看运行时的输出：

```bash
# CPCG 版本
🚀 开始多模态消融实验: brca

# mRMR+Stage2 版本
🚀 开始多模态消融实验: brca (mRMR+Stage2)
📁 特征路径: features/mrmr_stage2_brca
```

## 💡 最佳实践

### 1. 完整工作流程

```bash
# ===== 使用 mRMR+Stage2 特征 =====

# Step 1: 数据划分
bash create_nested_splits.sh brca

# Step 2: mRMR 特征选择
python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca --fold all \
  --split_dir splits/nested_cv \
  --data_root_dir datasets_csv/raw_rna_data/combine \
  --clinical_dir datasets_csv/clinical_data \
  --threshold 200

# Step 3: Stage2 特征精炼
bash scripts/quick_stage2_refine.sh brca

# Step 4: 基因比对（可选）
bash scripts/quick_mrmr_compare.sh brca stage2

# Step 5: 运行消融实验
bash scripts/run_ablation_study_mrmr_stage2.sh brca
```

### 2. 批量运行多个癌种

```bash
#!/bin/bash

STUDIES="brca blca luad stad hnsc"

for study in $STUDIES; do
    echo "Processing $study with mRMR+Stage2..."
    
    # 运行消融实验
    bash scripts/run_ablation_study_mrmr_stage2.sh $study
    
    echo "$study completed!"
done
```

### 3. 结果对比脚本

```bash
#!/bin/bash
# 对比 CPCG 和 mRMR+Stage2 的消融实验结果

STUDY=$1

echo "======================================"
echo "消融实验结果对比: $STUDY"
echo "======================================"

echo ""
echo "CPCG 版本结果:"
cat results/ablation/${STUDY}/final_comparison.csv

echo ""
echo "mRMR+Stage2 版本结果:"
cat results/ablation_mrmr_stage2/${STUDY}/final_comparison.csv

echo ""
echo "======================================"
```

## 📚 相关文档

- ✅ `run_ablation_study.sh` - CPCG 版本消融实验
- ✅ `run_ablation_study_mrmr_stage2.sh` - mRMR+Stage2 版本消融实验
- ✅ `run_mrmr.py` - mRMR 特征选择
- ✅ `run_stage2_refinement.py` - Stage2 特征精炼
- ✅ `README_stage2_refinement.md` - Stage2 详细说明
- ✅ `README_compare_modes.md` - 基因比对模式切换

## 🎉 总结

现在你有两个版本的消融实验脚本：

| 需求 | 脚本 |
|------|------|
| 使用 CPCG 特征 | `bash scripts/run_ablation_study.sh brca` |
| 使用 mRMR+Stage2 特征 | `bash scripts/run_ablation_study_mrmr_stage2.sh brca` |

**关键区别**: 只有**基因特征路径**不同，其他完全一致！

祝实验顺利！🎊
