# 🚀 批量流程一键运行指南

## 📋 概述

`run_full_pipeline_batch.sh` 脚本可以**一键运行完整流程**，包括：

1. ✅ **mRMR 特征选择** (k=30)
2. ✅ **Stage2 PC算法精炼**
3. ✅ **消融实验** (Gene Only / Text Only / Fusion)

支持**批量处理多个癌症类型**，自动生成详细日志和结果汇总。

---

## ⚡ 快速开始

### 方式1: 运行默认的5种癌症

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 一句命令运行完整流程（brca, blca, luad, stad, hnsc）
bash scripts/run_full_pipeline_batch.sh
```

### 方式2: 指定特定癌症类型

```bash
# 运行单个癌症
bash scripts/run_full_pipeline_batch.sh "brca"

# 运行2-3个癌症
bash scripts/run_full_pipeline_batch.sh "brca blca luad"

# 运行自定义组合
bash scripts/run_full_pipeline_batch.sh "brca kirc lihc"
```

---

## 📊 执行流程详解

### 对每个癌症类型，脚本会自动执行：

```
癌症类型 (e.g., brca)
    ↓
Step 1: 检查数据划分
    ├─ 验证 splits/nested_cv/{study}/ 存在
    └─ 如果缺失，提示用户先运行 create_nested_splits.sh
    ↓
Step 2: mRMR 特征选择 (k=30)
    ├─ 读取 RNA 表达数据
    ├─ 读取临床生存数据
    ├─ 为每个 fold 运行 mRMR 算法
    └─ 输出: features/mrmr_{study}/fold_{0-4}_genes.csv
    ↓
Step 3: Stage2 PC算法精炼
    ├─ 读取 mRMR 筛选结果
    ├─ 应用 PC 算法找因果关联基因
    └─ 输出: features/mrmr_stage2_{study}/fold_{0-4}_genes.csv
    ↓
Step 4: 消融实验
    ├─ Gene Only 模式 (ab_model=2)
    ├─ Text Only 模式 (ab_model=1)
    ├─ Fusion 模式 (ab_model=3)
    └─ 输出: results/ablation_mrmr_stage2/{study}/final_comparison.csv
    ↓
完成 ✅
```

---

## 📁 输出文件结构

### 日志文件

```
log/batch_pipeline_2026-01-31_15-30-45/
├── main.log                    # 主日志（所有癌症的汇总）
├── brca_full.log              # brca 完整流程日志
├── brca_mrmr.log              # brca mRMR 详细日志
├── brca_stage2.log            # brca Stage2 详细日志
├── brca_ablation.log          # brca 消融实验详细日志
├── blca_full.log              # blca 完整流程日志
├── blca_mrmr.log
├── blca_stage2.log
├── blca_ablation.log
└── ... (其他癌症)
```

### 特征文件

```
features/
├── mrmr_brca/                  # mRMR 原始输出
│   ├── fold_0_genes.csv       # ~30 个基因
│   ├── fold_1_genes.csv
│   └── ...
├── mrmr_stage2_brca/          # Stage2 精炼输出
│   ├── fold_0_genes.csv       # ~15-25 个基因
│   ├── fold_1_genes.csv
│   └── ...
├── mrmr_blca/
├── mrmr_stage2_blca/
└── ... (其他癌症)
```

### 消融实验结果

```
results/ablation_mrmr_stage2/
├── brca/
│   ├── gene/
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   ├── ...
│   │   └── summary.csv
│   ├── text/
│   │   └── summary.csv
│   ├── fusion/
│   │   └── summary.csv
│   └── final_comparison.csv    # 最终结果
├── blca/
│   └── final_comparison.csv
└── ... (其他癌症)
```

### 报告文件

```
report/
├── 2026-01-31_brca_ablation_mrmr_stage2_comparison.csv
├── 2026-01-31_blca_ablation_mrmr_stage2_comparison.csv
└── ...
```

---

## 🎯 前置条件

### ✅ 必须完成

在运行批处理脚本前，**必须**先为每个癌症创建数据划分：

```bash
# 为每个癌症创建嵌套交叉验证划分
bash create_nested_splits.sh brca
bash create_nested_splits.sh blca
bash create_nested_splits.sh luad
bash create_nested_splits.sh stad
bash create_nested_splits.sh hnsc
```

### 📂 必需的数据文件

脚本会自动检查以下文件是否存在：

```
splits/nested_cv/{study}/
├── nested_splits_0.csv
├── nested_splits_1.csv
├── nested_splits_2.csv
├── nested_splits_3.csv
└── nested_splits_4.csv

datasets_csv/raw_rna_data/combine/{study}/
└── rna_clean.csv

datasets_csv/clinical_data/
└── tcga_{study}_clinical.csv
```

---

## 📊 实时监控进度

### 方式1: 查看主日志（推荐）

```bash
# 实时查看总体进度
tail -f log/batch_pipeline_*/main.log
```

### 方式2: 查看特定癌症的详细日志

```bash
# 找到最新的日志目录
LOG_DIR=$(ls -td log/batch_pipeline_* | head -1)

# 查看 brca 的 mRMR 日志
tail -f ${LOG_DIR}/brca_mrmr.log

# 查看 brca 的 Stage2 日志
tail -f ${LOG_DIR}/brca_stage2.log

# 查看 brca 的消融实验日志
tail -f ${LOG_DIR}/brca_ablation.log
```

### 方式3: 使用进程监控

```bash
# 查看正在运行的进程
ps aux | grep -E "run_mrmr|run_stage2|run_ablation"

# 查看 Python 进程
watch -n 5 'ps aux | grep python3'
```

---

## ⏱️ 预计耗时

### 单个癌症类型

| 步骤 | 预计耗时 |
|------|---------|
| mRMR (k=30, 5 folds) | 5-15 分钟 |
| Stage2 (5 folds) | 2-5 分钟 |
| 消融实验 (3 模式 × 5 folds) | 30-120 分钟* |
| **总计** | **约 40-140 分钟** |

\* 取决于数据量、GPU性能、训练轮数(epochs)

### 5个癌症类型

| 模式 | 预计总耗时 |
|------|----------|
| **串行执行** | 3-12 小时 |
| 并行执行 (GPU允许) | 可自行修改脚本实现 |

---

## 🎨 输出示例

### 控制台输出示例

```
==============================================
🚀 批量运行完整流程
==============================================
📋 癌症类型: brca blca luad stad hnsc
📊 流程步骤:
   1️⃣  mRMR 特征选择 (k=30)
   2️⃣  Stage2 PC算法精炼
   3️⃣  消融实验 (Gene/Text/Fusion)
==============================================

📁 日志目录: log/batch_pipeline_2026-01-31_15-30-45

==============================================
🧬 开始处理: BRCA
==============================================
[15:30:45] [brca] 检查数据划分 - 开始
[15:30:45] [brca] 检查数据划分 - 通过
[15:30:45] [brca] mRMR特征选择 - 开始 (k=30)
✅
[15:35:20] [brca] mRMR特征选择 - 完成
   📂 生成文件:
      features/mrmr_brca/fold_0_genes.csv
      features/mrmr_brca/fold_1_genes.csv
      features/mrmr_brca/fold_2_genes.csv
      features/mrmr_brca/fold_3_genes.csv
      features/mrmr_brca/fold_4_genes.csv
[15:35:20] [brca] Stage2精炼 - 开始 (PC算法)
✅
[15:38:15] [brca] Stage2精炼 - 完成
   📂 生成文件:
      fold_0_genes.csv: 18 个基因
      fold_1_genes.csv: 22 个基因
      fold_2_genes.csv: 19 个基因
      fold_3_genes.csv: 20 个基因
      fold_4_genes.csv: 21 个基因
[15:38:15] [brca] 消融实验 - 开始 (Gene/Text/Fusion)
✅
[16:45:30] [brca] 消融实验 - 完成
   📊 消融实验结果:
      Gene Only:  0.6234
      Text Only:  0.5891
      Fusion:     0.6543
      提升: +4.96%
   ⏱️  耗时: 75分0秒

==============================================
🧬 开始处理: BLCA
==============================================
[16:45:30] [blca] 检查数据划分 - 开始
...

==============================================
🎉 批处理完成！
==============================================
📊 执行汇总:
   ✅ 成功: 5 个癌症
   ❌ 失败: 0 个癌症
   ⏱️  总耗时: 320分15秒
   📁 日志目录: log/batch_pipeline_2026-01-31_15-30-45

📊 所有成功癌症的结果汇总:
==============================================

🧬 BRCA:
   Gene Only:  0.6234
   Text Only:  0.5891
   Fusion:     0.6543
   提升: +4.96%

🧬 BLCA:
   Gene Only:  0.6015
   Text Only:  0.5723
   Fusion:     0.6298
   提升: +4.70%

...

==============================================

✅ 全部完成！查看详细日志请访问: log/batch_pipeline_2026-01-31_15-30-45
```

---

## 🐛 故障排查

### 问题1: 提示"找不到数据划分"

**错误信息**:
```
❌ 错误: 找不到 splits/nested_cv/brca，请先运行 create_nested_splits.sh brca
```

**解决方案**:
```bash
# 为缺失的癌症创建划分
bash create_nested_splits.sh brca
```

### 问题2: mRMR 失败

**可能原因**:
- RNA 数据文件缺失
- 临床数据文件缺失
- 样本 ID 不匹配

**检查方法**:
```bash
# 查看 mRMR 详细日志
LOG_DIR=$(ls -td log/batch_pipeline_* | head -1)
cat ${LOG_DIR}/brca_mrmr.log
```

### 问题3: Stage2 失败

**可能原因**:
- mRMR 输出文件缺失
- 临床 OS 列缺失

**检查方法**:
```bash
# 验证 mRMR 输出
ls -lh features/mrmr_brca/

# 查看 Stage2 详细日志
LOG_DIR=$(ls -td log/batch_pipeline_* | head -1)
cat ${LOG_DIR}/brca_stage2.log
```

### 问题4: 消融实验失败

**可能原因**:
- GPU 内存不足
- Stage2 特征文件缺失
- 训练数据问题

**检查方法**:
```bash
# 验证 Stage2 输出
ls -lh features/mrmr_stage2_brca/

# 查看消融实验详细日志
LOG_DIR=$(ls -td log/batch_pipeline_* | head -1)
cat ${LOG_DIR}/brca_ablation.log

# 查看具体 fold 的训练日志
cat results/ablation_mrmr_stage2/brca/gene/fold_0/training.log
```

### 问题5: 中途中断后如何继续

**场景**: 运行到 blca 时中断了，已完成 brca

**解决方案**:
```bash
# 只运行剩余的癌症
bash scripts/run_full_pipeline_batch.sh "blca luad stad hnsc"
```

---

## 🔧 高级用法

### 1. 修改 mRMR 的 k 值

编辑脚本第 24 行：

```bash
# 从 k=30 改为 k=50
THRESHOLD=50
```

### 2. 修改训练超参数

编辑 `run_ablation_study_mrmr_stage2.sh` 中的参数：

```bash
EPOCHS=20      # 改为 30
LR=0.00005     # 改为 0.0001
MAX_JOBS=3     # 改为 2 (降低GPU并发)
```

### 3. 只运行特定步骤

**只运行 mRMR**:
```bash
# 手动运行
for study in brca blca luad; do
    python3 preprocessing/CPCG_algo/stage0/run_mrmr.py \
        --study $study --fold all \
        --split_dir splits/nested_cv \
        --data_root_dir datasets_csv/raw_rna_data/combine \
        --clinical_dir datasets_csv/clinical_data \
        --threshold 30
done
```

**只运行 Stage2**:
```bash
for study in brca blca luad; do
    bash scripts/quick_stage2_refine.sh $study
done
```

**只运行消融实验**:
```bash
for study in brca blca luad; do
    bash scripts/run_ablation_study_mrmr_stage2.sh $study
done
```

### 4. 并行运行多个癌症（需要多GPU）

创建自定义脚本：

```bash
#!/bin/bash
# 并行运行（需要多个GPU）

export CUDA_VISIBLE_DEVICES=0
bash scripts/run_full_pipeline_batch.sh "brca" &

export CUDA_VISIBLE_DEVICES=1
bash scripts/run_full_pipeline_batch.sh "blca" &

export CUDA_VISIBLE_DEVICES=2
bash scripts/run_full_pipeline_batch.sh "luad" &

wait
```

---

## 📚 相关文档

- ✅ `run_mrmr.py` - mRMR 特征选择
- ✅ `run_stage2_refinement.py` - Stage2 PC算法精炼
- ✅ `run_ablation_study_mrmr_stage2.sh` - 消融实验（mRMR+Stage2）
- ✅ `README_ablation_comparison.md` - 消融实验详细说明
- ✅ `QUICK_START.md` - 完整工作流程快速入门

---

## 🎉 总结

### 一句命令，完成全流程

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy && bash scripts/run_full_pipeline_batch.sh
```

**包含**:
- ✅ 5种癌症 (brca, blca, luad, stad, hnsc)
- ✅ mRMR 特征选择 (k=30)
- ✅ Stage2 PC算法精炼
- ✅ 消融实验 (Gene/Text/Fusion)
- ✅ 自动生成日志和结果汇总

**输出**:
- ✅ 特征文件: `features/mrmr_stage2_{study}/`
- ✅ 实验结果: `results/ablation_mrmr_stage2/{study}/`
- ✅ 详细日志: `log/batch_pipeline_{timestamp}/`
- ✅ 结果报告: `report/`

🚀 简单、高效、全自动！
