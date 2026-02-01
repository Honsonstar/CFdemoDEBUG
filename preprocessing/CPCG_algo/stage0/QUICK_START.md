# 🚀 快速开始指南 - mRMR + Stage2 工作流程

## 📋 概述

这是一个完整的特征选择工作流程，包含两个步骤：
1. **mRMR**: 基于最大相关性和最小冗余筛选基因
2. **Stage2**: 使用 PC 算法进一步精炼，保留与生存时间直接相关的基因

## 🎯 一键运行（推荐）

### 运行完整工作流程

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# Step 1: mRMR 特征选择 (k=200)
python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca \
  --fold all \
  --split_dir splits/nested_cv \
  --data_root_dir datasets_csv/raw_rna_data/combine \
  --clinical_dir datasets_csv/clinical_data \
  --threshold 200

# Step 2: Stage2 PC算法精炼
bash scripts/quick_stage2_refine.sh brca

# Step 3: 基因签名比对（可选）
bash scripts/quick_mrmr_compare.sh brca
```

## 📊 分步运行

### Step 1: mRMR 特征选择

```bash
cd preprocessing/CPCG_algo/stage0

# 单个 fold
python run_mrmr.py \
  --study brca \
  --fold 0 \
  --split_dir ../../../splits/nested_cv \
  --data_root_dir ../../../datasets_csv/raw_rna_data/combine \
  --clinical_dir ../../../datasets_csv/clinical_data \
  --threshold 200

# 所有 folds
python run_mrmr.py \
  --study brca \
  --fold all \
  --split_dir ../../../splits/nested_cv \
  --data_root_dir ../../../datasets_csv/raw_rna_data/combine \
  --clinical_dir ../../../datasets_csv/clinical_data \
  --threshold 200
```

**输出**: `features/mrmr_brca/fold_{0-4}_genes.csv`

### Step 2: Stage2 特征精炼

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 方式1: 使用快捷脚本（推荐）
bash scripts/quick_stage2_refine.sh brca

# 方式2: 直接运行 Python 脚本
python preprocessing/CPCG_algo/stage0/run_stage2_refinement.py \
  --study brca \
  --fold all \
  --clinical_dir datasets_csv/clinical_data
```

**输出**: `features/mrmr_stage2_brca/fold_{0-4}_genes.csv`

## 📁 输出文件结构

```
features/
├── mrmr_brca/                      # Step 1: mRMR 输出
│   ├── fold_0_genes.csv           (200 基因 x 样本)
│   ├── fold_1_genes.csv
│   ├── fold_2_genes.csv
│   ├── fold_3_genes.csv
│   └── fold_4_genes.csv
│
└── mrmr_stage2_brca/              # Step 2: Stage2 输出
    ├── fold_0_genes.csv           (150-180 基因 x 样本)
    ├── fold_1_genes.csv
    ├── fold_2_genes.csv
    ├── fold_3_genes.csv
    └── fold_4_genes.csv
```

## 🔄 批量处理多个癌种

```bash
#!/bin/bash
# 批量处理脚本示例

STUDIES="brca blca luad stad hnsc"

for study in $STUDIES; do
    echo "=========================================="
    echo "Processing: $study"
    echo "=========================================="
    
    # Step 1: mRMR
    python preprocessing/CPCG_algo/stage0/run_mrmr.py \
      --study $study \
      --fold all \
      --split_dir splits/nested_cv \
      --data_root_dir datasets_csv/raw_rna_data/combine \
      --clinical_dir datasets_csv/clinical_data \
      --threshold 200
    
    # Step 2: Stage2
    bash scripts/quick_stage2_refine.sh $study
    
    echo "✅ $study completed!"
    echo ""
done

echo "🎉 All studies completed!"
```

## ⚙️ 自定义参数

### mRMR 参数

```bash
python run_mrmr.py \
  --study brca \
  --fold all \
  --split_dir ../../../splits/nested_cv \
  --data_root_dir ../../../datasets_csv/raw_rna_data/combine \
  --clinical_dir ../../../datasets_csv/clinical_data \
  --threshold 200 \          # 选择的基因数量（默认: 200）
  --n_jobs -1                # 并行任务数（-1: 使用所有CPU）
```

### Stage2 参数修改

在 `run_stage2_refinement.py` 第 163 行修改：

```python
# 更严格的筛选
G = skeleton(data, alpha=0.05, max_l=2)  # alpha: 0.10 -> 0.05

# 更深的条件集
G = skeleton(data, alpha=0.10, max_l=3)  # max_l: 2 -> 3

# 更近的 Markov Blanket
neighbors = list(nx.single_source_shortest_path_length(
    G_nx, OS_idx, cutoff=1  # cutoff: 2 -> 1
).keys())
```

## 📊 查看结果

### 统计基因数量

```bash
# mRMR 基因数
wc -l features/mrmr_brca/fold_0_genes.csv

# Stage2 基因数
wc -l features/mrmr_stage2_brca/fold_0_genes.csv

# 对比
echo "mRMR 基因数: $(tail -n +2 features/mrmr_brca/fold_0_genes.csv | wc -l)"
echo "Stage2 基因数: $(tail -n +2 features/mrmr_stage2_brca/fold_0_genes.csv | wc -l)"
```

### 查看基因列表

```bash
# mRMR 基因
cut -d',' -f1 features/mrmr_brca/fold_0_genes.csv | tail -n +2

# Stage2 基因
cut -d',' -f1 features/mrmr_stage2_brca/fold_0_genes.csv | tail -n +2
```

### 对比基因差异

```bash
# 提取基因名
cut -d',' -f1 features/mrmr_brca/fold_0_genes.csv | tail -n +2 > /tmp/mrmr_genes.txt
cut -d',' -f1 features/mrmr_stage2_brca/fold_0_genes.csv | tail -n +2 > /tmp/stage2_genes.txt

# 找出被 Stage2 过滤掉的基因
comm -23 <(sort /tmp/mrmr_genes.txt) <(sort /tmp/stage2_genes.txt)
```

## 🎓 理解输出

### 输出格式

所有文件格式一致：
- **行**: 基因名称
- **列**: 样本ID（TCGA-XX-XXXX）
- **值**: log2(TPM+1) 表达值

示例：
```csv
gene_name,TCGA-3C-AALI,TCGA-4H-AAAK,...
GSTT2,1.0151,-1.7322,...
MYBPC1,-2.6349,-1.7322,...
```

### 特征数量变化

典型情况（k=200）：
- **原始基因组**: ~20,000 基因
- **mRMR 筛选**: 200 基因 ⬇️ 99%
- **Stage2 精炼**: 150-180 基因 ⬇️ 10-25%

## ⏱️ 预计运行时间

| 步骤 | 单 Fold | 5 Folds | 备注 |
|------|---------|---------|------|
| mRMR (k=200) | 5-8 分钟 | 25-40 分钟 | 取决于CPU |
| Stage2 | 30-60 秒 | 3-5 分钟 | 已优化 |
| **总计** | 6-9 分钟 | 30-45 分钟 | |

## 🐛 故障排查

### 问题1: 找不到输入文件

```bash
# 检查 split 文件
ls splits/nested_cv/brca/

# 检查表达数据
ls datasets_csv/raw_rna_data/combine/brca/

# 检查临床数据
ls datasets_csv/clinical_data/
```

### 问题2: mRMR 运行中断

```bash
# 检查进程
ps aux | grep run_mrmr

# 终止进程
kill -9 <PID>

# 从特定 fold 继续
python run_mrmr.py --study brca --fold 2 ...
```

### 问题3: Stage2 找不到 mRMR 输出

```bash
# 检查 mRMR 输出
ls -lh features/mrmr_brca/

# 如果缺失，重新运行 mRMR
python run_mrmr.py --study brca --fold all ...
```

## 📚 相关文档

- **run_mrmr.py**: mRMR 特征选择脚本
- **run_stage2_refinement.py**: Stage2 精炼脚本
- **README_stage2_refinement.md**: Stage2 详细文档
- **SUMMARY_stage2.md**: 创建总结和测试结果
- **quick_stage2_refine.sh**: Stage2 快捷脚本
- **quick_mrmr_compare.sh**: 基因比对脚本

## 💡 最佳实践

1. **测试单个 fold**
   ```bash
   # 先测试单个 fold 确保流程正确
   python run_mrmr.py --study brca --fold 0 ...
   python run_stage2_refinement.py --study brca --fold 0 ...
   ```

2. **批量运行**
   ```bash
   # 确认无误后批量运行
   python run_mrmr.py --study brca --fold all ...
   bash scripts/quick_stage2_refine.sh brca
   ```

3. **保存日志**
   ```bash
   # 重定向输出到日志文件
   bash scripts/quick_stage2_refine.sh brca 2>&1 | tee logs/brca_stage2.log
   ```

4. **验证输出**
   ```bash
   # 每个步骤后验证输出文件
   ls -lh features/mrmr_brca/
   ls -lh features/mrmr_stage2_brca/
   ```

## 🎯 下一步

完成特征选择后，你可以：

1. **训练模型**: 使用精炼后的特征训练预后模型
2. **特征分析**: 分析被选中的基因生物学意义
3. **对比实验**: 对比 mRMR vs Stage2 的模型性能
4. **可视化**: 绘制基因重合度热图

祝使用愉快！🎉
