# MRMR基因签名比对工具使用说明

## 📋 概述

这套工具用于比对嵌套交叉验证（Nested CV）中MRMR筛选出的基因在不同折（fold）之间的重合度和一致性。

## 📂 文件说明

- **compare_mrmr_gene_signatures.py**: 核心Python脚本，执行MRMR基因签名的比对分析
- **quick_mrmr_compare.sh**: 自动化Shell脚本，检查必要文件并运行比对
- **README_mrmr_compare.md**: 本说明文档

## 🔧 使用方法

### 快速开始

```bash
# 进入项目目录
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 运行MRMR基因比对（以brca为例）
bash scripts/quick_mrmr_compare.sh brca
```

### 支持的癌种

任何已完成MRMR特征选择的癌种，例如：
- `brca` (乳腺癌)
- `blca` (膀胱癌)
- `luad` (肺腺癌)
- 等等

## 📊 输出结果

运行脚本后，会在 `results/` 目录下生成以下文件：

### 1. 统计文件
- **`{study}_mrmr_overlap_stats.csv`**: 折间重合度统计
  - `Fold_A`, `Fold_B`: 比对的两个折
  - `Intersection`: 交集基因数
  - `Jaccard`: Jaccard相似度系数
  - `Overlap_Rate`: 重合率（相对于较小基因集）

### 2. 基因列表
- **`{study}_mrmr_all_genes.csv`**: 所有折中筛选出的基因列表
  - `gene`: 基因名称
  - `fold`: 所属的折编号

### 3. 可视化
- **`mrmr_gene_overlap_heatmap_{study}.png`**: 基因重合度热力图
  - 展示各折之间的重合率矩阵
  - 颜色越深表示重合度越高

## 🔍 先决条件

运行此脚本前，需要确保：

1. **MRMR特征选择已完成**
   ```bash
   python preprocessing/CPCG_algo/stage0/run_mrmr.py --study {癌种}
   ```

2. **必要的目录和文件存在**
   - `features/mrmr_{study}/fold_{0-4}_genes.csv`: 各折的MRMR筛选基因
   - `splits/nested_cv/{study}/nested_splits_{0-4}.csv`: 数据分割信息（可选，用于统计训练样本数）
   - `preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_{study}/`: 全局CPCG基因（可选，用于对比）

## 📈 解读结果

### 平均一致性（Overlap Rate）
- **>0.5**: 非常好的一致性，基因选择稳定
- **0.3-0.5**: 中等一致性，可接受
- **<0.3**: 较低一致性，可能需要调整参数或检查数据质量

### 全局CPCG vs MRMR基因
- 比较传统CPCG方法与MRMR方法筛选的基因差异
- 交集较小是正常的，因为两种方法的筛选策略不同

## 🆚 与原始脚本的区别

| 特性 | compare_gene_signatures.py | compare_mrmr_gene_signatures.py |
|------|---------------------------|--------------------------------|
| 输入路径 | `features/{study}/` | `features/mrmr_{study}/` |
| 输出标识 | 无后缀 | 带`mrmr_`前缀 |
| 热图配色 | Blues | Oranges |
| 用途 | CPCG基因比对 | MRMR基因比对 |

## 💡 示例输出

```
============================================================
MRMR基因签名比对: brca
============================================================
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
```

## 🐛 常见问题

### Q1: 提示"缺少MRMR目录"
**A**: 请先运行MRMR特征选择：
```bash
python preprocessing/CPCG_algo/stage0/run_mrmr.py --study {癌种}
```

### Q2: 找不到全局CPCG文件
**A**: 全局CPCG文件是可选的，不影响折间比对。如需对比，请先运行全局CPCG分析。

### Q3: 重合率很低是否正常？
**A**: 这取决于：
- MRMR参数设置（特别是选择的基因数量）
- 训练集的差异程度
- 数据的噪声水平
- 一般20-40%的重合率是可以接受的

## 📞 联系与支持

如有问题或建议，请查看项目文档或联系开发团队。
