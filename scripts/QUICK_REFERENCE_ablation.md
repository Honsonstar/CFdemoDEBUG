# 消融实验脚本 - 快速参考卡片

## 🎯 两个版本一览

```
┌─────────────────────────────────────────────────────────────┐
│  消融实验脚本版本选择                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  版本1: CPCG 特征                                           │
│  ├─ 脚本: run_ablation_study.sh                            │
│  ├─ 特征: features/{study}/                                │
│  ├─ 结果: results/ablation/{study}/                        │
│  └─ 用法: bash scripts/run_ablation_study.sh brca         │
│                                                             │
│  版本2: mRMR+Stage2 特征 ✨                                │
│  ├─ 脚本: run_ablation_study_mrmr_stage2.sh                │
│  ├─ 特征: features/mrmr_stage2_{study}/                    │
│  ├─ 结果: results/ablation_mrmr_stage2/{study}/            │
│  └─ 用法: bash scripts/run_ablation_study_mrmr_stage2.sh brca │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 快速命令

### CPCG 版本

```bash
# 完整流程
bash create_nested_splits.sh brca
bash run_all_cpog.sh brca
bash scripts/run_ablation_study.sh brca
```

### mRMR+Stage2 版本 ⭐

```bash
# 完整流程
bash create_nested_splits.sh brca

python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca --fold all \
  --split_dir splits/nested_cv \
  --data_root_dir datasets_csv/raw_rna_data/combine \
  --clinical_dir datasets_csv/clinical_data \
  --threshold 200

bash scripts/quick_stage2_refine.sh brca

bash scripts/run_ablation_study_mrmr_stage2.sh brca
```

## 🔑 关键区别

| 项目 | CPCG | mRMR+Stage2 |
|------|------|-------------|
| **特征路径** | `features/{study}/` | `features/mrmr_stage2_{study}/` |
| **基因数** | CPCG 筛选 | ~150-180 |
| **筛选方法** | CPCG 全流程 | 相关性+因果性 |
| **计算时间** | 较长 | 较短 |
| **结果目录** | `results/ablation/{study}/` | `results/ablation_mrmr_stage2/{study}/` |

## ⚡ 一键对比

```bash
#!/bin/bash
# 快速对比两个版本的结果

STUDY="brca"

echo "CPCG 平均 C-Index:"
python -c "import pandas as pd; df=pd.read_csv('results/ablation/${STUDY}/final_comparison.csv'); print(f\"Gene: {df['Gene_C_Index'].mean():.4f}, Fusion: {df['Fusion_C_Index'].mean():.4f}\")"

echo "mRMR+Stage2 平均 C-Index:"
python -c "import pandas as pd; df=pd.read_csv('results/ablation_mrmr_stage2/${STUDY}/final_comparison.csv'); print(f\"Gene: {df['Gene_C_Index'].mean():.4f}, Fusion: {df['Fusion_C_Index'].mean():.4f}\")"
```

## 📊 结果文件

### CPCG 版本
```
results/ablation/brca/final_comparison.csv
report/2026-01-31_brca_ablation_comparison.csv
```

### mRMR+Stage2 版本
```
results/ablation_mrmr_stage2/brca/final_comparison.csv
report/2026-01-31_brca_ablation_mrmr_stage2_comparison.csv
```

## 🚨 故障排查

### 问题: 找不到特征文件

**CPCG 版本**:
```bash
# 检查
ls features/brca/fold_*.csv

# 如果缺失，运行
bash run_all_cpog.sh brca
```

**mRMR+Stage2 版本**:
```bash
# 检查
ls features/mrmr_stage2_brca/fold_*.csv

# 如果缺失，依次运行
python preprocessing/CPCG_algo/stage0/run_mrmr.py --study brca --fold all ...
bash scripts/quick_stage2_refine.sh brca
```

## 💡 推荐使用

### 新项目推荐: mRMR+Stage2 ⭐

**优势**:
- ✅ 特征更精炼（150-180 vs 原始数量）
- ✅ 因果关联更强（PC 算法筛选）
- ✅ 计算效率更高
- ✅ 可解释性更好

**步骤**:
```bash
bash scripts/run_ablation_study_mrmr_stage2.sh brca
```

### 基准对比: CPCG

**适用场景**:
- 已有 CPCG 结果
- 需要与传统方法对比
- 文献对比基准

**步骤**:
```bash
bash scripts/run_ablation_study.sh brca
```

## 📚 文档索引

- 📖 `README_ablation_comparison.md` - 详细对比说明
- 📖 `run_ablation_study.sh` - CPCG 版本脚本
- 📖 `run_ablation_study_mrmr_stage2.sh` - mRMR+Stage2 版本脚本
- 📖 `README_stage2_refinement.md` - Stage2 原理说明

## 🎯 记住这个

**唯一区别**: 基因特征路径
- CPCG: `features/{study}/`
- mRMR+Stage2: `features/mrmr_stage2_{study}/`

**其他完全一致**: 训练逻辑、超参数、输出格式！

---

**快速开始**: 
```bash
bash scripts/run_ablation_study_mrmr_stage2.sh brca
```

🎉 就这么简单！
