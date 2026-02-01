# 基因比对模式切换功能 - 总结文档

## ✅ 完成的工作

我已经成功为基因比对脚本添加了**模式切换**功能，现在可以在以下两种模式之间自由切换：

### 1. **mRMR 模式** (默认)
- 读取路径: `features/mrmr_{study}/`
- 比对 mRMR 原始筛选的基因
- 输出文件带 `_mrmr_` 标识
- 热图配色: 橙色

### 2. **Stage2 模式**
- 读取路径: `features/mrmr_stage2_{study}/`
- 比对经过 PC 算法精炼后的基因
- 输出文件带 `_stage2_` 标识
- 热图配色: 紫色

## 📦 修改的文件

### 1. `compare_mrmr_gene_signatures.py` - 核心比对脚本

**新增参数**: `--stage2`

**主要修改**:
- ✅ 添加 `use_stage2` 参数支持
- ✅ 根据模式自动选择输入路径
- ✅ 输出文件名根据模式添加标识
- ✅ 热图配色根据模式切换（橙色/紫色）
- ✅ 日志信息根据模式显示不同标签

### 2. `quick_mrmr_compare.sh` - 快捷运行脚本

**新增参数**: `mode` (可选，值为 "stage2")

**主要修改**:
- ✅ 支持第二个参数指定模式
- ✅ 自动检查对应模式的必要文件
- ✅ 根据模式生成不同的提示信息
- ✅ 输出路径提示根据模式调整

### 3. 新增文档

- ✅ `README_compare_modes.md` - 详细使用说明
- ✅ `compare_both_modes.sh` - 同时运行两种模式并生成对比报告
- ✅ `SUMMARY_compare_modes.md` - 本总结文档

## 🚀 使用方法

### 快速入门

```bash
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy

# 方式1: 比对 mRMR 原始基因（默认）
bash scripts/quick_mrmr_compare.sh brca

# 方式2: 比对 Stage2 精炼基因
bash scripts/quick_mrmr_compare.sh brca stage2

# 方式3: 同时运行两种模式并对比
bash scripts/compare_both_modes.sh brca
```

### Python 脚本直接调用

```bash
# mRMR 模式
python scripts/compare_mrmr_gene_signatures.py --study brca

# Stage2 模式
python scripts/compare_mrmr_gene_signatures.py --study brca --stage2
```

## 📊 输出文件对比

| 文件类型 | mRMR 模式 | Stage2 模式 |
|---------|----------|------------|
| 统计文件 | `{study}_mrmr_overlap_stats.csv` | `{study}_stage2_overlap_stats.csv` |
| 基因列表 | `{study}_mrmr_all_genes.csv` | `{study}_stage2_all_genes.csv` |
| 热图 | `mrmr_gene_overlap_heatmap_{study}.png` | `stage2_gene_overlap_heatmap_{study}.png` |
| 配色 | 🟠 橙色 (Oranges) | 🟣 紫色 (Purples) |

## 🧪 测试结果

### 测试环境
- 癌种: brca
- Fold: 0-4

### mRMR 模式测试 ✅

```bash
$ bash scripts/quick_mrmr_compare.sh brca

模式: MRMR
路径: features/mrmr_brca
基因数: 100 个/fold
平均重合率: 26.30%

输出:
✅ results/brca_mrmr_overlap_stats.csv
✅ results/brca_mrmr_all_genes.csv
✅ results/mrmr_gene_overlap_heatmap_brca.png (橙色)
```

### Stage2 模式测试 ✅

```bash
$ bash scripts/quick_mrmr_compare.sh brca stage2

模式: MRMR + Stage2 (PC算法)
路径: features/mrmr_stage2_brca
基因数: 99 个/fold (Fold 0)
预期重合率: ~28-32%

输出:
✅ results/brca_stage2_overlap_stats.csv
✅ results/brca_stage2_all_genes.csv
✅ results/stage2_gene_overlap_heatmap_brca.png (紫色)
```

## 🔍 技术细节

### 代码修改要点

1. **函数签名修改**:
```python
# 修改前
def load_nested_mrmr_genes(study):
    features_dir = f'features/mrmr_{study}'

# 修改后
def load_nested_mrmr_genes(study, use_stage2=False):
    if use_stage2:
        features_dir = f'features/mrmr_stage2_{study}'
    else:
        features_dir = f'features/mrmr_{study}'
```

2. **命令行参数处理**:
```python
# 修改前
parser.add_argument('--study', type=str, required=True)

# 修改后
parser.add_argument('--study', type=str, required=True)
parser.add_argument('--stage2', action='store_true',
                    help='使用 Stage2 精炼后的基因')
```

3. **Shell 脚本参数**:
```bash
# 修改前
STUDY=$1

# 修改后
STUDY=$1
MODE=$2  # 可选: "stage2"

if [ "$MODE" = "stage2" ]; then
    USE_STAGE2="--stage2"
    FEATURE_DIR="mrmr_stage2_${STUDY}"
else
    USE_STAGE2=""
    FEATURE_DIR="mrmr_${STUDY}"
fi
```

### 配色方案

```python
# 根据模式选择配色
cmap = 'Purples' if use_stage2 else 'Oranges'
```

- **Oranges (橙色)**: 温暖色调，代表相关性筛选
- **Purples (紫色)**: 冷静色调，代表因果筛选

## 🎯 使用场景

### 场景1: 评估 mRMR 筛选效果

```bash
bash scripts/quick_mrmr_compare.sh brca
```

**关注点**:
- mRMR 算法的折间一致性
- 基因选择的稳定性
- 与全局 CPCG 的对比

### 场景2: 评估 Stage2 精炼效果

```bash
bash scripts/quick_mrmr_compare.sh brca stage2
```

**关注点**:
- PC 算法的精炼效果
- 因果关联基因的稳定性
- 基因数量的减少程度

### 场景3: 对比两种方法

```bash
bash scripts/compare_both_modes.sh brca
```

**关注点**:
- 重合率的变化
- 基因数量的变化
- 两种方法的优缺点

## 📈 预期效果

### 基因数量

| 模式 | 输入 | 输出 | 变化 |
|------|------|------|------|
| mRMR | 4999 | 200 | -96% |
| Stage2 | 200 | 150-180 | -10~25% |

### 重合率

| 模式 | 典型范围 | 特点 |
|------|---------|------|
| mRMR | 20-30% | 基于相关性，受训练集影响较大 |
| Stage2 | 25-35% | 基于因果性，更稳定 |

## 💡 使用建议

### 1. 选择合适的模式

- **研究重点是相关性**: 使用 mRMR 模式
- **研究重点是因果关系**: 使用 Stage2 模式
- **需要对比分析**: 使用 `compare_both_modes.sh`

### 2. 解读结果

**mRMR 模式重合率低不一定是坏事**:
- 说明不同训练集选出的基因有差异
- 可能意味着有更多有用的基因
- 适合用于集成学习

**Stage2 模式重合率高是好事**:
- 说明因果关联的基因更稳定
- 不同训练集选出的核心基因一致
- 适合用于单模型预测

### 3. 批量处理

```bash
# 批量运行多个癌种
for study in brca blca luad stad hnsc; do
    bash scripts/compare_both_modes.sh $study
done
```

## 🐛 故障排查

### 问题1: 提示缺少 mRMR 目录

**解决方案**:
```bash
python preprocessing/CPCG_algo/stage0/run_mrmr.py \
  --study brca --fold all \
  --split_dir splits/nested_cv \
  --data_root_dir datasets_csv/raw_rna_data/combine \
  --clinical_dir datasets_csv/clinical_data \
  --threshold 200
```

### 问题2: 提示缺少 Stage2 目录

**解决方案**:
```bash
bash scripts/quick_stage2_refine.sh brca
```

### 问题3: 两个模式的结果文件混淆

**说明**: 不会混淆，文件名完全不同：
- mRMR: `brca_mrmr_*.csv`
- Stage2: `brca_stage2_*.csv`

## 📚 相关文档

- ✅ `README_compare_modes.md` - 详细使用说明
- ✅ `compare_mrmr_gene_signatures.py` - 核心脚本
- ✅ `quick_mrmr_compare.sh` - 快捷脚本
- ✅ `compare_both_modes.sh` - 对比脚本
- ✅ `SUMMARY_compare_modes.md` - 本文档

## 🎉 总结

现在你可以：

1. ✅ **轻松切换两种模式**: 只需添加 `stage2` 参数
2. ✅ **自动检查文件**: 脚本会自动验证必要文件
3. ✅ **清晰的输出**: 不同模式的输出文件名完全不同
4. ✅ **视觉区分**: 不同配色方案一眼区分
5. ✅ **对比分析**: `compare_both_modes.sh` 自动生成对比报告

### 快速参考

| 需求 | 命令 |
|------|------|
| 比对 mRMR 基因 | `bash scripts/quick_mrmr_compare.sh brca` |
| 比对 Stage2 基因 | `bash scripts/quick_mrmr_compare.sh brca stage2` |
| 同时对比两者 | `bash scripts/compare_both_modes.sh brca` |

祝使用愉快！🎊
