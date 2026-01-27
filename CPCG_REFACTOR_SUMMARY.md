# CPCG筛选脚本重构总结

## 重构概述

成功重构了CPCG筛选流程的三个关键脚本，解决了参数解析错误和逻辑缺失问题，实现了清晰的模块化设计。

## 🎯 重构目标

- ✅ **统一参数解析**：使用argparse处理命令行参数
- ✅ **清晰的数据流**：明确划分文件解析和特征筛选逻辑
- ✅ **模块化设计**：每个脚本职责单一，易于维护
- ✅ **错误处理**：完善的参数验证和错误提示
- ✅ **可追溯性**：清晰的日志输出和状态提示

---

## 📋 重构内容详述

### 任务1：新建 `run_cpog_nested_cv.py`

**文件路径**：`/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/run_cpog_nested_cv.py`

**核心特性**：

1. **参数解析（argparse）**
   ```python
   parser.add_argument('--study', type=str, required=True, help='癌症类型')
   parser.add_argument('--fold', type=int, required=True, help='折数 (0-4)')
   parser.add_argument('--split_file', type=str, required=True, help='划分文件路径')
   ```

2. **划分文件解析**
   - 支持逗号分隔的字符串或列表格式
   - 提取train_ids, val_ids, test_ids
   - 详细的数据统计输出

3. **NestedCVFeatureSelector集成**
   ```python
   from preprocessing.CPCG_algo.nested_cv_wrapper import NestedCVFeatureSelector

   selector = NestedCVFeatureSelector(
       study=study,
       data_root_dir='preprocessing/CPCG_algo/raw_data',
       threshold=100,
       n_jobs=-1
   )
   ```

4. **文件管理**
   - 自动创建features目录
   - 将结果复制到`features/{study}/fold_{fold}_genes.csv`
   - 详细的验证和错误提示

**输出示例**：
```
============================================================
CPCG嵌套交叉验证特征筛选
============================================================
Study: blca
Fold: 0
Split file: /path/to/splits_0.csv
============================================================

📖 读取划分文件: /path/to/splits_0.csv
✅ 划分文件解析成功:
   Train样本数: 180
   Val样本数: 45
   Test样本数: 45

📁 特征输出目录: features/blca

🔄 初始化CPCG特征选择器...
   Study: blca
   Data root: preprocessing/CPCG_algo/raw_data
   Threshold: 100
   n_jobs: -1

🚀 开始CPCG特征筛选 (Fold 0)...
...
✅ CPCG特征筛选完成!
   特征文件: features/blca/fold_0_genes.csv
   筛选基因数: 100
```

---

### 任务2：重写 `scripts/run_cpog_nested.sh`

**文件路径**：`/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/scripts/run_cpog_nested.sh`

**职责**：单折CPCG筛选执行器

**核心逻辑**：

1. **参数处理**
   ```bash
   STUDY=$1      # 癌症类型
   FOLD=$2       # 折数 (0-4)
   SPLIT_BASE=$3 # 划分文件基础目录

   # 构造划分文件路径
   SPLIT_FILE="${SPLIT_BASE}/splits_${FOLD}.csv"
   ```

2. **文件验证**
   - 检查3个参数是否完整
   - 验证划分文件是否存在
   - 提供详细的使用说明

3. **调用Python脚本**
   ```bash
   CMD="python3 run_cpog_nested_cv.py --study \"${STUDY}\" --fold \"${FOLD}\" --split_file \"${SPLIT_FILE}\""
   eval $CMD
   ```

4. **错误处理**
   - 捕获并报告Python脚本错误
   - 适当的退出码处理

**使用示例**：
```bash
# 基本用法
bash scripts/run_cpog_nested.sh blca 0 "/path/to/splits"

# 错误示例（缺少参数）
bash scripts/run_cpog_nested.sh
# 输出：用法说明和使用示例
```

---

### 任务3：重写 `scripts/run_all_cpog.sh`

**文件路径**：`/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/scripts/run_all_cpog.sh`

**职责**：CPCG筛选总控脚本

**核心逻辑**：

1. **硬编码数据路径**
   ```bash
   SPLIT_BASE="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_${STUDY}"
   ```

2. **批量文件检查**
   - 检查5个划分文件是否存在（splits_0.csv到splits_4.csv）
   - 提供清晰的缺失文件提示

3. **串行执行控制**
   ```bash
   for fold in {0..4}; do
       bash scripts/run_cpog_nested.sh $STUDY $fold "$SPLIT_BASE"
       if [ $? -ne 0 ]; then
           echo "❌ 折 $fold 失败"
           exit 1
       fi
   done
   ```

4. **结果验证**
   - 检查所有特征文件是否生成
   - 统计基因数量
   - 生成汇总报告

5. **统计输出**
   ```python
   # 统计每折基因数
   gene_counts = []
   for fold in range(5):
       df = pd.read_csv(f'{features_dir}/fold_{fold}_genes.csv')
       gene_counts.append(len(df))

   # 计算折间重叠
   overlaps = []
   for i in range(5):
       for j in range(i+1, 5):
           overlap_pct = calculate_overlap(i, j)
           overlaps.append(overlap_pct)

   avg_overlap = sum(overlaps) / len(overlaps)
   ```

**输出示例**：
```
==========================================
一键运行所有折的CPCG筛选
==========================================
   癌种: blca
   划分基础目录: /root/autodl-tmp/.../splits/5foldcv_ramdom/tcga_blca
==========================================

🔍 检查外部划分文件...
✅ 外部划分文件检查通过

📁 特征输出目录: features/blca

🚀 开始运行所有折的CPCG筛选...
==========================================

>>> 折数: 0 / 4 <<<
------------------------------------------
✅ 折 0 完成 (耗时: 180s)
...

✅ 所有折CPCG筛选完成!
   总耗时: 950s (15m)

🔍 验证结果...
   ✓ Fold 0: 100 个基因
   ✓ Fold 1: 98 个基因
   ...

📊 生成基因筛选汇总...
   Fold 0: 100 个基因
   Fold 1: 98 个基因
   ...

📈 统计:
   平均每折基因数: 99.0
   总唯一基因数: 247

🔍 折间基因重叠分析:
   Fold 0 vs 1: 65.2% 重叠
   Fold 0 vs 2: 58.3% 重叠
   ...

   平均重叠率: 61.5%
   ✅ 重叠率适中，特征有一定稳定性

✅ 汇总保存到: features/blca/summary.csv

🎉 CPCG筛选流程全部完成!
```

---

## 🔄 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_all_cpog.sh (总控)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. 接收STUDY参数                                          │  │
│  │ 2. 硬编码SPLIT_BASE路径                                    │  │
│  │ 3. 检查splits_0.csv到splits_4.csv                        │  │
│  │ 4. 循环0-4折                                               │  │
│  │    └─ 调用 run_cpog_nested.sh                             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ 调用
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                scripts/run_cpog_nested.sh (单折执行)              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. 接收STUDY, FOLD, SPLIT_BASE                            │  │
│  │ 2. 构造SPLIT_FILE="${SPLIT_BASE}/splits_${FOLD}.csv"    │  │
│  │ 3. 检查文件存在性                                         │  │
│  │ 4. 调用Python脚本                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ 调用
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              run_cpog_nested_cv.py (核心逻辑)                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. argparse解析: --study, --fold, --split_file            │  │
│  │ 2. 解析split_file (train_idx, val_idx, test_idx)         │  │
│  │ 3. 创建NestedCVFeatureSelector                            │  │
│  │ 4. 调用selector.select_features_for_fold()               │  │
│  │ 5. 复制到features/{study}/fold_{fold}_genes.csv         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ 使用
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│            NestedCVFeatureSelector (特征筛选核心)                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. 读取表达数据和临床数据                                  │  │
│  │ 2. 运行Stage1 (parametric + semi-parametric)            │  │
│  │ 3. 运行Stage2 (causal discovery)                         │  │
│  │ 4. 生成特征文件                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 对比重构前后

### 重构前

| 问题 | 描述 |
|------|------|
| 参数解析 | 使用sys.argv手写解析，易出错 |
| 逻辑混乱 | 所有逻辑混在一个脚本中 |
| 文件管理 | 手动复制文件，无验证 |
| 错误处理 | 缺乏参数验证和错误提示 |
| 维护性 | 代码耦合度高，难以修改 |

### 重构后

| 改进点 | 描述 |
|--------|------|
| 参数解析 | argparse统一处理，支持--help |
| 逻辑清晰 | 三层架构：总控→单折→核心 |
| 文件管理 | 自动创建、复制、验证 |
| 错误处理 | 完善的参数验证和错误提示 |
| 维护性 | 模块化设计，易于扩展和修改 |

---

## ⚙️ 使用方法

### 1. 运行所有折

```bash
# 基本用法
bash scripts/run_all_cpog.sh blca

# 如果缺少文件，会提示：
# ❌ 缺少外部划分文件，请确保文件位于:
#    /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/
```

### 2. 运行单折

```bash
# 基本用法
bash scripts/run_cpog_nested.sh blca 0 "/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca"

# 查看帮助
bash scripts/run_cpog_nested.sh
```

### 3. 直接调用Python脚本

```bash
# 查看帮助
python3 run_cpog_nested_cv.py --help

# 运行示例
python3 run_cpog_nested_cv.py \
    --study blca \
    --fold 0 \
    --split_file "/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/splits_0.csv"
```

---

## 🔍 验证清单

- [x] ✅ `run_cpog_nested_cv.py` Python语法检查通过
- [x] ✅ `scripts/run_cpog_nested.sh` Bash语法检查通过
- [x] ✅ `scripts/run_all_cpog.sh` Bash语法检查通过
- [x] ✅ argparse参数解析正确（--study, --fold, --split_file）
- [x] ✅ 导入NestedCVFeatureSelector成功
- [x] ✅ 划分文件解析函数正确（train_idx, val_idx, test_idx）
- [x] ✅ 文件路径构造正确（splits_${FOLD}.csv）
- [x] ✅ 硬编码数据路径正确
- [x] ✅ 循环调用逻辑正确（0-4折）
- [x] ✅ 错误处理和验证完善
- [x] ✅ 帮助信息显示正确

---

## 📁 输出文件

### 输入文件
```
/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_{study}/
├── splits_0.csv  (包含 train_idx, val_idx, test_idx 列)
├── splits_1.csv
├── splits_2.csv
├── splits_3.csv
└── splits_4.csv
```

### 输出文件
```
features/{study}/
├── fold_0_genes.csv  (筛选的基因列表)
├── fold_1_genes.csv
├── fold_2_genes.csv
├── fold_3_genes.csv
├── fold_4_genes.csv
└── summary.csv  (汇总统计)
```

---

## 🎉 重构总结

✅ **成功完成三个脚本的重构**

1. **统一化**：所有脚本使用统一的参数命名和错误处理
2. **模块化**：每个脚本职责单一，便于维护和测试
3. **健壮性**：完善的参数验证和错误提示
4. **可扩展性**：模块化设计易于添加新功能
5. **用户友好**：清晰的日志输出和帮助信息

**下一步**：确保外部划分文件就绪，然后运行 `bash scripts/run_all_cpog.sh {study}` 开始CPCG筛选！

---

## 相关文件

- **Python脚本**：`run_cpog_nested_cv.py`
- **Bash脚本**：`scripts/run_cpog_nested.sh`、`scripts/run_all_cpog.sh`
- **依赖模块**：`preprocessing.CPCG_algo.nested_cv_wrapper.NestedCVFeatureSelector`
- **输出目录**：`features/{study}/`

---

**重构完成时间**：2026-01-27
**版本**：v3.0 (彻底重构版)
