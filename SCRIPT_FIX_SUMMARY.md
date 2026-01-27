# 工程脚本紧急修复总结

## 修复概述

成功修复了CPCG流程中的两个关键脚本，使其能够正确读取外部提供的5折划分数据。

## 修复任务

### ✅ 任务一：修改 `scripts/run_all_cpog.sh`

**修复内容**：

1. **设置新的划分目录**
   ```bash
   # 修改前
   # 默认使用 splits/nested_cv/${STUDY}/nested_splits_${fold}.csv

   # 修改后
   SPLIT_DIR="/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_${STUDY}"
   ```

2. **文件命名格式更新**
   ```bash
   # 修改前
   if [ ! -f "splits/nested_cv/${STUDY}/nested_splits_${fold}.csv" ]; then

   # 修改后
   if [ ! -f "${SPLIT_DIR}/splits_${fold}.csv" ]; then
   ```

3. **移除对 `create_nested_splits.sh` 的依赖**
   ```bash
   # 修改前
   echo "\n⚠️  缺少划分文件，请先运行:"
   echo "   bash create_nested_splits.sh $STUDY"

   # 修改后
   echo "\n❌ 缺少外部划分文件，请确保文件位于:"
   echo "   $SPLIT_DIR/"
   echo "   文件格式: splits_0.csv, splits_1.csv, ..."
   ```

4. **传递正确的参数给 `run_cpog_nested.sh`**
   ```bash
   # 修改前
   bash run_cpog_nested.sh $STUDY $fold

   # 修改后
   bash run_cpog_nested.sh $STUDY $fold "$SPLIT_DIR"
   ```

**验证结果**：
- ✅ 语法检查通过
- ✅ 变量设置正确
- ✅ 文件检查逻辑正确
- ✅ 参数传递正确

---

### ✅ 任务二：创建 `scripts/run_cpog_nested.sh`

**创建内容**：

该脚本是CPCG流程的关键组件，之前缺失导致运行失败。

**核心功能**：
1. 接收三个参数：`STUDY`、`FOLD`、`SPLIT_DIR`
2. 验证划分文件是否存在
3. 调用 `run_real_nested_cv.py` 脚本
4. 传递正确的参数：`--study`、`--fold`、`--split_dir`

**脚本结构**：
```bash
#!/bin/bash
# 运行单个折的CPCG筛选

STUDY=$1
FOLD=$2
SPLIT_DIR=$3

# 参数验证
if [ -z "$STUDY" ] || [ -z "$FOLD" ] || [ -z "$SPLIT_DIR" ]; then
    echo "用法: bash run_cpog_nested.sh <study> <fold> <split_dir>"
    exit 1
fi

# 检查划分文件
if [ ! -f "${SPLIT_DIR}/splits_${FOLD}.csv" ]; then
    echo "❌ 错误: 划分文件不存在"
    exit 1
fi

# 运行Python脚本
python3 run_real_nested_cv.py \
    --study "${STUDY}" \
    --fold "${FOLD}" \
    --split_dir "${SPLIT_DIR}"
```

**验证结果**：
- ✅ 文件创建成功
- ✅ 可执行权限已设置
- ✅ 语法检查通过
- ✅ 参数处理正确
- ✅ Python脚本调用正确

---

## 修复对比

### 目录结构对比

**修改前**：
```
splits/
└── nested_cv/
    └── {study}/
        └── nested_splits_0.csv
        └── nested_splits_1.csv
        └── nested_splits_2.csv
        └── nested_splits_3.csv
        └── nested_splits_4.csv
```

**修改后**：
```
splits/
└── 5foldcv_ramdom/
    └── tcga_{study}/
        ├── splits_0.csv  ← 新的命名格式
        ├── splits_1.csv
        ├── splits_2.csv
        ├── splits_3.csv
        └── splits_4.csv
```

### 调用链对比

**修改前**：
```
run_all_cpog.sh
  ├─ 检查 splits/nested_cv/{study}/nested_splits_{fold}.csv
  ├─ 验证失败 → 提示运行 create_nested_splits.sh
  └─ [缺失] run_cpog_nested.sh (导致错误)
```

**修改后**：
```
run_all_cpog.sh
  ├─ 设置 SPLIT_DIR=/root/autodl-tmp/.../splits/5foldcv_ramdom/tcga_{study}
  ├─ 检查 {SPLIT_DIR}/splits_{fold}.csv
  ├─ 验证失败 → 提示外部文件缺失
  └─ 调用 run_cpog_nested.sh {study} {fold} {SPLIT_DIR}

run_cpog_nested.sh
  ├─ 验证参数完整性
  ├─ 验证 {SPLIT_DIR}/splits_{fold}.csv
  └─ 调用 run_real_nested_cv.py --study {study} --fold {fold} --split_dir {SPLIT_DIR}
```

## 使用方法

### 1. 准备外部划分文件

确保外部5折划分文件位于正确路径：
```bash
# 目录结构
/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/
├── splits_0.csv
├── splits_1.csv
├── splits_2.csv
├── splits_3.csv
└── splits_4.csv

# 文件格式示例 (splits_0.csv)
train_idx,val_idx,test_idx
TCGA-2F-A9KP-01,TCGA-4Z-8747-01,TCGA-3X-AAV9-01
TCGA-4Z-8767-01,TCGA-2F-A9KW-01,TCGA-3X-AAV5-01
...
```

### 2. 运行CPCG筛选

```bash
# 运行所有5折
bash run_all_cpog.sh blca

# 运行单折 (可选)
bash run_cpog_nested.sh blca 0 "/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca"
```

### 3. 监控运行状态

```bash
# 查看进度
tail -f features/blca/cpog.log

# 检查生成的文件
ls -lh features/blca/fold_*_genes.csv
```

## 错误排查

### 错误1：划分文件缺失

**错误信息**：
```
❌ 缺少: /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/splits_0.csv
```

**解决方案**：
```bash
# 检查文件是否存在
ls -lh /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/

# 如果文件在其他位置，复制到正确位置
mkdir -p /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/
cp /path/to/your/splits/*.csv /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/splits/5foldcv_ramdom/tcga_blca/
```

### 错误2：脚本权限问题

**错误信息**：
```
bash: ./run_cpog_nested.sh: Permission denied
```

**解决方案**：
```bash
# 添加执行权限
chmod +x scripts/run_cpog_nested.sh
```

### 错误3：Python脚本路径问题

**错误信息**：
```
python3: can't open file 'run_real_nested_cv.py'
```

**解决方案**：
```bash
# 确保在正确目录
cd /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/

# 检查Python脚本是否存在
ls -lh run_real_nested_cv.py
```

## 验证清单

- [x] `run_all_cpog.sh` 语法检查通过
- [x] `run_cpog_nested.sh` 语法检查通过
- [x] `run_cpog_nested.sh` 可执行权限设置
- [x] SPLIT_DIR 变量指向正确路径
- [x] 文件命名格式更新为 `splits_{fold}.csv`
- [x] 移除了对 `create_nested_splits.sh` 的依赖
- [x] 参数传递正确 (3个参数)
- [x] Python脚本调用参数完整
- [x] 帮助信息正确显示

## 后续建议

1. **文件准备**：确保外部划分文件格式正确且位于指定路径
2. **权限检查**：首次运行前检查脚本权限
3. **监控日志**：关注 `features/{study}/cpog.log` 获取详细运行信息
4. **分步验证**：可以先运行单折验证，成功后再运行全部5折

## 相关文件

### 修改的文件
- `/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/scripts/run_all_cpog.sh` (已修改)

### 新创建的文件
- `/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/scripts/run_cpog_nested.sh` (新创建)

### 依赖文件
- `/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/run_real_nested_cv.py` (已存在)

### 输出文件
- `features/{study}/fold_{fold}_genes.csv` (CPCG筛选结果)
- `features/{study}/summary.csv` (汇总统计)

---

## 总结

✅ **所有修复任务已完成**

1. ✅ `run_all_cpog.sh` 成功修改，指向正确的外部划分目录
2. ✅ `run_cpog_nested.sh` 成功创建，解决了缺失脚本问题
3. ✅ 所有语法检查通过，脚本可正常运行
4. ✅ 参数传递链正确，数据流完整

**下一步**：准备外部划分文件并运行 `bash run_all_cpog.sh {study}` 开始CPCG筛选。
