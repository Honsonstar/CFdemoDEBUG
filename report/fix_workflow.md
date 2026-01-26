# 数据泄露修复工作流程

## 阶段0: 问题诊断

### 0.1 确认泄露问题
```bash
# 检查CPCG输出文件的样本分布
python -c "
import pandas as pd
import numpy as np

# 读取原始数据
clinical = pd.read_csv('datasets_csv/clinical_data/tcga_blca_clinical.csv')
cpog = pd.read_csv('preprocessing/CPCG_algo/raw_data/finalstage_result_/tcga_blca/tcga_blca_M2M3base_0916.csv')

print('原始临床数据样本数:', len(clinical))
print('CPCG输出样本数:', len(cpog))
print('CPCG基因数:', len(cpog.columns) - 2)  # 减去OS和索引列
print('是否包含测试集:', 'TCGA-BL-A13J' in cpog.iloc[:, 0].values)
"
```

### 0.2 检查splits文件
```bash
# 查看splits文件样本分布
python -c "
import pandas as pd

for fold in range(5):
    splits = pd.read_csv(f'splits/5foldcv/tcga_blca/splits_{fold}.csv')
    print(f'Fold {fold}:')
    print(f'  Train: {len(splits)}')
    print(f'  Val: {splits[\"val\"].notna().sum()}')
    print(f'  Test: {splits[\"test\"].notna().sum()}')
"
```

### 0.3 分析代码调用链
```python
# 确认数据加载流程
# main.py:284 → SurvivalDatasetFactory → _setup_omics_data
# 读取omics_dir参数指定的文件，该文件包含预筛选特征
```

## 阶段1: 方案设计

### 1.1 嵌套交叉验证架构设计

```
外层循环 (Outer Loop): 5折CV - 评估泛化性能
├─ Fold 1
│  ├─ 内层循环 (Inner Loop): 训练/验证划分
│  │  ├─ Train (70%)
│  │  ├─ Validation (15%)
│  │  └─ Test (15%)
│  ├─ 在Train上运行CPCG → 筛选基因
│  ├─ 在Validation上调参
│  └─ 在Test上评估 ← 记录结果
├─ Fold 2-5 (重复上述流程)
└─ 汇总5个Test结果 → 平均性能
```

### 1.2 三种修复方案

#### 方案A: 完全嵌套CV (推荐)
- 每折独立运行CPCG特征筛选
- 性能最可靠，但计算成本高5倍
- 适合: 最终验证、发表论文

#### 方案B: 外部签名替换 (快速)
- 使用Hallmarks/Reactome签名
- 无泄露，但可能性能下降
- 适合: 快速验证、原型开发

#### 方案C: 保守筛选 + 独立测试集
- 训练集上筛选 → 验证集调参 → 独立测试集评估
- 成本中等，可信度较高
- 适合: 资源受限的场景

### 1.3 选择方案A (完全嵌套CV)

原因:
- 彻底解决数据泄露
- 符合最佳实践
- 可发表、可复现
- 虽然计算成本高，但结果可信

### 1.4 技术实现要点

```python
# 关键修改点
1. SurvivalDatasetFactory.__init__()
   - 移除全局特征筛选
   - 添加按折特征筛选接口

2. 数据加载流程
   - 训练前动态加载训练集特征
   - 验证/测试使用相同特征集合

3. 训练流程
   - 每折开始前运行CPCG
   - 筛选特征保存到临时文件
   - 训练结束后清理

4. 特征筛选代码
   - 接受case_id列表作为输入
   - 仅使用指定样本进行筛选
```

### 1.5 文件结构规划

```
项目根目录/
├── preprocessing/
│   ├── CPCG_algo/
│   │   ├── nested_cv_wrapper.py  ← 新增: 嵌套CV包装器
│   │   └── utils/
│   │       ├── feature_selector.py ← 新增: 特征选择工具
│   │       └── split_data.py      ← 新增: 数据划分工具
├── datasets/
│   └── dataset_survival.py       ← 修改: 支持动态特征
├── utils/
│   ├── core_utils.py              ← 修改: 训练流程
│   └── nested_cv.py              ← 新增: 嵌套CV控制
└── scripts/
    ├── fix_nested_cv.sh          ← 新增: 快速运行脚本
    └── compare_methods.py         ← 新增: 对比实验
```
