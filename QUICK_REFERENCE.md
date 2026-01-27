# 快速参考卡

## 常用命令

### 1. 数据预处理
```bash
# 创建嵌套CV划分
bash create_nested_splits.sh blca

# 运行CPCG特征筛选 (所有折)
bash run_all_cpog.sh blca

# 或使用优化版本
bash run_cpog_optimized.sh blca
```

### 2. 模型训练
```bash
# 并行训练 (默认4并发)
bash train_all_folds.sh blca

# 自定义并发数
MAX_JOBS=2 bash train_all_folds.sh blca
MAX_JOBS=6 bash train_all_folds.sh blca

# 单折训练
python3 main.py \
    --study tcga_blca \
    --k_start 0 \
    --k_end 1 \
    --split_dir "splits/nested_cv/blca" \
    --results_dir "results/nested_cv/blca/fold_0" \
    --seed 42 \
    --label_file datasets_csv/clinical_data/tcga_blca_clinical.csv \
    --task survival \
    --n_classes 4 \
    --modality snn \
    --omics_dir "datasets_csv/raw_rna_data/combine/blca" \
    --data_root_dir "data/blca/pt_files" \
    --label_col survival_months \
    --type_of_path combine \
    --max_epochs 20 \
    --lr 0.00005 \
    --opt adam \
    --reg 0.00001 \
    --alpha_surv 0.5 \
    --weighted_sample \
    --batch_size 1 \
    --bag_loss nll_surv \
    --encoding_dim 256 \
    --num_patches 4096 \
    --wsi_projection_dim 256 \
    --encoding_layer_1_dim 8 \
    --encoding_layer_2_dim 16 \
    --encoder_dropout 0.25
```

### 3. 监控与调试
```bash
# 检查GPU使用情况
watch -n 1 nvidia-smi

# 实时查看训练日志
tail -f results/nested_cv/blca/fold_0.log

# 查看所有折的日志
tail -f results/nested_cv/blca/fold_*.log

# 检查特征文件
ls -lh features/blca/fold_*_genes.csv

# 检查训练进度 (通过文件大小)
watch -n 5 'ls -lh results/nested_cv/blca/fold_*.log'

# 查看汇总结果
cat results/nested_cv/blca/summary.csv

# 搜索错误信息
grep -i "error\|exception\|failed" results/nested_cv/blca/fold_0.log
```

## 关键文件路径

### 输入文件
```
datasets_csv/
└── clinical_data/
    └── tcga_{study}_clinical.csv        # 临床数据

splits/
└── nested_cv/
    └── {study}/
        └── nested_splits_{fold}.csv     # 嵌套CV划分

features/
└── {study}/
    └── fold_{fold}_genes.csv            # CPCG特征文件
```

### 输出文件
```
results/nested_cv/{study}/
├── fold_{fold}/
│   ├── summary.csv                      # 单折结果
│   └── ...                              # 其他输出
├── fold_{fold}.log                      # 完整训练日志
└── summary.csv                           # 所有折汇总
```

## 参数说明

### 训练参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_epochs | 20 | 最大训练轮数 |
| lr | 0.00005 | 学习率 |
| batch_size | 1 | 批次大小 |
| encoding_dim | 256 | 编码维度 |
| num_patches | 4096 | 图像块数量 |

### CPCG参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| threshold | 100 | 筛选基因数量 |
| n_jobs | -1 | 并行作业数 (-1=所有核心) |
| p_value | 0.01 | 显著性阈值 |

### 并行参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| MAX_JOBS | 4 | 最大并发训练任务数 |

## 故障排除

### 错误: "Training on 0 samples"
**原因**: ID格式不匹配
**解决**:
```bash
# 检查ID格式
head -5 datasets_csv/clinical_data/tcga_blca_clinical.csv
head -5 splits/nested_cv/blca/nested_splits_0.csv
```

### 错误: "CUDA out of memory"
**原因**: GPU显存不足
**解决**:
```bash
# 减少并发数
MAX_JOBS=2 bash train_all_folds.sh blca

# 或使用单折训练
python3 main.py --k_start 0 --k_end 1 ...
```

### 错误: 特征文件缺失
**原因**: CPCG未运行或失败
**解决**:
```bash
# 重新生成特征
bash run_all_cpog.sh blca

# 检查特征文件
ls -lh features/blca/fold_*_genes.csv
```

### 训练速度慢
**检查项**:
```bash
# 1. 检查GPU利用率
nvidia-smi

# 2. 检查CPU利用率
top

# 3. 检查磁盘I/O
iotop

# 4. 确认数据在SSD上
df -h
```

## 性能调优

### GPU显存优化
```bash
# 显存 < 8GB
MAX_JOBS=2

# 显存 8-16GB
MAX_JOBS=4

# 显存 > 16GB
MAX_JOBS=6
```

### CPU优化
```bash
# 使用所有CPU核心 (CPCG筛选)
python3 -c "import os; print(os.cpu_count())"

# 在脚本中设置
export OMP_NUM_THREADS=8
```

### I/O优化
```bash
# 确保数据在SSD上
# 检查路径
pwd
df -h

# 如果数据在机械硬盘，考虑迁移到SSD
```

## 快捷脚本

### 快速开始
```bash
# 一键运行 (预处理 + 训练)
bash create_nested_splits.sh blca && \
bash run_all_cpog.sh blca && \
bash train_all_folds.sh blca
```

### 检查状态
```bash
# 检查所有关键文件
bash -c '
echo "=== 临床文件 ===" && \
ls datasets_csv/clinical_data/tcga_*_clinical.csv && \
echo "=== 划分文件 ===" && \
ls splits/nested_cv/*/nested_splits_*.csv && \
echo "=== 特征文件 ===" && \
ls features/*/fold_*_genes.csv && \
echo "=== 结果文件 ===" && \
ls results/nested_cv/*/fold_*/summary.csv 2>/dev/null || echo "无结果文件"
'
```

### 清理临时文件
```bash
# 清理训练日志 (保留汇总)
find results/nested_cv -name "*.log" -type f

# 清理临时特征文件
rm -rf /tmp/cpog_*

# 清理Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## 配置参考

### 环境变量
```bash
# Python路径
export PYTHONPATH=/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy:$PYTHONPATH

# CUDA设置
export CUDA_VISIBLE_DEVICES=0

# OpenMP线程数
export OMP_NUM_THREADS=8
```

### Python依赖
```
torch>=1.9.0
torchvision
pandas
numpy
scikit-learn
joblib
tqdm
lifelines
pingouin
statsmodels
scipy
```

## 常用调试命令

### Python调试
```python
# 检查数据加载
python3 -c "
import pandas as pd
df = pd.read_csv('datasets_csv/clinical_data/tcga_blca_clinical.csv')
print('Clinical data shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Sample IDs:', df['case_id'][:5].tolist())
"

# 检查特征文件
python3 -c "
import pandas as pd
df = pd.read_csv('features/blca/fold_0_genes.csv')
print('Feature file shape:', df.shape)
print('Genes:', df['gene'][:10].tolist())
"
```

### Bash调试
```bash
# 检查脚本语法
bash -n scripts/train_all_folds.sh

# 查看详细错误
bash -x scripts/train_all_folds.sh blca

# 测试Python脚本
python3 preprocessing/CPCG_algo/nested_cv_wrapper.py --help
```

## 联系与支持

### 相关文档
- `CPCG_OPTIMIZATION_REPORT.md` - CPCG优化详情
- `PARALLEL_TRAINING_OPTIMIZATION.md` - 并行训练优化
- `PROJECT_OPTIMIZATION_SUMMARY.md` - 项目总结

### 日志位置
- 训练日志: `results/nested_cv/{study}/fold_{fold}.log`
- CPCG日志: `features/{study}/cpog.log` (如果启用)

---
**更新时间**: 2026-01-27
**版本**: v2.0 (并行优化版)
