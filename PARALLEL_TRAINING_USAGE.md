# 并行训练脚本使用指南

## 快速开始

### 1. 运行并行训练 (推荐)
```bash
# 使用默认并发数 (4个任务)
bash train_all_folds.sh blca
```

### 2. 根据GPU显存调整并发数

#### 高显存GPU (≥16GB)
```bash
# 6个并发任务 (充分利用GPU)
MAX_JOBS=6 bash train_all_folds.sh blca
```

#### 中等显存GPU (8-12GB)
```bash
# 3个并发任务 (平衡性能与稳定性)
MAX_JOBS=3 bash train_all_folds.sh blca
```

#### 低显存GPU (<8GB)
```bash
# 2个并发任务 (避免OOM)
MAX_JOBS=2 bash train_all_folds.sh blca
```

## 监控训练进度

### 方法1: 查看终端输出
```bash
bash train_all_folds.sh blca
```
终端将显示简洁的状态信息：
```
🚀 启动 Fold 0 (当前并发: 0/4)
🚀 启动 Fold 1 (当前并发: 1/4)
🚀 启动 Fold 2 (当前并发: 2/4)
🚀 启动 Fold 3 (当前并发: 3/4)
🚀 启动 Fold 4 (当前并发: 4/4)

✅ Fold 0 完成 (PID: 12345, 耗时: 980s)
```

### 方法2: 实时查看日志文件
```bash
# 监控所有折的日志
tail -f results/nested_cv/blca/fold_*.log

# 监控特定折
tail -f results/nested_cv/blca/fold_0.log
```

### 方法3: 检查日志文件大小 (间接监控进度)
```bash
watch -n 5 'ls -lh results/nested_cv/blca/fold_*.log'
```

## 训练完成后查看结果

### 1. 查看汇总结果
```bash
cat results/nested_cv/blca/summary.csv
```

### 2. 查看各折详细日志
```bash
# 查看Fold 0的训练日志
cat results/nested_cv/blca/fold_0.log

# 搜索错误信息
grep -i "error\|exception\|failed" results/nested_cv/blca/fold_0.log
```

### 3. 查看性能指标
```bash
# 提取C-index结果
grep "val_cindex" results/nested_cv/blca/fold_*/summary.csv
```

## 故障排除

### 1. 部分折训练失败

**现象**: 某些折的日志显示错误信息

**解决方案**:
```bash
# 手动重新运行失败的折
python3 main.py \
    --study tcga_blca \
    --k_start 2 \
    --k_end 3 \
    --split_dir "splits/nested_cv/blca" \
    --results_dir "results/nested_cv/blca/fold_2" \
    --seed 44 \
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

### 2. GPU内存不足 (OOM)

**现象**: 日志显示 "CUDA out of memory" 或训练中断

**解决方案**:
```bash
# 减少并发数
MAX_JOBS=2 bash train_all_folds.sh blca

# 或进一步减少
MAX_JOBS=1 bash train_all_folds.sh blca
```

### 3. 训练速度慢

**可能原因及解决方案**:

- **I/O瓶颈**: 确保数据在SSD上
- **GPU利用率低**: 检查是否正确使用GPU (`nvidia-smi`)
- **批次大小**: 当前为1，可以考虑增加 (需要修改代码)
- **数据预处理**: 确保特征文件已正确生成

### 4. 数据加载失败

**现象**: "Training on 0 samples" 或类似错误

**解决方案**:
```bash
# 检查特征文件是否存在
ls -lh features/blca/fold_*_genes.csv

# 检查嵌套划分文件
ls -lh splits/nested_cv/blca/nested_splits_*.csv

# 重新生成CPCG特征
bash run_all_cpog.sh blca
```

## 性能优化建议

### 1. 监控GPU利用率
```bash
# 实时监控GPU
watch -n 1 nvidia-smi
```

### 2. 调整并发数策略
```bash
# 观察GPU显存使用
nvidia-smi

# 根据实际情况调整
# 如果显存使用率 < 50%: 增加MAX_JOBS
# 如果出现OOM: 减少MAX_JOBS
```

### 3. 使用高速存储
```bash
# 确保数据在SSD上
df -h
```

### 4. 预生成特征
```bash
# 在训练前确保CPCG特征已生成
bash run_all_cpog.sh blca
```

## 高级用法

### 1. 后台运行
```bash
# 放到后台运行，终端断开后继续执行
nohup bash train_all_folds.sh blca > train.log 2>&1 &

# 查看后台任务
jobs
```

### 2. 限制CPU使用
```bash
# 限制Python进程的CPU使用率 (可选)
taskset -c 0-7 bash train_all_folds.sh blca
```

### 3. 设置进程优先级
```bash
# 以较低优先级运行 (避免影响其他进程)
nice -n 10 bash train_all_folds.sh blca
```

## 输出文件说明

```
results/nested_cv/{study}/
├── fold_0/
│   ├── summary.csv          # Fold 0 结果汇总
│   └── ...                  # 其他训练输出
├── fold_1/
│   └── ...
├── ...
├── fold_4/
│   └── ...
├── fold_0.log               # Fold 0 完整日志
├── fold_1.log               # Fold 1 完整日志
├── ...
├── fold_4.log               # Fold 4 完整日志
└── summary.csv              # 所有折的汇总结果
```

## 常见问题FAQ

**Q: 为什么选择MAX_JOBS=4？**
A: 基于22% GPU利用率观察，4个并发任务可以在大多数8GB GPU上稳定运行，同时获得良好的加速比。

**Q: 可以设置MAX_JOBS大于GPU数量吗？**
A: 可以，但GPU数量决定了实际并行度。更多任务会在队列中等待。

**Q: 训练过程中可以中断吗？**
A: 可以使用Ctrl+C中断，但建议等待当前启动的任务完成，或手动kill进程。

**Q: 如何恢复中断的训练？**
A: 目前不支持自动恢复，需要手动重新运行失败的折。

**Q: 并行训练会影响模型性能吗？**
A: 不会。训练是独立的并行执行，最终结果与串行训练一致。

**Q: 为什么终端输出这么少？**
A: 为了减少I/O开销，所有详细日志都写入文件。终端只显示关键状态信息。
