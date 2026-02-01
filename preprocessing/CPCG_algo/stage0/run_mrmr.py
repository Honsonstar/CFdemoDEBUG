"""
mRMR 特征选择脚本 - 用于嵌套交叉验证

使用 mRMR (minimum Redundancy Maximum Relevance) 算法替代 CPCG 流程进行基因特征筛选。
本脚本独立于 Stage1/Stage2，可直接运行。

用法：
    # 方式1: 直接指定 split_file
    python run_mrmr.py --study brca --fold 0 \\
        --split_file ../../../splits/5foldcv/tcga_brca/splits_0.csv \\
        --data_root_dir ../../../datasets_csv/raw_rna_data/combine \\
        --clinical_dir /root/autodl-tmp/newcfdemo/CFdemo_gene_text/datasets_csv/clinical_data

    # 方式2: 使用 nested_cv 目录格式 (自动构建路径)
    python run_mrmr.py --study brca --fold 0 \\
        --split_dir ../../../splits/nested_cv \\
        --data_root_dir ../../../datasets_csv/raw_rna_data/combine \\
        --clinical_dir /root/autodl-tmp/newcfdemo/CFdemo_gene_text/datasets_csv/clinical_data

    # 方式3: 批量运行所有 folds
    python run_mrmr.py --study brca --fold all \\
        --split_dir ../../../splits/nested_cv \\
        --data_root_dir ../../../datasets_csv/raw_rna_data/combine
"""

import os
import pandas as pd
import numpy as np
from mrmr import mrmr_classif


class MRMRFeatureSelector:
    """基于 mRMR 的特征选择器，用于嵌套交叉验证"""
    
    # 输出根目录（绝对路径）
    OUTPUT_ROOT = '/root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/features'
    
    def __init__(self, study, data_root_dir, clinical_dir=None, threshold=200, n_jobs=-1):
        """
        初始化 mRMR 特征选择器
        
        Args:
            study: 癌症类型 (e.g., 'brca', 'luad')
            data_root_dir: 数据根目录，包含 {study}/rna_clean.csv
            clinical_dir: 临床数据目录，包含 tcga_{study}_clinical.csv
            threshold: 选择的特征数量 K (默认: 200)
            n_jobs: 并行任务数 (-1 表示使用所有 CPU)
        """
        self.study = study
        self.data_root_dir = data_root_dir
        self.clinical_dir = clinical_dir
        self.threshold = threshold
        self.n_jobs = n_jobs
        
    def select_features_for_fold(self, fold, train_ids, val_ids, test_ids):
        """
        为指定的 fold 执行 mRMR 特征选择
        
        Args:
            fold: fold 编号
            train_ids: 训练集样本 ID 列表
            val_ids: 验证集样本 ID 列表
            test_ids: 测试集样本 ID 列表
            
        Returns:
            输出文件路径
        """
        # 1. ID 截断对齐（TCGA ID 只取前 12 位）
        train_ids = [str(i)[:12] for i in train_ids if str(i) != 'nan']
        val_ids = [str(i)[:12] for i in val_ids if str(i) != 'nan']
        test_ids = [str(i)[:12] for i in test_ids if str(i) != 'nan']
        
        print(f"\n[{self.study}] Fold {fold}: 开始 mRMR 特征筛选 (Train={len(train_ids)})")
        
        # 2. 读取表达数据
        exp_data = self._load_expression_data()
        
        # 3. 读取临床数据
        clinical_data = self._load_clinical_data()
        
        # 4. 准备训练数据
        X_train, y_train = self._prepare_training_data(
            exp_data, clinical_data, train_ids
        )
        
        if X_train is None or len(X_train) < 10:
            print("  ⚠️ Warning: 训练样本数不足，无法执行特征选择")
            return self._generate_empty_feature_file(fold)
        
        # 5. 执行 mRMR 特征选择
        selected_genes = self._run_mrmr(X_train, y_train)
        
        # 6. 生成特征文件
        all_ids = train_ids + val_ids + test_ids
        return self._generate_feature_file(fold, selected_genes, exp_data, all_ids)
    
    def _load_expression_data(self):
        """加载基因表达数据"""
        exp_file = os.path.join(self.data_root_dir, self.study, 'rna_clean.csv')
        if not os.path.exists(exp_file):
            exp_file = os.path.join(self.data_root_dir, self.study, 'data.csv')
        if not os.path.exists(exp_file):
            raise FileNotFoundError(f"找不到表达数据文件: {exp_file}")
        
        # 读取数据，第一列为样本 ID（索引）
        exp_data = pd.read_csv(exp_file, index_col=0)
        # 截断样本 ID 为 12 位
        exp_data.index = [str(i)[:12] for i in exp_data.index]
        
        print(f"  [数据] 表达矩阵: {exp_data.shape[0]} 样本 x {exp_data.shape[1]} 基因")
        return exp_data
    
    def _load_clinical_data(self):
        """加载临床数据"""
        clinical_file = None
        
        # 优先使用指定的 clinical_dir
        if self.clinical_dir:
            # 尝试两种文件名格式
            file_patterns = [
                os.path.join(self.clinical_dir, f'tcga_{self.study}', 'clinical.CSV'),
                os.path.join(self.clinical_dir, f'tcga_{self.study}_clinical.csv'),
            ]
            for pattern in file_patterns:
                if os.path.exists(pattern):
                    clinical_file = pattern
                    break
        else:
            # 默认路径（兼容 CPCG 目录结构）
            # 尝试多个可能的路径
            default_paths = [
                f'../raw_data/tcga_{self.study}/clinical.CSV',  # CPCG 格式
                f'datasets_csv/clinical_data/tcga_{self.study}_clinical.csv',  # 旧格式
            ]
            for path in default_paths:
                if os.path.exists(path):
                    clinical_file = path
                    break
        
        if clinical_file is None or not os.path.exists(clinical_file):
            raise FileNotFoundError(
                f"找不到临床数据文件。尝试过的路径:\n"
                f"  - ../raw_data/tcga_{self.study}/clinical.CSV\n"
                f"  - datasets_csv/clinical_data/tcga_{self.study}_clinical.csv\n"
                f"请使用 --clinical_dir 参数指定临床数据目录"
            )
        
        clinical_data = pd.read_csv(clinical_file)
        
        # 识别样本 ID 列（兼容多种列名）
        id_col = None
        for col in ['case_id', 'case_submitter_id', 'CASE_ID', 'patient_id', 'PATIENT_ID']:
            if col in clinical_data.columns:
                id_col = col
                break
        
        if id_col is None:
            # 如果没有明确的 ID 列，使用第一列
            id_col = clinical_data.columns[0]
            print(f"  [警告] 未找到标准 ID 列，使用第一列作为 ID: {id_col}")
        
        # 添加截断的 ID 列用于匹配
        clinical_data['case_id_truncated'] = clinical_data[id_col].astype(str).str[:12]
        
        print(f"  [数据] 临床数据: {len(clinical_data)} 样本 (ID列: {id_col}, 文件: {clinical_file})")
        return clinical_data
    
    def _prepare_training_data(self, exp_data, clinical_data, train_ids):
        """
        准备训练数据：对齐表达数据和临床数据
        
        Args:
            exp_data: 表达矩阵 (行=样本, 列=基因)
            clinical_data: 临床数据
            train_ids: 训练集样本 ID
            
        Returns:
            X_train: 特征矩阵 (pandas DataFrame)
            y_train: 目标变量 (pandas Series)
        """
        # 筛选训练集的临床数据
        train_mask = clinical_data['case_id_truncated'].isin(train_ids)
        train_clinical = clinical_data[train_mask].copy()
        
        print(f"  [匹配] 输入 Train IDs={len(train_ids)} -> 匹配到临床样本={len(train_clinical)}")
        
        if len(train_clinical) < 10:
            print("  ⚠️ Warning: 匹配到的训练样本太少，请检查 ID 格式")
            return None, None
        
        # 准备目标变量 y: 使用 censorship 或 OS_STATUS 作为分类目标
        # censorship: 0=死亡, 1=存活（删失）
        if 'censorship' in train_clinical.columns:
            y_col = 'censorship'
        elif 'OS_STATUS' in train_clinical.columns:
            y_col = 'OS_STATUS'
        elif 'Censor' in train_clinical.columns:
            y_col = 'Censor'
        else:
            raise ValueError("临床数据中找不到生存状态列 (censorship/OS_STATUS/Censor)")
        
        # 创建 ID 到 y 值的映射
        id_to_y = dict(zip(
            train_clinical['case_id_truncated'],
            train_clinical[y_col]
        ))
        
        # 筛选在表达数据中存在的训练样本
        valid_train_ids = [i for i in train_ids if i in exp_data.index and i in id_to_y]
        
        if len(valid_train_ids) < 10:
            print(f"  ⚠️ Warning: 有效训练样本太少 ({len(valid_train_ids)})")
            return None, None
        
        # 提取特征矩阵 X 和目标变量 y
        X_train = exp_data.loc[valid_train_ids].copy()
        y_train = pd.Series([id_to_y[i] for i in valid_train_ids], index=valid_train_ids)
        
        # 去除常量列（方差为 0 的基因）
        var = X_train.var()
        constant_cols = var[var == 0].index.tolist()
        if constant_cols:
            print(f"  [过滤] 移除 {len(constant_cols)} 个常量基因")
            X_train = X_train.drop(columns=constant_cols)
        
        print(f"  [准备] 训练数据: {X_train.shape[0]} 样本 x {X_train.shape[1]} 特征")
        print(f"  [目标] {y_col}: {y_train.value_counts().to_dict()}")
        
        return X_train, y_train
    
    def _run_mrmr(self, X, y):
        """
        执行 mRMR 特征选择
        
        Args:
            X: 特征矩阵 (pandas DataFrame, 行=样本, 列=基因)
            y: 目标变量 (pandas Series)
            
        Returns:
            selected_genes: 选中的基因列表
        """
        # 确定要选择的特征数量（不超过可用特征数）
        K = min(self.threshold, X.shape[1])
        
        print(f"  [mRMR] 开始选择 Top {K} 个特征...")
        print(f"         输入: {X.shape[0]} 样本 x {X.shape[1]} 特征")
        
        # 执行 mRMR 分类特征选择
        # mrmr_classif 要求 y 是分类变量
        selected_genes = mrmr_classif(
            X=X,
            y=y,
            K=K,
            relevance='f',      # F统计量
            redundancy='c',     # 皮尔逊相关系数
            n_jobs=self.n_jobs,
            show_progress=True
        )
        
        print(f"  [mRMR] 完成! 选择了 {len(selected_genes)} 个特征")
        
        return selected_genes
    
    def _generate_feature_file(self, fold, genes, exp_data, all_ids):
        """
        生成特征文件
        
        Args:
            fold: fold 编号
            genes: 选中的基因列表
            exp_data: 完整表达数据 (行=样本, 列=基因)
            all_ids: 所有样本 ID (train + val + test)
            
        Returns:
            输出文件路径
        """
        # 输出目录: /root/autodl-tmp/newcfdemo/CFdemo_gene_text_copy/features/mrmr_{study}/
        output_dir = os.path.join(self.OUTPUT_ROOT, f'mrmr_{self.study}')
        os.makedirs(output_dir, exist_ok=True)
        
        out_file = os.path.join(output_dir, f'fold_{fold}_genes.csv')
        
        # 筛选有效的样本 ID（存在于表达数据中）
        valid_ids = [i for i in all_ids if i in exp_data.index]
        
        if not genes:
            print(f"  [Warning] 没有选中任何基因，创建空文件")
            pd.DataFrame(columns=['gene_name'] + valid_ids).to_csv(out_file, index=False)
        else:
            # 筛选有效的基因（存在于表达数据中）
            valid_genes = [g for g in genes if g in exp_data.columns]
            
            # 构建输出 DataFrame：行=基因，列=样本
            # 先转置表达数据，使得行=基因，列=样本
            exp_subset = exp_data.loc[valid_ids, valid_genes].T
            exp_subset.index.name = 'gene_name'
            exp_subset = exp_subset.reset_index()
            
            exp_subset.to_csv(out_file, index=False)
        
        print(f"  [保存] {out_file}")
        print(f"         基因数: {len(genes)}, 样本数: {len(valid_ids)}")
        
        return out_file
    
    def _generate_empty_feature_file(self, fold):
        """生成空的特征文件（当数据不足时）"""
        output_dir = os.path.join(self.OUTPUT_ROOT, f'mrmr_{self.study}')
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'fold_{fold}_genes.csv')
        
        pd.DataFrame(columns=['gene_name']).to_csv(out_file, index=False)
        print(f"  [保存] {out_file} (空文件)")
        
        return out_file


def main():
    """主函数：命令行入口"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description='mRMR Feature Selection for Nested CV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用 split_file 直接指定划分文件
    python run_mrmr.py --study brca --fold 0 --split_file splits.csv --data_root_dir /path/to/data
    
    # 使用 nested_cv 目录格式 (推荐)
    python run_mrmr.py --study brca --fold 0 --split_dir /path/to/splits/nested_cv --data_root_dir /path/to/data
    
    # 批量运行所有 folds
    python run_mrmr.py --study brca --fold all --split_dir /path/to/splits/nested_cv --data_root_dir /path/to/data
        """
    )
    parser.add_argument('--study', type=str, required=True,
                        help='癌症类型 (e.g., brca, luad, stad)')
    parser.add_argument('--fold', type=str, required=True,
                        help='Fold 编号 (0-4) 或 "all" 运行所有 folds')
    parser.add_argument('--split_file', type=str, default=None,
                        help='数据划分 CSV 文件路径 (与 --split_dir 二选一)')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='nested_cv 目录路径，会自动构建 {split_dir}/{study}/nested_splits_{fold}.csv')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='数据根目录，包含 {study}/rna_clean.csv')
    parser.add_argument('--clinical_dir', type=str, default=None,
                        help='临床数据目录，包含 tcga_{study}_clinical.csv')
    parser.add_argument('--threshold', type=int, default=200,
                        help='选择的特征数量 K (默认: 200)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='并行任务数 (-1 表示使用所有 CPU)')
    
    args = parser.parse_args()
    
    # 参数校验：split_file 和 split_dir 必须二选一
    if args.split_file is None and args.split_dir is None:
        parser.error("必须指定 --split_file 或 --split_dir 其中之一")
    if args.split_file is not None and args.split_dir is not None:
        parser.error("--split_file 和 --split_dir 不能同时指定")
    
    # 确定要运行的 folds
    if args.fold.lower() == 'all':
        if args.split_dir is None:
            parser.error("使用 --fold all 时必须指定 --split_dir")
        # 自动发现所有 nested_splits_*.csv 文件
        pattern = os.path.join(args.split_dir, args.study, 'nested_splits_*.csv')
        split_files = sorted(glob.glob(pattern))
        if not split_files:
            raise FileNotFoundError(f"在 {args.split_dir}/{args.study}/ 下找不到 nested_splits_*.csv 文件")
        folds = [int(os.path.basename(f).replace('nested_splits_', '').replace('.csv', '')) for f in split_files]
        print(f"[Info] 发现 {len(folds)} 个 folds: {folds}")
    else:
        folds = [int(args.fold)]
    
    # 创建特征选择器
    selector = MRMRFeatureSelector(
        study=args.study,
        data_root_dir=args.data_root_dir,
        clinical_dir=args.clinical_dir,
        threshold=args.threshold,
        n_jobs=args.n_jobs
    )
    
    output_files = []
    for fold in folds:
        # 确定划分文件路径
        if args.split_file is not None:
            split_file = args.split_file
        else:
            # nested_cv 格式: {split_dir}/{study}/nested_splits_{fold}.csv
            split_file = os.path.join(args.split_dir, args.study, f'nested_splits_{fold}.csv')
        
        if not os.path.exists(split_file):
            print(f"[Warning] 划分文件不存在: {split_file}, 跳过 fold {fold}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[Fold {fold}] 处理中...")
        print(f"         划分文件: {split_file}")
        print(f"{'='*60}")
        
        # 读取划分文件
        split_df = pd.read_csv(split_file)
        
        # 检测并解析划分文件格式
        train_ids, val_ids, test_ids = parse_split_file(split_df)
        
        print(f"[Info] 划分信息: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
        
        # 运行 mRMR 特征选择
        output_file = selector.select_features_for_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids
        )
        output_files.append(output_file)
        
        print(f"\n✅ Fold {fold} 完成!")
        print(f"   输出文件: {output_file}")
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"全部完成! 共处理 {len(output_files)} 个 folds")
    for f in output_files:
        print(f"  - {f}")
    print(f"{'='*60}")


def parse_split_file(split_df):
    """
    解析数据划分文件，支持多种格式
    
    格式1: train_ids, val_ids, test_ids (每行一个 fold 的 ID 列表)
    格式2: train, val (每行分别放一个训练 ID 和一个验证 ID) - 现有项目格式
    格式3: train, val, test (每行一个样本 ID)
    格式4: case_id + fold_X_train/fold_X_val/fold_X_test 列
    
    Returns:
        train_ids, val_ids, test_ids: 三个 ID 列表
    """
    if 'train_ids' in split_df.columns:
        # 格式1: train_ids, val_ids, test_ids
        train_ids = split_df['train_ids'].dropna().tolist()
        val_ids = split_df['val_ids'].dropna().tolist() if 'val_ids' in split_df.columns else []
        test_ids = split_df['test_ids'].dropna().tolist() if 'test_ids' in split_df.columns else []
        
    elif 'train' in split_df.columns:
        # 格式2/3: train, val, test 列 (每行一个 ID)
        # 现有项目格式：train 列包含所有训练 ID，val 列包含所有验证 ID
        train_ids = split_df['train'].dropna().unique().tolist()
        val_ids = split_df['val'].dropna().unique().tolist() if 'val' in split_df.columns else []
        test_ids = split_df['test'].dropna().unique().tolist() if 'test' in split_df.columns else []
        
    elif 'case_id' in split_df.columns:
        # 格式4: 需要根据 fold 列来筛选（此处简化处理）
        train_ids = split_df['case_id'].tolist()
        val_ids = []
        test_ids = []
        
    else:
        raise ValueError(
            f"无法识别划分文件格式。\n"
            f"期望列名: train_ids,val_ids,test_ids 或 train,val,test\n"
            f"实际列名: {list(split_df.columns)}"
        )
    
    # 清理 ID（去除空格，转字符串）
    train_ids = [str(i).strip() for i in train_ids if pd.notna(i)]
    val_ids = [str(i).strip() for i in val_ids if pd.notna(i)]
    test_ids = [str(i).strip() for i in test_ids if pd.notna(i)]
    
    return train_ids, val_ids, test_ids


if __name__ == '__main__':
    main()
