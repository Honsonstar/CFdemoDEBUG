"""
嵌套交叉验证包装器
用于在每折训练前动态筛选特征，避免数据泄露
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

class NestedCVFeatureSelector:
    """嵌套交叉验证特征选择器"""
    
    def __init__(self, study, data_root_dir, threshold=100):
        """
        Args:
            study: 癌症类型 (如 'blca', 'brca')
            data_root_dir: CPCG原始数据目录
            threshold: 筛选基因数量阈值，如果实际筛选的基因少于这个数，则使用实际数量
        """
        self.study = study
        self.data_root_dir = data_root_dir
        self.threshold = threshold
        self.temp_dir = None
        
    def __enter__(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp(prefix='cpog_')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def select_features_for_fold(self, fold, train_ids, val_ids, test_ids):
        """
        为指定折筛选特征
        
        Args:
            fold: 折数 (0-4)
            train_ids: 训练集样本ID列表
            val_ids: 验证集样本ID列表
            test_ids: 测试集样本ID列表
            
        Returns:
            str: 特征文件路径
        """
        print(f"\n[{self.study}] Fold {fold}: 开始特征筛选...")
        print(f"  Train样本数: {len(train_ids)}")
        print(f"  Val样本数: {len(val_ids)}")
        print(f"  Test样本数: {len(test_ids)}")
        
        # 加载原始数据
        clinical_file = os.path.join(self.data_root_dir, 'tcga_brca', 'clinical.CSV')
        exp_file = os.path.join(self.data_root_dir, 'tcga_brca', 'data.csv')
        
        if not os.path.exists(clinical_file):
            raise FileNotFoundError(f"找不到临床文件: {clinical_file}")
        if not os.path.exists(exp_file):
            raise FileNotFoundError(f"找不到表达文件: {exp_file}")
            
        clinical_data = pd.read_csv(clinical_file)
        exp_data = pd.read_csv(exp_file)
        
        # 筛选训练集样本 (仅在训练集上筛选!)
        train_mask = clinical_data['case_submitter_id'].isin(train_ids)
        train_clinical = clinical_data[train_mask].copy()
        
        print(f"  实际筛选样本数: {len(train_clinical)}")
        
        # 运行CPCG Stage1 (参数化模型)
        selected_genes = self._run_cpog_stage1(train_clinical, exp_data)
        
        # 生成特征文件
        feature_file = self._generate_feature_file(
            fold, selected_genes, exp_data, train_ids + val_ids + test_ids
        )
        
        print(f"  ✓ 生成特征文件: {feature_file}")
        print(f"  筛选基因数: {len(selected_genes)}")
        
        return feature_file
        
    def _run_cpog_stage1(self, clinical_data, exp_data):
        """运行CPCG Stage1筛选基因"""
        import sys
        import os
        # Add the CPCG directory to path
        cpog_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, cpog_dir)
        from Stage1_parametric_model.screen import screen_step_1
        
        # 预处理数据
        clinical_final = clinical_data[
            clinical_data['Censor'] == 1
        ].copy()
        
        # 筛选生存事件样本
        exp_subset = exp_data[
            exp_data.iloc[:, 0].isin(clinical_final['case_submitter_id'])
        ].copy()
        
        # 运行筛选
        result = screen_step_1(
            clinical_final=clinical_final,
            exp_data=exp_subset,
            h_type='OS',
            threshold=self.threshold
        )
        
        # 提取基因名
        gene_columns = [col for col in result.columns if col != 'OS']
        return gene_columns
        
    def _generate_feature_file(self, fold, genes, exp_data, all_ids):
        """生成特征CSV文件"""
        # 筛选需要的样本和基因
        sample_mask = exp_data.iloc[:, 0].isin(all_ids)
        exp_subset = exp_data[sample_mask].copy()
        
        # 选择基因列
        gene_cols = ['Unnamed: 0'] + ['OS'] + genes
        available_cols = [col for col in gene_cols if col in exp_subset.columns]
        
        result = exp_subset[available_cols].copy()
        
        # 保存到临时文件
        output_file = os.path.join(
            self.temp_dir, 
            f'{self.study}_fold_{fold}_features.csv'
        )
        result.to_csv(output_file, index=False)
        
        return output_file

def create_nested_splits(clinical_file, output_dir, n_splits=5, val_size=0.15):
    """
    创建嵌套交叉验证划分
    
    Args:
        clinical_file: 临床数据文件路径
        output_dir: 输出目录
        n_splits: 折数
        val_size: 验证集比例
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(clinical_file)
    
    # 获取ID和标签
    ids = df['case_id'].values if 'case_id' in df.columns else df.iloc[:, 0].values
    labels = df['censorship'].values if 'censorship' in df.columns else df.iloc[:, 1].values
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    splits_info = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(ids, labels)):
        train_val_ids = ids[train_val_idx]
        train_val_labels = labels[train_val_idx]
        test_ids = ids[test_idx]
        
        # 划分训练/验证
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_ids)),
            test_size=val_size,
            stratify=train_val_labels,
            random_state=42
        )
        
        train_ids = train_val_ids[train_idx]
        val_ids = train_val_ids[val_idx]
        
        # 保存划分
        split_df = pd.DataFrame({
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        })
        
        split_file = os.path.join(output_dir, f'nested_splits_{fold}.csv')
        split_df.to_csv(split_file, index=False)
        
        splits_info.append({
            'fold': fold,
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'test_size': len(test_ids),
            'file': split_file
        })
        
        print(f"Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # 保存汇总信息
    info_df = pd.DataFrame(splits_info)
    info_file = os.path.join(output_dir, 'splits_summary.csv')
    info_df.to_csv(info_file, index=False)
    
    print(f"\n✓ 嵌套CV划分完成: {output_dir}")
    return splits_info

if __name__ == "__main__":
    # 示例用法
    clinical_file = 'datasets_csv/clinical_data/tcga_blca_clinical.csv'
    output_dir = 'splits/nested_cv'
    
    splits_info = create_nested_splits(
        clinical_file=clinical_file,
        output_dir=output_dir,
        n_splits=5
    )
    
    print("\n嵌套CV划分完成！")
    print("接下来运行:")
    print(f"  python nested_cv_wrapper.py --study blca --splits_dir {output_dir}")
