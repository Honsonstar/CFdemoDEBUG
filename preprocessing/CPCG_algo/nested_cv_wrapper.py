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
import sys

# Add the CPCG directory to path for imports
cpog_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cpog_dir)

class NestedCVFeatureSelector:
    """嵌套交叉验证特征选择器"""
    
    def __init__(self, study, data_root_dir, threshold=100, n_jobs=-1):
        """
        Args:
            study: 癌症类型 (如 'blca', 'brca')
            data_root_dir: CPCG原始数据目录
            threshold: 筛选基因数量阈值，如果实际筛选的基因少于这个数，则使用实际数量
            n_jobs: 并行作业数，-1表示使用所有CPU核心
        """
        self.study = study
        self.data_root_dir = data_root_dir
        self.threshold = threshold
        self.n_jobs = n_jobs
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
        print(f"\n[{self.study}] Fold {fold}: 开始完整CPCG特征筛选...")
        print(f"  Train样本数: {len(train_ids)}")
        print(f"  Val样本数: {len(val_ids)}")
        print(f"  Test样本数: {len(test_ids)}")

        # 加载原始数据
        # 1. 读取表达数据（基因 x 样本）
        exp_file = os.path.join(self.data_root_dir, 'tcga_blca', 'data.csv')
        if not os.path.exists(exp_file):
            raise FileNotFoundError(f"找不到表达文件: {exp_file}")
        exp_data = pd.read_csv(exp_file)

        # 2. 读取用于划分的临床数据（获取正确的样本ID）
        clinical_split_file = f'datasets_csv/clinical_data/tcga_{self.study}_clinical.csv'
        if not os.path.exists(clinical_split_file):
            raise FileNotFoundError(f"找不到划分用临床文件: {clinical_split_file}")
        clinical_split_data = pd.read_csv(clinical_split_file)

        # 【修复】ID截取：确保ID格式匹配
        # 1. 从划分数据中截取ID为前12位
        clinical_split_data['case_id_truncated'] = clinical_split_data['case_id'].str[:12]

        # 2. 截取exp_data的列名（样本ID）为前12位
        sample_columns = exp_data.columns[1:]  # 跳过第一列gene_name
        truncated_columns = [col[:12] for col in sample_columns]
        exp_data_renamed = exp_data.copy()
        exp_data_renamed.columns = ['gene_name'] + truncated_columns

        # 3. 将exp_data转换为基因名作为索引的格式（第一列转索引）
        exp_data_renamed.set_index('gene_name', inplace=True)

        # 筛选训练集样本 (仅在训练集上筛选!)
        train_mask = clinical_split_data['case_id_truncated'].isin(train_ids)
        train_clinical_split = clinical_split_data[train_mask].copy()

        # 【调试】打印匹配信息
        print(f"  [DEBUG] clinical_split_data样本数: {len(clinical_split_data)}")
        print(f"  [DEBUG] train_ids示例: {train_ids[:5]}")
        print(f"  [DEBUG] clinical_split_data.case_id_truncated示例: {clinical_split_data['case_id_truncated'].head().tolist()}")
        print(f"  [DEBUG] 匹配到样本数: {train_mask.sum()}")
        print(f"  实际筛选样本数: {len(train_clinical_split)}")

        # 运行完整CPCG流程（使用划分数据中的样本ID来筛选表达数据）
        selected_genes = self._run_full_cpcg(train_clinical_split, exp_data_renamed)

        # 生成特征文件
        feature_file = self._generate_feature_file(
            fold, selected_genes, exp_data_renamed, train_ids + val_ids + test_ids
        )
        
        print(f"  ✓ 生成特征文件: {feature_file}")
        print(f"  筛选基因数: {len(selected_genes)}")
        
        return feature_file

    def _run_semi_parametric_stage1(self, clinical_data, exp_data):
        """运行CPCG Stage1 (半参数化模型) 筛选基因"""
        from Stage1_semi_parametric_model.screen import screen_step_2

        # 预处理数据（添加Censor列以兼容原始CPCG代码）
        clinical_final = clinical_data.copy()

        # 添加case_submitter_id列（如果缺少）
        if 'case_submitter_id' not in clinical_final.columns:
            if 'case_id' in clinical_final.columns:
                clinical_final['case_submitter_id'] = clinical_final['case_id']
            else:
                raise ValueError("找不到case_submitter_id或case_id列")

        if 'Censor' not in clinical_final.columns:
            if 'censorship' in clinical_final.columns:
                clinical_final['Censor'] = clinical_final['censorship']
            else:
                raise ValueError("找不到Censor或censorship列")

        # 筛选生存事件样本
        clinical_final = clinical_final[clinical_final['Censor'] == 1].copy()

        # 添加OS列（如果没有的话）
        if 'OS' not in clinical_final.columns:
            # 从survival_months获取OS值（通常就是survival_months列）
            if 'survival_months' in clinical_final.columns:
                clinical_final['OS'] = clinical_final['survival_months']
            else:
                raise ValueError("找不到OS列或survival_months列")

        # 准备表达数据（转换为原始格式：基因名为第一列，样本为列）
        exp_subset = exp_data.copy()
        exp_subset.reset_index(inplace=True)

        # 运行筛选（传递n_jobs参数）
        result = screen_step_2(
            clinical_final=clinical_final,
            exp_data=exp_subset,
            h_type='OS',
            threshold=self.threshold,
            n_jobs=self.n_jobs
        )

        # 提取基因名
        gene_columns = [col for col in result.columns if col not in ['OS', 'Censor', 'censorship']]
        return gene_columns

    def _run_stage2_causal_discovery(self, clinical_data, exp_data, stage1_genes):
        """运行CPCG Stage2 因果发现 (PC筛选基因"""
        from Stage2.main import cs_step_2

        # 准备表达数据（转换为原始格式）
        exp_subset = exp_data.copy()
        exp_subset.reset_index(inplace=True)

        # 添加OS列（如果没有的话）
        if 'OS' not in clinical_data.columns:
            if 'survival_months' in clinical_data.columns:
                clinical_data['OS'] = clinical_data['survival_months']
            else:
                raise ValueError("找不到OS列或survival_months列")

        # 合并临床数据和表达数据
        # 临床数据的索引应该是样本ID，表达数据的列应该是样本ID
        merged_data = pd.merge(
            clinical_data[['OS']],
            exp_subset,
            left_index=True,
            right_on='gene_name',
            how='inner'
        )
        merged_data.set_index('gene_name', inplace=True)

        # 只保留Stage1筛选出的基因
        available_genes = [g for g in stage1_genes if g in merged_data.columns]
        print(f"    [Stage2] Stage1基因中可用基因数: {len(available_genes)}/{len(stage1_genes)}")

        if len(available_genes) == 0:
            print(f"    [Stage2] 警告: 没有可用基因，直接返回空列表")
            return []

        # 选择数据
        selected_data = merged_data[['OS'] + available_genes]

        # 运行因果发现
        causal_result = cs_step_2(selected_data, hazard_type='OS')

        # 提取基因名（排除OS列）
        causal_genes = [col for col in causal_result.columns if col != 'OS']

        print(f"    [Stage2] 因果发现后保留基因数: {len(causal_genes)}")
        return causal_genes

    def _run_cpog_stage1(self, clinical_data, exp_data):
        """运行CPCG Stage1筛选基因"""
        import sys
        import os
        # Add the CPCG directory to path
        cpog_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, cpog_dir)
        from Stage1_parametric_model.screen import screen_step_1

        # 预处理数据（添加Censor列以兼容原始CPCG代码）
        clinical_final = clinical_data.copy()

        # 添加case_submitter_id列（如果缺少）
        if 'case_submitter_id' not in clinical_final.columns:
            if 'case_id' in clinical_final.columns:
                clinical_final['case_submitter_id'] = clinical_final['case_id']
            else:
                raise ValueError("找不到case_submitter_id或case_id列")

        if 'Censor' not in clinical_final.columns:
            if 'censorship' in clinical_final.columns:
                clinical_final['Censor'] = clinical_final['censorship']
            else:
                raise ValueError("找不到Censor或censorship列")

        # 筛选生存事件样本
        clinical_final = clinical_final[clinical_final['Censor'] == 1].copy()

        # 添加OS列（如果没有的话）
        if 'OS' not in clinical_final.columns:
            # 从survival_months获取OS值（通常就是survival_months列）
            if 'survival_months' in clinical_final.columns:
                clinical_final['OS'] = clinical_final['survival_months']
            else:
                raise ValueError("找不到OS列或survival_months列")

        # 准备表达数据（转换为原始格式：基因名为第一列，样本为列）
        exp_subset = exp_data.copy()
        exp_subset.reset_index(inplace=True)

        # 运行筛选（传递n_jobs参数）
        result = screen_step_1(
            clinical_final=clinical_final,
            exp_data=exp_subset,
            h_type='OS',
            threshold=self.threshold,
            n_jobs=self.n_jobs
        )

        # 提取基因名
        gene_columns = [col for col in result.columns if col != 'OS']
        return gene_columns

    def _run_full_cpcg(self, clinical_data, exp_data):
        """
        运行完整的CPCG流程：
        1. Stage1 (参数化) -> 基因列表A
        2. Stage1 (半参数化) -> 基因列表B
        3. 合并A和B (取并集)
        4. Stage2 (因果发现) -> 最终基因列表C

        如果Stage2返回空结果，则回退到Stage1结果中P值最小的前threshold个基因
        """
        print(f"  1. 运行 Stage1 (参数化模型)...")
        genes_parametric = self._run_cpog_stage1(clinical_data, exp_data)
        print(f"     -> 筛选出 {len(genes_parametric)} 个基因")

        print(f"  2. 运行 Stage1 (半参数化模型)...")
        genes_semi_parametric = self._run_semi_parametric_stage1(clinical_data, exp_data)
        print(f"     -> 筛选出 {len(genes_semi_parametric)} 个基因")

        # Stage1结果合并 (取并集)
        stage1_genes = list(set(genes_parametric) | set(genes_semi_parametric))
        print(f"  -> Stage1 合并后总基因数: {len(stage1_genes)} 个")

        # Stage2因果发现
        print(f"  3. 运行 Stage2 (因果发现)...")
        final_genes = self._run_stage2_causal_discovery(clinical_data, exp_data, stage1_genes)

        # ============================================================
        # 【安全措施】零特征兜底机制
        # ============================================================
        if len(final_genes) == 0:
            print(f"  ⚠️  警告: Stage2 返回空结果，启用兜底机制")
            print(f"      回退到 Stage1 结果，使用前 {self.threshold} 个基因")
            # 使用Stage1结果的并集，如果为空则返回空列表
            if len(stage1_genes) > 0:
                final_genes = stage1_genes[:self.threshold]
                print(f"  -> 兜底机制: 从 Stage1 结果中选取前 {len(final_genes)} 个基因")
            else:
                print(f"  -> 兜底机制: Stage1 也为空，返回空列表")
        else:
            print(f"  -> Stage2 最终保留基因数: {len(final_genes)} 个")

        return final_genes
        
    def _generate_feature_file(self, fold, genes, exp_data, all_ids):
        """生成特征CSV文件"""
        # exp_data 已经是处理过的格式：索引是基因名，列是样本ID
        # 筛选需要的样本
        available_samples = [sid for sid in all_ids if sid in exp_data.columns]
        print(f"    [生成文件] 可用样本数: {len(available_samples)}/{len(all_ids)}")

        # 选择基因和样本
        available_genes = [g for g in genes if g in exp_data.index]
        print(f"    [生成文件] 可用基因数: {len(available_genes)}/{len(genes)}")

        if len(available_genes) == 0:
            print(f"    [生成文件] 警告: 没有可用基因，创建空文件")
            # 创建空文件
            output_file = os.path.join(
                self.temp_dir,
                f'{self.study}_fold_{fold}_features.csv'
            )
            # 创建只有OS列的空DataFrame
            empty_df = pd.DataFrame(columns=['sample_id'] + available_samples)
            empty_df.to_csv(output_file, index=False)
            return output_file

        # 筛选数据
        exp_subset = exp_data.loc[available_genes, available_samples].copy()

        # 转置：样本为行，基因为列
        result = exp_subset.T
        result.reset_index(inplace=True)
        result.rename(columns={'index': 'sample_id'}, inplace=True)

        # 添加OS列（从临床数据中获取）
        # 这里简化处理，假设OS值都是1（实际应从临床数据获取）
        # 由于我们不在这里处理OS，直接添加placeholder
        # result.insert(1, 'OS', 1.0)  # 暂时不加OS列

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
