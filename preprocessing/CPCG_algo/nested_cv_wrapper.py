"""
嵌套交叉验证包装器
[修复版 2.0]
1. 修正半参数模型输入：不再过滤 Censor==1，使用全量数据
2. 修正 Stage 2 合并逻辑：转置表达数据，确保 SampleID 对齐
3. 保留参数模型输入：仅使用 Censor==1
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
        self.study = study
        self.data_root_dir = data_root_dir
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.temp_dir = None
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='cpog_')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def select_features_for_fold(self, fold, train_ids, val_ids, test_ids):
        print(f"\n[{self.study}] Fold {fold}: 开始完整CPCG特征筛选...")
        print(f"  Train样本数: {len(train_ids)}")

        # 1. 读取表达数据
        exp_file = os.path.join(self.data_root_dir, f'tcga_{self.study}', 'data.csv')
        # 兼容部分癌种文件夹可能没有前缀的情况
        if not os.path.exists(exp_file):
             exp_file_alt = os.path.join(self.data_root_dir, self.study, 'data.csv')
             if os.path.exists(exp_file_alt):
                 exp_file = exp_file_alt
                 
        if not os.path.exists(exp_file):
            raise FileNotFoundError(f"找不到表达文件: {exp_file}")
        exp_data = pd.read_csv(exp_file)

        # 2. 读取用于划分的临床数据
        clinical_split_file = f'datasets_csv/clinical_data/tcga_{self.study}_clinical.csv'
        if not os.path.exists(clinical_split_file):
            raise FileNotFoundError(f"找不到临床文件: {clinical_split_file}")
        clinical_split_data = pd.read_csv(clinical_split_file)

        # ID 截取对齐
        clinical_split_data['case_id_truncated'] = clinical_split_data['case_id'].str[:12]

        sample_columns = exp_data.columns[1:]
        truncated_columns = [col[:12] for col in sample_columns]
        exp_data_renamed = exp_data.copy()
        exp_data_renamed.columns = ['gene_name'] + truncated_columns
        exp_data_renamed.set_index('gene_name', inplace=True)

        # 筛选训练集
        train_mask = clinical_split_data['case_id_truncated'].isin(train_ids)
        train_clinical_split = clinical_split_data[train_mask].copy()

        print(f"  实际筛选样本数: {len(train_clinical_split)}")

        # 运行完整流程
        selected_genes = self._run_full_cpcg(train_clinical_split, exp_data_renamed)

        # 生成文件
        feature_file = self._generate_feature_file(
            fold, selected_genes, exp_data_renamed, train_ids + val_ids + test_ids
        )
        
        return feature_file

    def _prepare_common_clinical(self, clinical_data):
        """通用临床数据预处理"""
        df = clinical_data.copy()
        if 'case_id' in df.columns and 'case_submitter_id' not in df.columns:
            df['case_submitter_id'] = df['case_id']
        
        if 'censorship' in df.columns:
            df['Censor'] = df['censorship']
        
        if 'survival_months' in df.columns:
            df['OS'] = df['survival_months']
            
        return df

    def _run_cpog_stage1(self, clinical_data, exp_data):
        """参数化模型：仅使用死亡样本 (Censor=1)"""
        from Stage1_parametric_model.screen import screen_step_1
        
        clinical_final = self._prepare_common_clinical(clinical_data)
        
        # 【关键】参数化模型只看发生事件的样本
        clinical_final = clinical_final[clinical_final['Censor'] == 1].copy()
        
        exp_subset = exp_data.copy()
        exp_subset.reset_index(inplace=True)

        result = screen_step_1(
            clinical_final=clinical_final,
            exp_data=exp_subset,
            h_type='OS',
            threshold=self.threshold,
            n_jobs=self.n_jobs
        )
        return [col for col in result.columns if col != 'OS']

    def _run_semi_parametric_stage1(self, clinical_data, exp_data):
        """半参数化模型：使用所有样本 (修正点)"""
        from Stage1_semi_parametric_model.screen import screen_step_2
        
        clinical_final = self._prepare_common_clinical(clinical_data)
        
        # 【关键修正】这里不再过滤 Censor=1，保留所有样本！
        # clinical_final = clinical_final[clinical_final['Censor'] == 1].copy() <--- 已删除
        
        exp_subset = exp_data.copy()
        exp_subset.reset_index(inplace=True)

        result = screen_step_2(
            clinical_final=clinical_final,
            exp_data=exp_subset,
            h_type='OS',
            threshold=self.threshold,
            n_jobs=self.n_jobs
        )
        # 排除非基因列
        return [col for col in result.columns if col not in ['OS', 'Censor', 'censorship']]

    def _run_stage2_causal_discovery(self, clinical_data, exp_data, stage1_genes):
        """修复了合并逻辑的 Stage 2"""
        from Stage2.main import cs_step_2

        # 1. 准备临床数据
        clinical_sub = self._prepare_common_clinical(clinical_data)
        
        id_col = 'case_id_truncated' if 'case_id_truncated' in clinical_sub.columns else 'case_id'
        if id_col in clinical_sub.columns:
            clinical_sub = clinical_sub.set_index(id_col)
        
        # 2. 准备表达数据 (转置：行=样本, 列=基因)
        # 【关键修正】这里进行转置，因为我们要把 SampleID 当作键来合并
        exp_T = exp_data.T 
        
        # 3. 过滤 Stage 1 基因
        valid_genes = [g for g in stage1_genes if g in exp_T.columns]
        if not valid_genes:
            print("    ⚠️  Stage 2: 没有有效的 Stage 1 基因")
            return []
            
        exp_T_filtered = exp_T[valid_genes]

        # 4. 合并 (Inner Join on Index = SampleID)
        merged_data = pd.merge(
            clinical_sub[['OS']],
            exp_T_filtered,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        if merged_data.empty:
            print("    ⚠️  Stage 2: 合并后数据为空 (样本ID不匹配)")
            return []

        # 5. 运行 PC 算法
        print(f"    [Stage2] 输入数据形状: {merged_data.shape}")
        causal_result = cs_step_2(merged_data, hazard_type='OS')
        
        return [col for col in causal_result.columns if col != 'OS']

    def _run_full_cpcg(self, clinical_data, exp_data):
        print(f"  1. 运行 Stage1 (参数化)...")
        genes_p = self._run_cpog_stage1(clinical_data, exp_data)
        print(f"     -> 筛选出 {len(genes_p)} 个基因")
        
        print(f"  2. 运行 Stage1 (半参数化)...")
        genes_sp = self._run_semi_parametric_stage1(clinical_data, exp_data)
        print(f"     -> 筛选出 {len(genes_sp)} 个基因")
        
        stage1_genes = list(set(genes_p) | set(genes_sp))
        print(f"  -> Stage1 并集: {len(stage1_genes)} 个")

        print(f"  3. 运行 Stage2 (因果发现)...")
        final_genes = self._run_stage2_causal_discovery(clinical_data, exp_data, stage1_genes)

        # 兜底机制
        if len(final_genes) == 0:
            print(f"  ⚠️  Stage 2 结果为空，触发兜底机制 (使用 Stage 1 前 {self.threshold} 个)")
            final_genes = stage1_genes[:self.threshold]
        else:
            print(f"  -> Stage 2 完成，保留 {len(final_genes)} 个基因")
            
        return final_genes

    def _generate_feature_file(self, fold, genes, exp_data, all_ids):
        available_samples = [sid for sid in all_ids if sid in exp_data.columns]
        available_genes = [g for g in genes if g in exp_data.index]
        
        output_file = os.path.join(self.temp_dir, f'{self.study}_fold_{fold}_features.csv')
        
        if not available_genes:
            pd.DataFrame(columns=['sample_id'] + available_samples).to_csv(output_file, index=False)
            return output_file

        exp_subset = exp_data.loc[available_genes, available_samples].T
        exp_subset.reset_index(inplace=True)
        exp_subset.rename(columns={'index': 'sample_id'}, inplace=True)
        
        exp_subset.to_csv(output_file, index=False)
        return output_file
