import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import sys

cpog_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, cpog_dir)

class NestedCVFeatureSelector:
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
        print(f"\n[{self.study}] Fold {fold}: 开始 CPCG 筛选 (Train={len(train_ids)})")
        
        # 1. 读取表达数据
        exp_file = os.path.join(self.data_root_dir, f'tcga_{self.study}', 'data.csv')
        if not os.path.exists(exp_file):
             exp_file = os.path.join(self.data_root_dir, self.study, 'data.csv')
        if not os.path.exists(exp_file): raise FileNotFoundError(f"Missing exp data: {exp_file}")
        exp_data = pd.read_csv(exp_file)

        # 2. 读取临床数据
        clinical_file = f'datasets_csv/clinical_data/tcga_{self.study}_clinical.csv'
        if not os.path.exists(clinical_file): raise FileNotFoundError(f"Missing clinical: {clinical_file}")
        clinical_data = pd.read_csv(clinical_file)
        clinical_data['case_id_truncated'] = clinical_data['case_id'].str[:12]

        # 3. 处理表达数据列名
        sample_cols = exp_data.columns[1:]
        exp_data.columns = ['gene_name'] + [c[:12] for c in sample_cols]
        exp_data.set_index('gene_name', inplace=True)

        # 4. 筛选训练集
        train_mask = clinical_data['case_id_truncated'].isin(train_ids)
        train_clinical = clinical_data[train_mask].copy()
        
        selected_genes = self._run_full_cpcg(train_clinical, exp_data)
        
        return self._generate_feature_file(fold, selected_genes, exp_data, train_ids + val_ids + test_ids)

    def _prepare_data(self, clinical):
        df = clinical.copy()
        # 【关键修复】补回 case_id -> case_submitter_id 的映射
        if 'case_id' in df.columns and 'case_submitter_id' not in df.columns:
            df['case_submitter_id'] = df['case_id']
            
        if 'censorship' in df.columns: df['Censor'] = df['censorship']
        if 'survival_months' in df.columns: df['OS'] = df['survival_months']
        return df

    def _run_full_cpcg(self, clinical, exp_data):
        from Stage1_parametric_model.screen import screen_step_1
        from Stage1_semi_parametric_model.screen import screen_step_2
        from Stage2.main import cs_step_2

        # 预处理 (生成 Censor, OS, case_submitter_id)
        clin = self._prepare_data(clinical)
        
        # Stage 1
        print("  1. Stage 1 (Parametric)...")
        # 参数化模型仅用死亡样本
        res_p = screen_step_1(clin[clin['Censor']==1].copy(), exp_data.copy().reset_index(), h_type='OS', threshold=self.threshold, n_jobs=self.n_jobs)
        genes_p = [c for c in res_p.columns if c != 'OS']
        
        print("  2. Stage 1 (Semi-Parametric)...")
        # 半参数化使用全量样本
        res_sp = screen_step_2(clin.copy(), exp_data.copy().reset_index(), h_type='OS', threshold=self.threshold, n_jobs=self.n_jobs)
        genes_sp = [c for c in res_sp.columns if c not in ['OS', 'Censor', 'censorship']]
        
        candidates = list(set(genes_p) | set(genes_sp))
        print(f"  -> Stage 1 Candidates: {len(candidates)}")

        # Stage 2
        print("  3. Stage 2 (Causal)...")
        # 强制索引转字符串对齐
        id_col = 'case_id_truncated' if 'case_id_truncated' in clin.columns else 'case_id'
        clin_idx = clin.set_index(id_col)
        clin_idx.index = clin_idx.index.astype(str)
        
        exp_T = exp_data.T
        exp_T.index = exp_T.index.astype(str)
        
        valid_genes = [g for g in candidates if g in exp_T.columns]
        
        # 合并数据
        merged = pd.merge(clin_idx[['OS']], exp_T[valid_genes], left_index=True, right_index=True, how='inner')
        
        if merged.empty:
            print("  ⚠️ Stage 2 Empty Merge. Fallback to Stage 1.")
            return candidates[:self.threshold]
            
        final_df = cs_step_2(merged, hazard_type='OS')
        final_genes = [c for c in final_df.columns if c != 'OS']
        
        return final_genes if final_genes else candidates[:self.threshold]

    def _generate_feature_file(self, fold, genes, exp_data, ids):
        out_file = os.path.join(self.temp_dir, f'{self.study}_fold_{fold}.csv')
        valid_ids = [i for i in ids if i in exp_data.columns]
        valid_genes = [g for g in genes if g in exp_data.index]
        
        if not valid_genes:
            pd.DataFrame(columns=['sample_id']+valid_ids).to_csv(out_file, index=False)
        else:
            df = exp_data.loc[valid_genes, valid_ids].T
            df.index.name = 'sample_id'
            df.reset_index().to_csv(out_file, index=False)
        return out_file
