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
        # --- 【修复开始】强制 ID 截断对齐 ---
        # 确保传入的 ID 列表也是字符串格式，并只取前 12 位 (TCGA-XX-XXXX)
        train_ids = [str(i)[:12] for i in train_ids if str(i) != 'nan']
        val_ids = [str(i)[:12] for i in val_ids if str(i) != 'nan']
        test_ids = [str(i)[:12] for i in test_ids if str(i) != 'nan']
        # ----------------------------------

        print(f"\n[{self.study}] Fold {fold}: 开始 CPCG 筛选 (Train={len(train_ids)})")

        # 1. 读取表达数据 (正确结构: 行=样本ID, 列=基因名)
        exp_file = os.path.join(self.data_root_dir, self.study, 'rna_clean.csv')
        if not os.path.exists(exp_file):
            exp_file = os.path.join(self.data_root_dir, self.study, 'data.csv')
        if not os.path.exists(exp_file): raise FileNotFoundError(f"Missing exp data: {exp_file}")
        # 设置第一列为索引（样本ID）
        exp_data = pd.read_csv(exp_file, index_col=0)
        # 截断样本ID为12位
        exp_data.index = [str(i)[:12] for i in exp_data.index]

        # 2. 读取临床数据
        clinical_file = f'datasets_csv/clinical_data/tcga_{self.study}_clinical.csv'
        if not os.path.exists(clinical_file): raise FileNotFoundError(f"Missing clinical: {clinical_file}")
        clinical_data = pd.read_csv(clinical_file)
        clinical_data['case_id_truncated'] = clinical_data['case_id'].str[:12]

        # 3. 转置表达数据 (变为行=基因, 列=样本)，供 CPCG 算法使用
        exp_data_T = exp_data.T
        exp_data_T.index.name = 'gene_name'
        exp_data_T = exp_data_T.reset_index()

        # 4. 筛选训练集 (使用 truncated ID 进行匹配)
        # 只有当 clinical data 中的短 ID 在我们的 train_ids (已转短 ID) 中时才保留
        train_mask = clinical_data['case_id_truncated'].isin(train_ids)
        train_clinical = clinical_data[train_mask].copy()

        # 【调试信息】打印实际匹配到的样本数
        print(f"  -> ID Matching: Input Train IDs={len(train_ids)} -> Matched Clinical Samples={len(train_clinical)}")

        if len(train_clinical) < 10:
            print("  ⚠️ Warning: Too few training samples matched! Check ID formats.")
            return [] # 避免后续报错

        selected_genes = self._run_full_cpcg(train_clinical, exp_data, exp_data_T)

        return self._generate_feature_file(fold, selected_genes, exp_data_T, train_ids + val_ids + test_ids)

    def _prepare_data(self, clinical):
        df = clinical.copy()

        # 修复：优先使用 case_id_truncated 以匹配表达数据的列名格式(12位)
        if 'case_id_truncated' in df.columns:
            df['case_submitter_id'] = df['case_id_truncated']
        elif 'case_id' in df.columns:
            # 如果没有 truncated，尝试截断原始ID (兜底策略)
            df['case_submitter_id'] = df['case_id'].astype(str).str[:12]

        if 'censorship' in df.columns: df['Censor'] = df['censorship']
        if 'survival_months' in df.columns: df['OS'] = df['survival_months']
        return df

    def _run_full_cpcg(self, clinical, exp_data, exp_data_T=None):
        from Stage1_parametric_model.screen import screen_step_1
        from Stage1_semi_parametric_model.screen import screen_step_2
        from Stage2.main import cs_step_2

        # 预处理 (生成 Censor, OS, case_submitter_id)
        clin = self._prepare_data(clinical)
        
        # Stage 1
        print("  1. Stage 1 (Parametric)...")
        # 参数化模型仅用死亡样本
        res_p = screen_step_1(clin[clin['Censor']==1].copy(), exp_data_T, h_type='OS', threshold=self.threshold, n_jobs=self.n_jobs)
        genes_p = [c for c in res_p.columns if c != 'OS']

        print("  2. Stage 1 (Semi-Parametric)...")
        # 半参数化使用全量样本
        res_sp = screen_step_2(clin.copy(), exp_data_T, h_type='OS', threshold=self.threshold, n_jobs=self.n_jobs)
        genes_sp = [c for c in res_sp.columns if c not in ['OS', 'Censor', 'censorship']]

        # 【修复】过滤掉临床特征列和非基因列
        # 已知需要排除的临床特征
        clinical_features = {
            'case_id', 'case_id_truncated', 'case_submitter_id', 'sample_id',
            'age', 'is_female', 'stage', 'grade', 'site',
            'censorship', 'censorship_dss', 'censorship_pfi',
            'survival_months', 'survival_months_dss', 'survival_months_pfi',
            'OS', 'Censor', 'index', 'oncotree_code', 'slide_id'
        }

        genes_p_filtered = [g for g in genes_p if g not in clinical_features]
        genes_sp_filtered = [g for g in genes_sp if g not in clinical_features]

        # 统计过滤掉的列
        filtered_out_p = set(genes_p) - set(genes_p_filtered)
        filtered_out_sp = set(genes_sp) - set(genes_sp_filtered)

        if filtered_out_p or filtered_out_sp:
            print(f"  [过滤] 移除了 {len(filtered_out_p | filtered_out_sp)} 个非基因列")
            print(f"       包括: {list(filtered_out_p | filtered_out_sp)[:10]}...")

        candidates = list(set(genes_p_filtered) | set(genes_sp_filtered))
        print(f"  -> Stage 1 Candidates: {len(candidates)}")

        # Stage 2
        print("  3. Stage 2 (Causal)...")
        # 强制索引转字符串对齐
        id_col = 'case_id_truncated' if 'case_id_truncated' in clin.columns else 'case_id'
        clin_idx = clin.set_index(id_col)
        clin_idx.index = clin_idx.index.astype(str)

        # 【修复】去除重复索引，保留第一个（避免 merge 失败）
        if clin_idx.index.duplicated().any():
            print(f"  [警告] 临床数据有 {clin_idx.index.duplicated().sum()} 个重复ID，去重处理")
            clin_idx = clin_idx[~clin_idx.index.duplicated(keep='first')]

        # 【关键修复】使用 exp_data (索引=样本ID) 直接 merge
        # 合并临床数据和基因表达数据
        merged = pd.merge(clin_idx[['OS']], exp_data[exp_data.columns[:]], left_index=True, right_index=True, how='inner')

        # 只保留在候选基因列表中的列
        gene_cols = [c for c in merged.columns if c in candidates and c != 'OS']
        merged = merged[['OS'] + gene_cols]

        # 过滤样本：只保留训练集、验证集、测试集的样本
        all_ids = set(train_ids + val_ids + test_ids)
        merged = merged[merged.index.isin(all_ids)]

        # 【关键修复】不要转置！
        # Stage 2 的 PC 算法期望：行=样本, 列=变量（基因+OS）
        # 转置后的 merged_T 会导致：行=变量, 列=样本
        # 这会让 PC 算法计算"样本与样本"的相关性，而不是"基因与基因"的相关性

        # 调试 Stage 2 输入
        print(f"  [Debug Stage 2] merged shape: {merged.shape}")
        print(f"  [Debug Stage 2] 行（样本）: {len(merged)}")
        print(f"  [Debug Stage 2] 列（基因+OS）: {len(merged.columns)}")
        print(f"  [Debug Stage 2] OS in columns: {'OS' in merged.columns}")

        if merged.empty:
            print("  ⚠️ Stage 2 Empty Merge. Fallback to Stage 1.")
            fallback_genes = candidates[:self.threshold]
            print(f"  -> Fallback: 取前 {len(fallback_genes)} 个基因")
            return fallback_genes

        final_df = cs_step_2(merged, hazard_type="OS")
        final_genes = [c for c in final_df.columns if c != 'OS']

        if final_genes:
            print(f"  -> Stage 2 输出: {len(final_genes)} 个基因")
            return final_genes
        else:
            print("  ⚠️ Stage 2 无输出，回退到 Stage 1")
            fallback_genes = candidates[:self.threshold]
            print(f"  -> Fallback: 取前 {len(fallback_genes)} 个基因")
            return fallback_genes

    def _generate_feature_file(self, fold, genes, exp_data, ids):
        # 直接保存到 features 目录（而不是临时目录）
        features_dir = f'features/{self.study}'
        os.makedirs(features_dir, exist_ok=True)
        out_file = os.path.join(features_dir, f'fold_{fold}_genes.csv')

        # exp_data 是转置后的数据：行=基因，列=样本（第一列是 'gene_name'）
        # 样本ID在列中（除了第一列 'gene_name'）
        sample_cols = [c for c in exp_data.columns if c != 'gene_name']
        valid_ids = [i for i in ids if i in sample_cols]
        valid_genes = genes  # genes 已经是基因名列表

        if not valid_genes:
            print(f"   [Warning] No valid genes, creating empty file")
            pd.DataFrame(columns=['sample_id']+valid_ids).to_csv(out_file, index=False)
        else:
            # 选择有效的基因和样本
            cols_to_select = ['gene_name'] + valid_ids
            df = exp_data[cols_to_select].copy()
            df = df[df['gene_name'].isin(valid_genes)]
            df.to_csv(out_file, index=False)

        print(f"   [Saved] {out_file}")
        return out_file

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CPCG Feature Selection for Nested CV')
    parser.add_argument('--study', type=str, required=True, help='Cancer study name (e.g., stad)')
    parser.add_argument('--fold', type=int, required=True, help='Fold number (0-4)')
    parser.add_argument('--split_file', type=str, required=True, help='Path to split CSV file')
    parser.add_argument('--data_root_dir', type=str, required=True, help='Root directory containing RNA data')
    parser.add_argument('--threshold', type=int, default=100, help='Maximum number of genes to select')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')

    args = parser.parse_args()

    # 读取划分文件
    split_df = pd.read_csv(args.split_file)

    # 检测列名格式
    if 'train_ids' in split_df.columns:
        # 格式1: train_ids, val_ids, test_ids (每行一个fold的列表)
        train_ids = split_df['train_ids'].tolist()
        val_ids = split_df['val_ids'].tolist() if 'val_ids' in split_df.columns else []
        test_ids = split_df['test_ids'].tolist()
    elif 'train' in split_df.columns:
        # 格式2: train, val, test (每行一个样本的列表)
        # 取所有非空的train/val/test样本ID
        train_ids = split_df[split_df['train'].notna()]['train'].tolist()
        val_ids = split_df[split_df['val'].notna()]['val'].tolist() if 'val' in split_df.columns else []
        test_ids = split_df[split_df['test'].notna()]['test'].tolist()
    else:
        print(f"❌ 错误: 无法识别划分文件格式")
        print(f"   期望列名: train_ids,val_ids,test_ids 或 train,val,test")
        print(f"   实际列名: {list(split_df.columns)}")
        exit(1)

    # 清理ID（去除空格）
    train_ids = [str(i).strip() for i in train_ids]
    val_ids = [str(i).strip() for i in val_ids]
    test_ids = [str(i).strip() for i in test_ids]

    print(f"   [Info] Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # 运行CPCG
    with NestedCVFeatureSelector(args.study, args.data_root_dir, args.threshold, args.n_jobs) as selector:
        output_file = selector.select_features_for_fold(args.fold, train_ids, val_ids, test_ids)
        print(f"\n✅ Fold {args.fold} 完成!")
        print(f"   输出文件: {output_file}")
