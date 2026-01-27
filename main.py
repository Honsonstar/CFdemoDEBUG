import torch
import pandas as pd
import numpy as np
import os
import warnings
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val_with_km_plots
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment
from utils.process_args import _process_args

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    folds = _get_start_end(args)
    
    # 存储结果
    results_store = {
        'cindex': [], 'cindex_ipcw': [], 'BS': [], 'IBS': [], 'iauc': [], 'loss': [], 'acc': []
    }

    for i in folds:
        # 路径处理
        nested_csv = '{}/nested_splits_{}.csv'.format(args.split_dir, i)
        std_csv = '{}/splits_{}.csv'.format(args.split_dir, i)
        csv_path = nested_csv if os.path.exists(nested_csv) else std_csv
        
        print(f"Loading Fold {i} data from {csv_path}")
        datasets = args.dataset_factory.return_splits(args, csv_path=csv_path, fold=i)
        
        # 训练
        fold_res = _train_val_with_km_plots(datasets, i, args)
        metrics = fold_res['metrics']
        
        # 解包指标
        if len(metrics) == 7:
            val_c, val_cipcw, val_bs, val_ibs, val_iauc, loss, val_acc = metrics
        else:
            val_c, val_cipcw, val_bs, val_ibs, val_iauc, loss = metrics
            val_acc = 0.0
            
        results_store['cindex'].append(val_c)
        results_store['cindex_ipcw'].append(val_cipcw)
        results_store['BS'].append(val_bs)
        results_store['IBS'].append(val_ibs)
        results_store['iauc'].append(val_iauc)
        results_store['loss'].append(loss)
        if args.enable_multitask: results_store['acc'].append(val_acc)

        # 保存单折详细结果
        try:
            with open(os.path.join(args.results_dir, f'split_{i}_results.pkl'), 'w') as f:
                f.write(str(fold_res['test_results']))
        except:
            _save_pkl(os.path.join(args.results_dir, f'split_{i}_results.pkl'), fold_res['test_results'])

    # 保存 CSV 汇总
    df = pd.DataFrame({'folds': folds, 'val_cindex': results_store['cindex']})
    
    # 自动补充其他列 (防止长度不一致报错)
    max_len = len(folds)
    for k, v in results_store.items():
        if k != 'cindex': # folds 和 cindex 已添加
            if len(v) < max_len: v.extend([None]*(max_len-len(v)))
            df[f'val_{k}'] = v

    save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1]) if len(folds) != args.k else 'summary.csv'
    df.to_csv(os.path.join(args.results_dir, save_name), index=False)

    # --- 打印当前运行的 Fold 结果 ---
    print("\n" + "="*50)
    print("CURRENT RUN SUMMARY")
    print("="*50)
    for fold_idx, cindex in zip(folds, results_store['cindex']):
        print(f"Fold {fold_idx}: C-index = {cindex:.4f}")
    print("="*50)

if __name__ == "__main__":
    args = _process_args()
    args = _prepare_for_experiment(args)
    
    # 【关键修复】添加 print_info=True
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study, 
        label_file=args.label_file, 
        omics_dir=args.omics_dir, 
        seed=args.seed,
        print_info=True,  # <--- 补回这个参数
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6, 
        num_patches=args.num_patches,
        is_mcat = "coattn" in args.modality, 
        is_survpath = args.modality == "survpath",
        type_of_pathway=args.type_of_path, 
        enable_multitask=args.enable_multitask
    )
    
    main(args)
