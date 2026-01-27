#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val, _train_val_with_km_plots
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用并行以避免警告

from utils.process_args import _process_args

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)

    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []

    # For multi-task learning
    if args.enable_multitask:
        all_val_stage_accuracy = []

    for i in folds:

        # ============================================================
        # 【新增】支持嵌套CV：检查使用哪种文件名格式
        # ============================================================
        nested_csv_path = '{}/nested_splits_{}.csv'.format(args.split_dir, i)
        standard_csv_path = '{}/splits_{}.csv'.format(args.split_dir, i)

        if os.path.exists(nested_csv_path):
            csv_path = nested_csv_path
            print(f"Using nested CV splits: {csv_path}")
        else:
            csv_path = standard_csv_path

        datasets = args.dataset_factory.return_splits(
            args,
            csv_path=csv_path,
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        fold_results = _train_val_with_km_plots(datasets, i, args)
        #     fold_results = {
        #     'test_results': test_results,
        #     'train_results': train_results,
        #     'metrics': metrics,
        #     'km_p_values': p_values,
        #     'sample_counts': sample_counts
        # }
        metrics = fold_results['metrics']
        results = fold_results['test_results']
        
        # if args.enable_multitask:
        #     val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = metrics
        #     all_val_stage_accuracy.append(val_stage_accuracy)
        # else:
        (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy) = metrics

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)
    
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        try:
            # 将 results 保存在txt文件中
            with open(filename, 'w') as f:
                f.write(str(results))
        except:
            _save_pkl(filename, results)

    # Create summary dataframe
    final_df_dict = {
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        'val_BS': all_val_BS,
    }
    
    if args.enable_multitask:
        final_df_dict['val_stage_accuracy'] = all_val_stage_accuracy

    # === [修改] 自动补齐列长度，防止生成表格时报错 ===
    # 1. 找出所有列中最长的一列是多少
    max_len = 0
    for v in final_df_dict.values():
        if isinstance(v, list):
            max_len = max(max_len, len(v))
    
    # 2. 遍历每一列，如果长度不够，用 None 填满
    padded_columns = []
    for k, v in final_df_dict.items():
        if isinstance(v, list) and len(v) < max_len:
            padded_columns.append(k)
            final_df_dict[k].extend([None] * (max_len - len(v)))
    # 3. 只在最后打印一次汇总（如果有需要padding的列）
    if padded_columns:
        print(f'[DEBUG] Padded columns to length {max_len}: {padded_columns}')
    # ====================================================
    final_df = pd.DataFrame(final_df_dict)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    # Print summary statistics
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for i, cindex in enumerate(all_val_cindex):
        print(f"Fold {i+1}: C-index = {cindex:.4f}")
    print(f"Mean validation C-index: {np.mean(all_val_cindex):.4f} ± {np.std(all_val_cindex):.4f}")
    print(f"Mean validation C-index IPCW: {np.mean(all_val_cindex_ipcw):.4f} ± {np.std(all_val_cindex_ipcw):.4f}")
    print(f"Mean validation IBS: {np.mean(all_val_IBS):.4f} ± {np.std(all_val_IBS):.4f}")
    print(f"Mean validation IAUC: {np.mean(all_val_iauc):.4f} ± {np.std(all_val_iauc):.4f}")
    print(f"Mean validation loss: {np.mean(all_val_loss):.4f} ± {np.std(all_val_loss):.4f}")
    
    if args.enable_multitask:
        #print(f"Mean validation stage accuracy: {np.mean(all_val_stage_accuracy):.4f} ± {np.std(all_val_stage_accuracy):.4f}")
        # === [修改] 安全打印 Stage Accuracy (过滤 None 值) ===
        # 先把 None 过滤掉，只保留有效数字
        valid_stage_acc = [x for x in all_val_stage_accuracy if x is not None]

        if len(valid_stage_acc) > 0:
            print(f"Mean validation stage accuracy: {np.mean(valid_stage_acc):.4f} ± {np.std(valid_stage_acc):.4f}")
        else:
            print("Mean validation stage accuracy: N/A (No valid predictions)")
        # ========================================================
    
    print("="*50)


def main0(args):

    #----> prep for 5 fold cv study
    
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    all_val_cindex = []
    all_val_cindex_ipcw = []
    all_val_BS = []
    all_val_IBS = []
    all_val_iauc = []
    all_val_loss = []
    
    # For multi-task learning
    if args.enable_multitask:
        all_val_stage_accuracy = []

    for i in folds:

        # ============================================================
        # 【新增】支持嵌套CV：检查使用哪种文件名格式
        # ============================================================
        nested_csv_path = '{}/nested_splits_{}.csv'.format(args.split_dir, i)
        standard_csv_path = '{}/splits_{}.csv'.format(args.split_dir, i)

        if os.path.exists(nested_csv_path):
            csv_path = nested_csv_path
            print(f"Using nested CV splits: {csv_path}")
        else:
            csv_path = standard_csv_path

        datasets = args.dataset_factory.return_splits(
            args,
            csv_path=csv_path,
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        results, metrics = _train_val(datasets, i, args)
        
        # if args.enable_multitask:
        #     val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = metrics
        #     all_val_stage_accuracy.append(val_stage_accuracy)
        # else:
        val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = metrics

        all_val_cindex.append(val_cindex)
        all_val_cindex_ipcw.append(val_cindex_ipcw)
        all_val_BS.append(val_BS)
        all_val_IBS.append(val_IBS)
        all_val_iauc.append(val_iauc)
        all_val_loss.append(total_loss)
    
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        print("Saving results...")
        try:
            # 将 results 保存在txt文件中
            with open(filename, 'w') as f:
                f.write(str(results))
        except:
            _save_pkl(filename, results)

    # Create summary dataframe
    final_df_dict = {
        'folds': folds,
        'val_cindex': all_val_cindex,
        'val_cindex_ipcw': all_val_cindex_ipcw,
        'val_IBS': all_val_IBS,
        'val_iauc': all_val_iauc,
        "val_loss": all_val_loss,
        'val_BS': all_val_BS,
    }
    
    if args.enable_multitask:
        final_df_dict['val_stage_accuracy'] = all_val_stage_accuracy

    final_df = pd.DataFrame(final_df_dict)

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
        
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    # Print summary statistics
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    for i, cindex in enumerate(all_val_cindex):
        print(f"Fold {i+1}: C-index = {cindex:.4f}")
    print(f"Mean validation C-index: {np.mean(all_val_cindex):.4f} ± {np.std(all_val_cindex):.4f}")
    print(f"Mean validation C-index IPCW: {np.mean(all_val_cindex_ipcw):.4f} ± {np.std(all_val_cindex_ipcw):.4f}")
    print(f"Mean validation IBS: {np.mean(all_val_IBS):.4f} ± {np.std(all_val_IBS):.4f}")
    print(f"Mean validation IAUC: {np.mean(all_val_iauc):.4f} ± {np.std(all_val_iauc):.4f}")
    print(f"Mean validation loss: {np.mean(all_val_loss):.4f} ± {np.std(all_val_loss):.4f}")
    
    if args.enable_multitask:
        print(f"Mean validation stage accuracy: {np.mean(all_val_stage_accuracy):.4f} ± {np.std(all_val_stage_accuracy):.4f}")
    
    print("="*50)

if __name__ == "__main__":
    start = timer()
    # import sys
    # sys.argv = [
    #     'main.py',
    #     '--study', 'tcga_blca',
    #     '--task', 'survival',
    #     '--split_dir', 'splits',
    #     '--which_splits', '5foldcv',
    #     '--type_of_path', 'combine',
    #     '--modality', 'snn',
    #     '--data_root_dir', '/data/TCGA/BLCA/features/pt_files/',
    #     '--label_file', 'datasets_csv/metadata/tcga_blca.csv',
    #     '--omics_dir', 'datasets_csv/raw_rna_data/combine/blca',
    #     '--results_dir', 'results_blca',
    #     '--batch_size', '1',
    #     '--lr', '0.00005',
    #     '--opt', 'adam',
    #     '--reg', '0.00001',
    #     '--alpha_surv', '0.5',
    #     '--weighted_sample',
    #     '--max_epochs', '5',
    #     '--encoding_dim', '256',
    #     '--label_col', 'survival_months_dss',
    #     '--k', '5',
    #     '--bag_loss', 'nll_surv',
    #     '--n_classes', '4',
    #     '--num_patches', '4090',
    #     '--wsi_projection_dim', '256',
    #     '--encoding_layer_1_dim', '8',
    #     '--encoding_layer_2_dim', '16',
    #     '--encoder_dropout', '0.25',
    #     '--enable_multitask', '--multitask_weight', '0.3'
    # ]

    #----> read the args
    args = _process_args()
    draw = 1
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality == "survpath" else False,
        type_of_pathway=args.type_of_path,
        enable_multitask=args.enable_multitask)

    #---> perform the experiment
    if draw:
        results = main(args)
    else:
        main0(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))