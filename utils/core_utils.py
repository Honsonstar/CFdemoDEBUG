from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
# from models.model_SNNOmics0805 import SNNOmics
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
from models.model_MCATPathways import MCATPathways
from models.model_SurvPath import SurvPath
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import pickle

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss, MultiTaskLoss

import torch.optim as optim


def _save_visualization_explanations(case_id, explanation_data, model, save_dir='explanation_results',
                                    dataset_name=None, fold_idx=None):
    """
    【新增函数】保存单个样本的可视化解释结果

    Args:
        case_id: 样本的case ID
        explanation_data: 从模型返回的解释数据字典
        model: 模型实例（用于调用可视化方法）
        save_dir: 保存根目录
        dataset_name: 数据集名称（如 'TCGA_BRCA'）
        fold_idx: Fold编号（如 0, 1, 2...）
    """
    # 【修改】构建带有数据集和fold的子目录结构
    if dataset_name and fold_idx is not None:
        # 格式： explanation_results/TCGA_BRCA-Fold0/
        fold_dir = os.path.join(save_dir, f'{dataset_name}-Fold_dbiase{fold_idx}')
    else:
        # 如果没有提供，使用默认目录
        fold_dir = save_dir

    # 创建保存目录
    os.makedirs(fold_dir, exist_ok=True)

    # 提取数据
    word_tokens = explanation_data['word_tokens']
    word_scores = explanation_data['word_scores']
    stop_words = explanation_data['stop_words']
    punctuation = explanation_data['punctuation']
    pathway_names = explanation_data['pathway_names']
    pathway_attention_weights = explanation_data['pathway_attention_weights']

    # ===== 1. 处理并保存文本关键词 =====
    filtered_words = []
    filtered_scores = []
    for word, score in zip(word_tokens, word_scores):
        if word.lower() not in stop_words and word not in punctuation:
            filtered_words.append(word)
            filtered_scores.append(score.item())

    # 【修改】保存原始分数到CSV（保持数值准确性）
    df = pd.DataFrame({'word': filtered_words, 'attention': filtered_scores})
    df = df.sort_values(by='attention', ascending=False)
    csv_path = os.path.join(fold_dir, f'{case_id}_text_keywords.csv')
    df.to_csv(csv_path, index=False)

    # 【修改】Min-Max归一化word_scores，使可视化更明显
    import torch
    word_scores_array = word_scores.cpu().numpy() if isinstance(word_scores, torch.Tensor) else np.array([s.item() if hasattr(s, 'item') else s for s in word_scores])
    if word_scores_array.max() > word_scores_array.min():
        normalized_word_scores = (word_scores_array - word_scores_array.min()) / (word_scores_array.max() - word_scores_array.min())
    else:
        normalized_word_scores = word_scores_array
    normalized_word_scores_tensor = torch.from_numpy(normalized_word_scores).float()

    # 保存高亮文本到HTML（使用归一化后的分数）
    html_path = os.path.join(fold_dir, f'{case_id}_highlighted_text.html')
    model.visualize_keywords_in_context(
        word_tokens, normalized_word_scores_tensor, stop_words, punctuation,
        save_path=html_path
    )

    # ===== 2. 保存pathway重要性 =====
    pathway_csv_path = os.path.join(fold_dir, f'{case_id}_top_pathways.csv')
    pathway_fig_path = os.path.join(fold_dir, f'{case_id}_top_pathways.png')

    # 【修改】先保存原始分数的CSV
    pathway_weights_array = pathway_attention_weights.cpu().detach().numpy()

    # === [关键修改] 只有当数组不为空时（size > 0），才进行可视化 ===
    if pathway_weights_array.size > 0:
        pathway_csv_path = os.path.join(fold_dir, f'{case_id}_top_pathways.csv')
        pathway_fig_path = os.path.join(fold_dir, f'{case_id}_top_pathways.png')

        pathway_df = pd.DataFrame({
            'Pathway': pathway_names,
            'Attention_Score': pathway_weights_array
        })
        pathway_df_sorted = pathway_df.sort_values(by='Attention_Score', ascending=False).head(15)
        pathway_df_sorted.to_csv(pathway_csv_path, index=False)

        # Min-Max归一化
        if pathway_weights_array.max() > pathway_weights_array.min():
            normalized_pathway_weights = (pathway_weights_array - pathway_weights_array.min()) / (pathway_weights_array.max() - pathway_weights_array.min())
        else:
            normalized_pathway_weights = pathway_weights_array
        
        normalized_pathway_weights_tensor = torch.from_numpy(normalized_pathway_weights).float()

        # 生成图片
        model.visualize_top_pathways(
            normalized_pathway_weights_tensor, pathway_names, top_k=15,
            save_path=None,
            save_figure=pathway_fig_path
        )

    print(f"✓ [{case_id}] 已保存可视化解释结果到: {fold_dir}")



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or MultiTaskLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        if args.enable_multitask:
            loss_fn = MultiTaskLoss(
                survival_alpha=args.alpha_surv,
                multitask_weight=args.multitask_weight
            )
        else:
            # loss_fn = NLLSurvLoss(alpha=args.alpha_surv) # 改
            loss_fn = MultiTaskLoss(
                survival_alpha=args.alpha_surv,
                multitask_weight=0
            )
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    # ======================================================================
    # 【核心修改】学习率配置策略
    # ======================================================================
    # 【修改时间】2026-01-22
    # 【修改目的】统一学习率配置，提升模型训练稳定性
    #
    # 历史变更:
    # 1. 最初: 文本 lr=1e-4, 基因 lr=3e-4 (差异化学习率)
    # 2. 原因: 差异化学习率在仅基因模式下有效，但可能导致不稳定
    # 3. 修改: 使用统一学习率 (args.lr)，所有参数组使用相同的lr
    # 4. 优势: 提升训练稳定性，简化超参数调优
    #
    # 兼容性说明:
    # - 支持命令行参数 --text_lr 和 --gene_lr (用于对比实验)
    # - 支持环境变量 TEXT_LR 和 GENE_LR (用于网格搜索)
    # - 如果未提供，则自动使用统一的 args.lr
    # ======================================================================
    import os

    # 文本模型学习率配置
    # 优先级: 命令行参数 > 环境变量 > 统一学习率
    if hasattr(args, 'text_lr') and args.text_lr is not None:
        text_lr = args.text_lr
        print(f"[Config] Using text_lr from command line: {text_lr}")
    elif os.environ.get('TEXT_LR'):
        text_lr = float(os.environ.get('TEXT_LR'))
        print(f"[Config] Using text_lr from environment: {text_lr}")
    else:
        text_lr = args.lr  # 使用统一学习率
        print(f"[Config] Using unified lr for text_lr: {text_lr}")

    # 基因网络学习率配置
    # 优先级: 命令行参数 > 环境变量 > 统一学习率
    if hasattr(args, 'gene_lr') and args.gene_lr is not None:
        gene_lr = args.gene_lr
        print(f"[Config] Using gene_lr from command line: {gene_lr}")
    elif os.environ.get('GENE_LR'):
        gene_lr = float(os.environ.get('GENE_LR'))
        print(f"[Config] Using gene_lr from environment: {gene_lr}")
    else:
        gene_lr = args.lr  # 使用统一学习率
        print(f"[Config] Using unified lr for gene_lr: {gene_lr}")

    print(f"[Learning Rate Config] Unified LR: {args.lr} (Text LR: {text_lr}, Gene LR: {gene_lr})")
    print(f"[Learning Rate Config] Strategy: {'Unified' if text_lr == gene_lr == args.lr else 'Differential'}")

    # 【已改】分离文本和基因参数
    text_params = []
    gene_params = []
    classifier_params = []

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'clinical_bert' in name:
                text_params.append(param)
            elif any(keyword in name for keyword in ['fc_omic', 'omic', 'pathway_encoder', 'cross_attention', 'query_projection']):
                gene_params.append(param)
            else:
                # 其他参数（如分类器）默认归类到基因组
                classifier_params.append(param)

    # 构建参数组
    param_groups = []
    if text_params:
        param_groups.append({'params': text_params, 'lr': text_lr})
    if gene_params:
        param_groups.append({'params': gene_params, 'lr': gene_lr})
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': gene_lr})

    # 如果没有找到特定参数，使用默认行为
    if not param_groups:
        print("[Warning] No specific parameter groups found, using default lr for all parameters")
        param_groups = [{'params': model.parameters(), 'lr': args.lr}]

    # 根据优化器类型创建优化器
    if args.opt == "adam":
        optimizer = optim.Adam(param_groups, lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(param_groups, lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(param_groups, lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):

    print('\nInit Model...', end=' ')

    # 【修复】优先使用嵌套CV的基因数量
    if hasattr(args, 'nested_cv_omic_dim') and args.nested_cv_omic_dim:
        # 获取当前 fold 的基因数量
        fold = getattr(args, 'k', 0)  # 获取当前 fold
        if fold in args.nested_cv_omic_dim:
            omics_input_dim = args.nested_cv_omic_dim[fold]
            print(f"  [Nested CV] Using gene dimension: {omics_input_dim} (fold {fold})")
        else:
            # 如果找不到，使用第一个可用的
            omics_input_dim = list(args.nested_cv_omic_dim.values())[0]
            print(f"  [Nested CV] Using gene dimension: {omics_input_dim}")
    elif hasattr(args, 'omic_sizes') and args.omic_sizes is not None and len(args.omic_sizes) > 0:
        # 使用数据集中更新后的基因数量
        omics_input_dim = args.omic_sizes[0]
        print(f"  [Nested CV] Using omic_sizes: {omics_input_dim}")
    elif args.type_of_path == "custom":
        import pandas as pd
        # 读取第一行来判断有多少列
        if os.path.exists(args.omics_dir):
            df_temp = pd.read_csv(args.omics_dir, index_col=0, nrows=1)
            # 如果有OS列要减去，如果没有直接取shape[1]
            # 假设你的筛选文件包含 'OS' 列，所以减 1；如果没有OS列，就去掉 '- 1'
            omics_input_dim = df_temp.shape[1] - 1 if 'OS' in df_temp.columns else df_temp.shape[1]
            print(f"  [Auto-Detect] Detected {omics_input_dim} features from custom file.")
        else:
            raise FileNotFoundError(f"Cannot find omics file: {args.omics_dir}")
    elif args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        # 【新增】从命令行参数获取运行模式，默认3（多模态融合）
        ab_model = getattr(args, 'ab_model', 3)
        print(f"🚀 [Init Model] 运行模式: {ab_model} "
              f"({'仅文本' if ab_model == 1 else '仅基因' if ab_model == 2 else '多模态融合'})")

        model_dict = {
             "omic_input_dim" : omics_input_dim,
             "n_stage_classes": getattr(args.dataset_factory, 'n_stage_classes', 4),
             "enable_multitask": args.enable_multitask,
             "multitask_weight": args.multitask_weight,
             "ab_model": ab_model  # 【新增】传递运行模式参数
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = ABMIL(**model_dict)

    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)

    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))
    

    # ----> 新增：计算并打印参数量 <----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('\n' + "="*30)
    print(f"Model: {args.modality}")
    print(f"总参数量 (Total):      {total_params / 1e9:.4f} B")
    print(f"可训练参数 (Trainable): {trainable_params / 1e9:.4f} B")
    print("="*30 + '\n')

    # ----> 新增：极其详细的参数分布 <----
    print("\nDetailed layer-by-layer parameters:")
    for name, module in model.named_modules():
        # 只打印含有参数的底层叶子节点（比如 Linear, Conv2d 等）
        # 如果只想看大块，可以去掉下面的 if 判断
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"Layer: {name:<50} | Params: {params:,}")

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data, enable_multitask=False):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
        - enable_multitask : Boolean
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
        - stage_labels : torch.Tensor (if multitask enabled)
    
    """
    
    stage_labels = None
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, stage_labels = data[2], data[3], data[4], data[5], data[6]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        mask_idx = 6 if not enable_multitask else 7
        if data[mask_idx][0,0] == 1:
            mask = None
        else:
            mask = data[mask_idx].to(device)

        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, stage_labels = data[2], data[3], data[4], data[5], data[7]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, mask, stage_labels = data[7], data[8], data[9], data[10], data[11], data[12]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        mask = mask.to(device)

    elif modality in ["survpath"]:

        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        mask_idx = 6 if not enable_multitask else 7
        if data[mask_idx][0,0] == 1:
            mask = None
        else:
            mask = data[mask_idx].to(device)

        if enable_multitask:
            y_disc, event_time, censor, clinical_data_list, stage_labels = data[2], data[3], data[4], data[5], data[7]
            stage_labels = stage_labels.to(device)
        else:
            y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)

    return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, stage_labels

def _process_data_and_forward(model, modality, device, data, enable_multitask=False):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
        - enable_multitask : Boolean
    
    Returns:
        - out : torch.Tensor or tuple
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
        - stage_labels : torch.Tensor (if multitask enabled)
    
    """
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, stage_labels = _unpack_data(
        modality, device, data, enable_multitask)

    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  

    elif modality == 'survpath':

        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)
        
    else:
        out = model(
            x_text_report = clinical_data_list[-1],
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    # Handle single tensor output
    # if not enable_multitask and len(out.shape) == 1:
    #     out = out.unsqueeze(0)
    
    if enable_multitask:
        return out, y_disc, event_time, censor, clinical_data_list, stage_labels
    else:
        return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list, 
                   all_stage_preds=None, all_stage_labels=None, stage_pred=None, stage_label=None):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
        - all_stage_preds : List (optional)
        - all_stage_labels : List (optional)
        - stage_pred : torch.Tensor (optional)
        - stage_label : torch.Tensor (optional)
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - all_stage_preds : List (optional)
        - all_stage_labels : List (optional)
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    
    if all_stage_preds is not None and stage_pred is not None:
        all_stage_preds.append(stage_pred.detach().cpu().numpy())
        all_stage_labels.append(stage_label.detach().cpu().numpy())
        return all_risk_scores, all_censorships, all_event_times, all_clinical_data, all_stage_preds, all_stage_labels
    
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn, enable_multitask=False, use_debiase=True, use_align=False):
    r"""
    Perform one epoch of training

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class
        - enable_multitask : Boolean

    Returns:
        - c_index : Float
        - total_loss : Float
        - stage_accuracy : Float (if multitask enabled)

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    # ============================================================
    # 【新增】动态检测模型的 use_debias 设置，而不是硬编码
    # ============================================================
    if hasattr(model, 'use_debias'):
        use_debiase = model.use_debias
        # 仅打印一次，避免每个epoch都刷屏
        if not hasattr(model, '_debug_printed'):
            print(f"🔍 [Debug] Detected model.use_debias = {use_debiase}")
            model._debug_printed = True
    else:
        # 仅打印一次
        if not hasattr(model, '_debug_printed'):
            print(f"⚠️  [Warn] Model doesn't have 'use_debias' attribute, using passed value: {use_debiase}")
            model._debug_printed = True


    total_loss = 0.
    total_survival_loss = 0.
    total_stage_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    
    if enable_multitask:
        all_stage_preds = []
        all_stage_labels = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        if enable_multitask:
            output, y_disc, event_time, censor, clinical_data_list, stage_labels = _process_data_and_forward(
                model, modality, device, data, enable_multitask)
            # 【修改】训练时忽略explanation_data（不需要保存）
            if use_debiase:
                if use_align:
                    #survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, align_loss, _ = output
                    # === [修复] 兼容 Multitask 模式 (3返回值 vs 7返回值) ===
                    if len(output) == 7:
                        survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, _ = output
                    elif len(output) == 3:
                        # 只有: 生存预测, 分期预测, 解释数据
                        survival_logits, stage_logits, _ = output
                        logits_text = logit_te = logit_tie = logits_nde = None
                        h = survival_logits
                    else:
                        raise ValueError(f'Unexpected output length: {len(output)}')
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                    # print('align_loss: ', align_loss)
                    # print('loss: ', loss)
                    loss += 0.1 * align_loss.squeeze()
                else:
                    #survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, _ = output
                    
                    if len(output) == 7:
                        survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, _ = output
                    elif len(output) == 3:
                        # 只有: 生存预测, 分期预测, 解释数据
                        survival_logits, stage_logits, _ = output
                        logits_text = logit_te = logit_tie = logits_nde = None
                        h = survival_logits
                    else:
                        raise ValueError(f'Unexpected output length: {len(output)}')
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                total_survival_loss += surv_loss.item()
                total_stage_loss += stage_loss.item()
                stage_pred = torch.argmax(stage_logits, dim=1)
                all_stage_preds.append(stage_pred.detach().cpu().numpy())
                all_stage_labels.append(stage_labels.detach().cpu().numpy())

                h = logit_tie
            else:
                survival_logits, stage_logits, _ = output
                loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, None, None, None)
                total_survival_loss += surv_loss.item()
                total_stage_loss += stage_loss.item()
                h = survival_logits

        else:
            output, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(
                model, modality, device, data, enable_multitask)
            # 【修改】训练时忽略explanation_data（不需要保存）
            if use_debiase:
                if use_align:
                    survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde, align_loss, _ = output
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                    # print('align_loss: ', align_loss)
                    # print('loss: ', loss)
                    loss += 0.1 * align_loss.squeeze()
                else:
                    survival_logits, logits_text, logit_te, logit_tie, logits_nde, _ = output
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, logits_text, logit_te, logits_nde)
                total_survival_loss += surv_loss.item()

                h = logit_tie
            else:
                survival_logits, _ = output
                loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, None, None, None)
                total_survival_loss += surv_loss.item()
                h = survival_logits
            # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        
        h = survival_logits#0115加入，确保h有值
        risk, _ = _calculate_risk(h)

        all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
            all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
            event_time, censor, risk, clinical_data_list)

        total_loss += loss_value 

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (batch_idx % 100) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if enable_multitask:
        # total_survival_loss /= len(loader.dataset)
        # total_stage_loss /= len(loader.dataset)
        
        # # Calculate stage accuracy
        # all_stage_preds = np.concatenate(all_stage_preds, axis=0)
        # all_stage_labels = np.concatenate(all_stage_labels, axis=0)
        
        # # Only calculate accuracy for valid stage labels
        # valid_mask = all_stage_labels != -1
        # if valid_mask.sum() > 0:
        #     stage_accuracy = (all_stage_preds[valid_mask] == all_stage_labels[valid_mask]).mean()
        # else:
        stage_accuracy = 0.0
        
        print('Epoch: {}, train_loss: {:.4f}, survival_loss: {:.4f}, stage_loss: {:.4f}, train_c_index: {:.4f}, stage_acc: {:.4f}'.format(
            epoch, total_loss, total_survival_loss, total_stage_loss, c_index, stage_accuracy))
        
        return c_index, total_loss, stage_accuracy
    else:
        print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
        return c_index, total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None, enable_multitask=False, use_debiase=True, use_align=False, save_visualizations=True, dataset_name=None, fold_idx=None):
    r"""
    Run a validation loop on the trained model

    Args:
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function class
        - survival_train : np.array
        - enable_multitask : Boolean
        - use_debiase : Boolean
        - use_align : Boolean
        - save_visualizations : Boolean (控制是否保存可视化结果，默认True)
        - dataset_name : String (数据集名称，如'TCGA_BRCA'，用于组织文件夹)
        - fold_idx : Int (Fold编号，用于组织文件夹)

    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float
        - stage_accuracy : Float (if multitask enabled)

    """

    # ============================================================
    # 【新增】动态检测模型的 use_debias 设置，而不是硬编码
    # ============================================================
    if hasattr(model, 'use_debias'):
        use_debiase = model.use_debias
        # 仅打印一次，避免每个epoch都刷屏
        if not hasattr(model, '_debug_printed'):
            print(f"🔍 [Debug] Detected model.use_debias = {use_debiase}")
            model._debug_printed = True
    else:
        # 仅打印一次
        if not hasattr(model, '_debug_printed'):
            print(f"⚠️  [Warn] Model doesn't have 'use_debias' attribute, using passed value: {use_debiase}")
            model._debug_printed = True

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.
    total_survival_loss = 0.
    total_stage_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []
    
    if enable_multitask:
        all_stage_preds = []
        all_stage_labels = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:

            if enable_multitask:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, stage_labels = _unpack_data(
                    modality, device, data, enable_multitask)

                if modality in ["coattn", "coattn_motcat"]:  
                    output = model(
                        x_path=data_WSI, 
                        x_omic1=data_omics[0], 
                        x_omic2=data_omics[1], 
                        x_omic3=data_omics[2], 
                        x_omic4=data_omics[3], 
                        x_omic5=data_omics[4], 
                        x_omic6=data_omics[5]
                    )  
                elif modality == "survpath":
                    input_args = {"x_path": data_WSI.to(device)}
                    for i in range(len(data_omics)):
                        input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                    input_args["return_attn"] = False
                    output = model(**input_args)
                else:
                    output = model(
                        x_text_report = clinical_data_list[-1], # last item is text report
                        data_omics = data_omics,
                        data_WSI = data_WSI,
                        mask = mask
                        )
                # 【修改】接收explanation_data
                if use_debiase:
                    if use_align:
                        #survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss, explanation_data = output
                        if len(output) == 7:
                            survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, explanation_data = output
                        elif len(output) == 3:
                            # 兼容: 只有生存预测、分期预测和解释数据
                            survival_logits, stage_logits, explanation_data = output
                            logits_text = logit_te = logits_tie = logits_nde = None
                        else:
                            # 兼容: 只有生存预测和解释数据 (非Multitask)
                            survival_logits = output[0]
                            stage_logits = None
                            explanation_data = output[-1]
                            logits_text = logit_te = logits_tie = logits_nde = None
                    else:
                        #survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, explanation_data = output
                        if len(output) == 7:
                            survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, explanation_data = output
                        elif len(output) == 3:
                            # 兼容: 只有生存预测、分期预测和解释数据
                            survival_logits, stage_logits, explanation_data = output
                            logits_text = logit_te = logits_tie = logits_nde = None
                        else:
                            # 兼容: 只有生存预测和解释数据 (非Multitask)
                            survival_logits = output[0]
                            stage_logits = None
                            explanation_data = output[-1]
                            logits_text = logit_te = logits_tie = logits_nde = None

                    # h = survival_logits
                    h = logits_tie # 改
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te,logits_nde)
                    total_survival_loss += surv_loss.item()
                    total_stage_loss += stage_loss.item()

                    # Stage predictions
                    stage_pred = torch.argmax(stage_logits, dim=1)
                    all_stage_preds.append(stage_pred.detach().cpu().numpy())
                    all_stage_labels.append(stage_labels.detach().cpu().numpy())
                else:
                    survival_logits, stage_logits, explanation_data = output
                    h = survival_logits
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, None, None, None)
                    total_survival_loss += surv_loss.item()
                    total_stage_loss += stage_loss.item()

                # 【修改】提取case_id并保存可视化解释（只保存测试集）
                if save_visualizations:
                    slide_id = slide_ids.values[count]
                    case_id = slide_id[:12]
                    _save_visualization_explanations(case_id, explanation_data, model,
                                                    dataset_name=dataset_name, fold_idx=fold_idx)
            else:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, _ = _unpack_data(
                    modality, device, data, enable_multitask)

                if modality in ["coattn", "coattn_motcat"]:  
                    h = model(
                        x_path=data_WSI, 
                        x_omic1=data_omics[0], 
                        x_omic2=data_omics[1], 
                        x_omic3=data_omics[2], 
                        x_omic4=data_omics[3], 
                        x_omic5=data_omics[4], 
                        x_omic6=data_omics[5]
                    )  
                elif modality == "survpath":
                    input_args = {"x_path": data_WSI.to(device)}
                    for i in range(len(data_omics)):
                        input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                    input_args["return_attn"] = False
                    output = model(**input_args)
                else:
                    output = model(
                        x_text_report = clinical_data_list[-1],
                        data_omics = data_omics,
                        data_WSI = data_WSI,
                        mask = mask
                        )

                # 【修改】接收explanation_data
                if use_debiase:
                    if use_align:
                        survival_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss, explanation_data = output
                    else:
                        survival_logits, logits_text, logit_te, logits_tie, logits_nde, explanation_data = output

                    # h = survival_logits
                    h = logits_tie # 改
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, logits_text, logit_te,logits_nde)
                    total_survival_loss += surv_loss.item()
                else:
                    survival_logits, explanation_data = output
                    h = survival_logits
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, None, y_disc, event_time, censor, None, None, None, None)
                    total_survival_loss += surv_loss.item()

                # 【修改】提取case_id并保存可视化解释（只保存测试集）
                if save_visualizations:
                    slide_id = slide_ids.values[count]
                    case_id = slide_id[:12]
                    _save_visualization_explanations(case_id, explanation_data, model,
                                                    dataset_name=dataset_name, fold_idx=fold_idx)

            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

            #0115加
            h = survival_logits
            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            
            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(
                all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
                event_time, censor, risk, clinical_data_list)
            
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    if enable_multitask:
        total_survival_loss /= len(loader.dataset)
        total_stage_loss /= len(loader.dataset)
        
        # Calculate stage accuracy
        # all_stage_preds = np.concatenate(all_stage_preds, axis=0)
        # all_stage_labels = np.concatenate(all_stage_labels, axis=0)
        
        # # Only calculate accuracy for valid stage labels
        # valid_mask = all_stage_labels != -1
        # if valid_mask.sum() > 0:
        #     stage_accuracy = (all_stage_preds[valid_mask] == all_stage_labels[valid_mask]).mean()
        # else:
        #     stage_accuracy = 0.0
        
        return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss, 0
    else:
        return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss, 0


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs


    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _step(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
        - val_stage_accuracy : Float (if multitask enabled)
    """

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    best_c_index = 0.0
    best_model_path = os.path.join(args.results_dir, f'best_model_fold_{cur}.pth')
    
    for epoch in range(args.max_epochs):
        if args.enable_multitask:
            _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args.enable_multitask)
        else:
            _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)

        if args.enable_multitask:
            results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
                args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask)
            # if val_cindex > best_c_index and best_c_index < 1.0:
            #     best_c_index = val_cindex
        else:
            results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
                args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask)
        if val_cindex > best_c_index and best_c_index < 1.0:
            best_c_index = val_cindex
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            #     'best_c_index': best_c_index,
            #     'fold': cur,
            #     'args': args
            # }, best_model_path)
            # print('model saved!')
        print('Best Val c-index: {:.4f}, Stage accuracy: {:.4f}'.format(best_c_index, val_stage_accuracy))

    return results_dict, (best_c_index, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy)

def _train_val(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
        - val_stage_accuracy : Float (if multitask enabled)
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    results_dict, metrics = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    return results_dict, metrics


def _step_with_train_test_results(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    """
    修改后的_step函数，同时获取训练集和测试集结果
    """
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    # 【修改】初始值从0.0改为-1.0，确保第一个epoch能保存结果
    best_c_index = -1.0
    best_test_results = None
    best_train_results = None
    best_model_path = os.path.join(args.results_dir, f'best_model_fold_{cur}.pth')
    
    # 【修改】从args中获取数据集名称（转换为大写，如tcga_brca -> TCGA_BRCA）
    dataset_name = args.study.upper().replace('_', '_')

    for epoch in range(args.max_epochs):
        # 训练
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args.enable_multitask)

        # 【修改】在测试集上评估，传递dataset_name和fold_idx
        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask,
            save_visualizations=True, dataset_name=dataset_name, fold_idx=cur)

        # 【修改】第一个epoch或val_cindex更大时强制保存
        if epoch == 0 or val_cindex > best_c_index:
            best_c_index = val_cindex
            best_test_results = results_dict.copy()

            print(f"  🔄 同时获取训练集结果...")
            # 【修改】训练集不保存可视化结果（save_visualizations=False）
            train_results_dict, _, _, _, _, _, _, _ = _summary(
                args.dataset_factory, model, args.modality, train_loader, loss_fn, all_survival, args.enable_multitask,
                use_debiase=True, use_align=False, save_visualizations=False,
                dataset_name=dataset_name, fold_idx=cur)
            best_train_results = train_results_dict.copy()
            
            # 保存模型
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(), 
            #     'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            #     'best_c_index': best_c_index,
            #     'fold': cur,
            #     'args': args
            # }, best_model_path)
            
        print('Best Val c-index: {:.4f}'.format(best_c_index))
    
    return best_test_results, best_train_results, (best_c_index, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy)

def generate_km_plot_and_save_data(train_results, test_results, dataset_name, save_dir=None):
    """
    Merges train and test results to plot a single Kaplan-Meier curve for the full cohort,
    and saves the underlying data to a CSV file for future use.
    """
    print("🎨 Generating Kaplan-Meier plot for the full cohort...")

    # 【新增】空值检查：如果传入的结果为None，直接返回默认值
    if train_results is None or test_results is None:
        print("⚠️  警告: train_results 或 test_results 为 None，跳过KM图生成")
        sample_counts = {'high_risk': 0, 'low_risk': 0}
        return 1.0, sample_counts, None  # 返回p-value=1.0表示无显著差异

    # 【新增】将整个绘图逻辑包裹在try...except中，防止matplotlib崩溃
    try:
        # 1. Merge training and testing data
        all_results = {**train_results, **test_results}

        # 2. Convert results to a DataFrame for saving and plotting
        case_ids = list(all_results.keys())
        times = [data['time'] for data in all_results.values()]
        risks = [data['risk'] for data in all_results.values()]
        censorships = [data['censorship'] for data in all_results.values()]

        df_for_plotting = pd.DataFrame({
            'case_id': case_ids,
            'time': times,
            'risk': risks,
            'censorship': censorships,
            'event': [1 - c for c in censorships]
        })

        # Stratify patients into high- and low-risk groups based on the median risk score
        median_risk = np.median(df_for_plotting['risk'])
        df_for_plotting['risk_group'] = np.where(df_for_plotting['risk'] >= median_risk, 'high risk', 'low risk')

        # 3. Save the plotting data to a CSV file
        data_save_path = None
        if save_dir:
            data_save_path = os.path.join(save_dir, f'{dataset_name}_km_plot_data.csv')
            df_for_plotting.to_csv(data_save_path, index=False)
            print(f"📁 KM plot data saved to: {data_save_path}")

        # 4. Create the Kaplan-Meier plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Separate data by risk group
        high_risk_data = df_for_plotting[df_for_plotting['risk_group'] == 'high risk']
        low_risk_data = df_for_plotting[df_for_plotting['risk_group'] == 'low risk']

        # === [修复] 检查是否有空组，避免KM图生成失败 ===
        if len(high_risk_data) == 0 or len(low_risk_data) == 0:
            print(f"⚠️  警告: 风险分组不均衡 (high_risk: {len(high_risk_data)}, low_risk: {len(low_risk_data)})")
            print(f"⚠️  所有样本risk值相同或接近，跳过KM图生成")

            # 检查risk值分布
            unique_risks = df_for_plotting['risk'].nunique()
            print(f"⚠️  唯一risk值数量: {unique_risks}")

            ax.text(0.5, 0.5, f'无法生成KM图\n风险分组不均衡\nhigh_risk: {len(high_risk_data)}\nlow_risk: {len(low_risk_data)}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{dataset_name}\n(KM plot skipped - unbalanced groups)', fontsize=14)

            if save_dir:
                figure_save_path = os.path.join(save_dir, f'{dataset_name}_km_plot_skipped.png')
                plt.savefig(figure_save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"🖼️ 占位图已保存到: {figure_save_path}")

            plt.close()

            sample_counts = {'high_risk': len(high_risk_data), 'low_risk': len(low_risk_data)}
            return 1.0, sample_counts, data_save_path  # 返回p_value=1.0表示无显著差异
        # ================================================

        # Fit data to Kaplan-Meier estimator
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_low.fit(low_risk_data['time'], low_risk_data['event'], label=f"low risk (n={len(low_risk_data)})")
        kmf_high.fit(high_risk_data['time'], high_risk_data['event'], label=f"high risk (n={len(high_risk_data)})")

        # Plot the survival functions
        kmf_low.plot_survival_function(ax=ax, color='green', show_censors=True)
        kmf_high.plot_survival_function(ax=ax, color='red', show_censors=True)

        # Perform log-rank test
        results = logrank_test(
            high_risk_data['time'], low_risk_data['time'],
            event_observed_A=high_risk_data['event'], event_observed_B=low_risk_data['event']
        )
        p_value = results.p_value

        # Configure plot aesthetics
        ax.set_xlabel('Timeline (month)', fontsize=12)
        ax.set_ylabel('Cumulative proportion surviving', fontsize=12)
        ax.set_title(f'{dataset_name}\n(P-value: {p_value:.2e})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the plot as an image file
        if save_dir:
            figure_save_path = os.path.join(save_dir, f'{dataset_name}_km_plot.png')
            plt.savefig(figure_save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"🖼️ KM plot image saved to: {figure_save_path}")

        plt.close()  # 使用plt.close()替代plt.show()避免阻塞

        sample_counts = {'high_risk': len(high_risk_data), 'low_risk': len(low_risk_data)}

        return p_value, sample_counts, data_save_path

    except Exception as e:
        # 【新增】捕获matplotlib等绘图相关异常，防止程序崩溃
        print(f"❌ 绘图过程中发生错误: {e}")
        print("⚠️  已跳过KM图生成，返回默认值")

        sample_counts = {'high_risk': 0, 'low_risk': 0}
        return 1.0, sample_counts, None

# Modified main training function to call the new plotting function
def _train_val_with_km_plots(datasets, cur, args):
    """
    Modified training and validation function that includes KM plot generation.
    """
    # Get training and validation data splits
    train_split, val_split = _get_splits(datasets, cur, args)
    
    # Initialize components
    loss_fn = _init_loss_function(args)
    model = _init_model(args)
    optimizer = _init_optim(args, model)
    train_loader, val_loader = _init_loaders(args, train_split, val_split)
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)
    
    # Run training and get results for both training and validation sets
    test_results, train_results, metrics = _step_with_train_test_results(
        cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)
    
    # Generate and save the KM plot and its data
    dataset_name = f"{args.study.upper()}-Fold{cur}"
    p_value, sample_counts, km_data_path = generate_km_plot_and_save_data(
        train_results, test_results, dataset_name, args.results_dir)
    
    # Save all results from the fold into a single pickle file
    fold_results = {
        'test_results': test_results,
        'train_results': train_results,
        'metrics': metrics,
        'km_p_value': p_value,
        'km_sample_counts': sample_counts,
        'km_plot_data_path': km_data_path,
    }
    
    save_path = os.path.join(args.results_dir, f'fold_{cur}_complete_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(fold_results, f)
    
    return fold_results