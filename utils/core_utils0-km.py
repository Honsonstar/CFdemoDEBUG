from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
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
            loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
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
    
    # 伪代码示例
    # optimizer = torch.optim.AdamW([
    #     {'params': model.fc_omic.parameters(), 'lr': 1e-4},
    #     {'params': model.survival_classifier.parameters(), 'lr': 1e-4},
    #     {'params': model.clinical_bert_model.parameters(), 'lr': 2e-5} # 为BERT设置一个更小的学习率
    # ], weight_decay=0.01)

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
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

        model_dict = {
             "omic_input_dim" : omics_input_dim,
             "n_stage_classes": getattr(args.dataset_factory, 'n_stage_classes', 4),
             "enable_multitask": args.enable_multitask,
             "multitask_weight": args.multitask_weight
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

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn, enable_multitask=False, use_debiase=False):
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
            if use_debiase:
                survival_logits, stage_logits, logits_text, logit_te, logit_tie, logits_nde = output
                loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, logits_text, logit_te, logits_nde)
                total_survival_loss += surv_loss.item()
                total_stage_loss += stage_loss.item()
                stage_pred = torch.argmax(stage_logits, dim=1)
                all_stage_preds.append(stage_pred.detach().cpu().numpy())
                all_stage_labels.append(stage_labels.detach().cpu().numpy())
                
                h = logit_tie
            else:
                survival_logits, stage_logits = output
                loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, None, None, None)
                total_survival_loss += surv_loss.item()
                total_stage_loss += stage_loss.item()
                h = survival_logits

        else:
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(
                model, modality, device, data, enable_multitask)
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        
        loss_value = loss.item()
        loss = loss / y_disc.shape[0]
        
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

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None, enable_multitask=False, use_debiase=False):
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
                if use_debiase:
                    survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde = output
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
                    survival_logits, stage_logits = output
                    h = survival_logits
                    loss, surv_loss, stage_loss = loss_fn(survival_logits, stage_logits, y_disc, event_time, censor, stage_labels, None, None, None)
                    total_survival_loss += surv_loss.item()
                    total_stage_loss += stage_loss.item()
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
                    h = model(**input_args)
                else:
                    h = model(
                        data_omics = data_omics, 
                        data_WSI = data_WSI, 
                        mask = mask
                        )
                
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]

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
        return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


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
    
    for epoch in range(args.max_epochs):
        # if args.enable_multitask:
        _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn, args.enable_multitask)
        # else:
        #     _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)

        # if args.enable_multitask:
        results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy = _summary(
            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, args.enable_multitask)
        if val_cindex > best_c_index and best_c_index < 1.0:
            best_c_index = val_cindex
            
        print('Best Val c-index: {:.4f}, Stage accuracy: {:.4f}'.format(best_c_index, val_stage_accuracy))
            
        # return results_dict, (best_c_index, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss, val_stage_accuracy)
        # else:
        #     results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(
        #         args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
            
        #     print('Final Val c-index: {:.4f}'.format(val_cindex))
            
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
    # 绘制KM曲线
    dataset_name = args.study.upper()  # 例如 'TCGA-BLCA'
    save_path = os.path.join(args.results_dir, f'km_curve_fold_{cur}.png')
    
    p_value = plot_km_curve_from_results(
        patient_results=results_dict,
        dataset_name=f'{dataset_name} - Fold {cur}',
        save_path=save_path
    )
    
    print(f"Kaplan-Meier analysis completed. P-value: {p_value:.3e}")
    
    return results_dict, metrics


# ===== 新增：文本归因分析相关函数 =====

def _inference_with_text_attribution(model, modality, text_report, data_omics=None, 
                                    attribution_methods=['attention', 'integrated_gradients'], 
                                    save_visualizations=True, output_dir='./attribution_results'):
    """
    对单个样本进行推理并获取文本归因分析结果
    
    Args:
        model: 训练好的模型
        modality: 模型类型
        text_report: 文本报告
        data_omics: omics数据 (可选)
        attribution_methods: 归因方法列表
        save_visualizations: 是否保存可视化结果
        output_dir: 输出目录
    
    Returns:
        dict: 包含预测结果和归因分析结果
    """
    import os
    from datetime import datetime
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # 创建输出目录
    if save_visualizations:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备数据
    if data_omics is None:
        # 如果没有提供omics数据，创建零张量
        if hasattr(model, 'size_dict_omic'):
            omic_dim = model.size_dict_omic['small'][0]
        else:
            omic_dim = 256  # 默认维度
        data_omics = torch.zeros(1, omic_dim).to(device)
    else:
        data_omics = data_omics.to(device)
    
    # 基础预测
    with torch.no_grad():
        if modality == "snn":
            prediction = model(
                data_omics=data_omics,
                x_text_report=[text_report]
            )
        else:
            # 适配其他模型类型
            prediction = model(
                data_omics=data_omics,
                data_WSI=None,
                mask=None,
                x_text_report=[text_report]
            )
    
    # 计算风险分数
    risk, survival_probs = _calculate_risk(prediction)
    predicted_class = torch.argmax(prediction, dim=-1).item()
    
    print(f"Prediction Results:")
    print(f"Predicted Class: {predicted_class}")
    print(f"Risk Score: {risk[0]:.4f}")
    print(f"Survival Probabilities: {survival_probs[0]}")
    
    # 归因分析
    attribution_results = {}
    
    for method in attribution_methods:
        print(f"\nPerforming {method} attribution analysis...")
        
        try:
            attribution_result = model.get_text_attribution_scores(
                text_report, 
                method=method, 
                target_class=predicted_class
            )
            
            attribution_results[method] = attribution_result
            
            # 可视化重要token
            top_tokens = model.visualize_text_attribution(
                attribution_result, 
                top_k=20
            )
            
            # 保存HTML可视化
            if save_visualizations:
                html_path = os.path.join(
                    output_dir, 
                    f"attribution_{method}_{timestamp}.html"
                )
                model.visualize_text_attribution(
                    attribution_result,
                    save_path=html_path
                )
            
        except Exception as e:
            print(f"Error in {method} attribution: {str(e)}")
            attribution_results[method] = None
    
    return {
        'prediction': {
            'logits': prediction.cpu().numpy(),
            'predicted_class': predicted_class,
            'risk_score': risk[0],
            'survival_probabilities': survival_probs[0]
        },
        'attribution_results': attribution_results,
        'text_report': text_report
    }

def _batch_inference_with_attribution(model, modality, test_loader, 
                                     attribution_methods=['attention'], 
                                     save_results=True, output_dir='./batch_attribution_results'):
    """
    对测试集进行批量推理和归因分析
    
    Args:
        model: 训练好的模型
        modality: 模型类型
        test_loader: 测试数据加载器
        attribution_methods: 归因方法列表
        save_results: 是否保存结果
        output_dir: 输出目录
    
    Returns:
        list: 每个样本的结果列表
    """
    import os
    import pickle
    from datetime import datetime
    from tqdm import tqdm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    all_results = []
    
    for batch_idx, data in enumerate(tqdm(test_loader, desc="Processing samples")):
        try:
            # 解包数据
            data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, _ = _unpack_data(
                modality, device, data, enable_multitask=False
            )
            
            text_report = clinical_data_list[-1]  # 最后一个是文本报告
            
            # 进行单个样本的归因分析
            sample_result = _inference_with_text_attribution(
                model=model,
                modality=modality,
                text_report=text_report,
                data_omics=data_omics,
                attribution_methods=attribution_methods,
                save_visualizations=False  # 批量处理时不保存个别可视化
            )
            
            # 添加真实标签信息
            sample_result['ground_truth'] = {
                'y_disc': y_disc.cpu().numpy(),
                'event_time': event_time.cpu().numpy(),
                'censor': censor.cpu().numpy()
            }
            
            sample_result['batch_idx'] = batch_idx
            all_results.append(sample_result)
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    # 保存批量结果
    if save_results:
        results_path = os.path.join(output_dir, f"batch_attribution_results_{timestamp}.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Batch results saved to: {results_path}")
        
        # 保存统计摘要
        _save_attribution_summary(all_results, output_dir, timestamp)
    
    return all_results

def _save_attribution_summary(results, output_dir, timestamp):
    """
    保存归因分析的统计摘要
    
    Args:
        results: 批量归因分析结果
        output_dir: 输出目录
        timestamp: 时间戳
    """
    import pandas as pd
    import numpy as np
    
    summary_data = []
    
    for i, result in enumerate(results):
        if result['attribution_results']:
            for method, attribution in result['attribution_results'].items():
                if attribution is not None:
                    tokens = attribution['tokens']
                    scores = attribution['scores']
                    
                    # 过滤特殊token
                    filtered_tokens = []
                    filtered_scores = []
                    for token, score in zip(tokens, scores):
                        if token not in ['[CLS]', '[SEP]', '[PAD]']:
                            filtered_tokens.append(token)
                            filtered_scores.append(score)
                    
                    if filtered_scores:
                        summary_data.append({
                            'sample_id': i,
                            'method': method,
                            'predicted_class': result['prediction']['predicted_class'],
                            'risk_score': result['prediction']['risk_score'],
                            'num_tokens': len(filtered_tokens),
                            'max_attribution': max(filtered_scores),
                            'min_attribution': min(filtered_scores),
                            'mean_attribution': np.mean(filtered_scores),
                            'std_attribution': np.std(filtered_scores),
                            'top_5_tokens': ', '.join([
                                f"{token}({score:.3f})" 
                                for token, score in sorted(
                                    zip(filtered_tokens, filtered_scores), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True
                                )[:5]
                            ])
                        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f"attribution_summary_{timestamp}.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"Attribution summary saved to: {summary_path}")
        
        # 打印一些统计信息
        print("\nAttribution Analysis Summary:")
        print("=" * 50)
        for method in df_summary['method'].unique():
            method_data = df_summary[df_summary['method'] == method]
            print(f"\n{method.upper()}:")
            print(f"  Samples analyzed: {len(method_data)}")
            print(f"  Mean attribution range: {method_data['max_attribution'].mean():.4f} to {method_data['min_attribution'].mean():.4f}")
            print(f"  Average std: {method_data['std_attribution'].mean():.4f}")

def run_attribution_analysis_example(model, test_loader, modality):
    """
    运行归因分析的示例函数
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器  
        modality: 模型类型
    """
    print("Starting Attribution Analysis...")
    
    # 对单个样本进行详细分析
    sample_data = next(iter(test_loader))
    data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, _ = _unpack_data(
        modality, torch.device("cuda" if torch.cuda.is_available() else "cpu"), sample_data
    )
    
    text_report = clinical_data_list[-1]
    print(f"\nAnalyzing sample text: {text_report[:200]}...")
    
    # 单样本详细分析
    single_result = _inference_with_text_attribution(
        model=model,
        modality=modality,
        text_report=text_report,
        data_omics=data_omics,
        attribution_methods=['attention', 'integrated_gradients', 'saliency'],
        save_visualizations=True,
        output_dir='./single_sample_attribution'
    )
    
    # 批量分析 (可选，如果数据集较小)
    print("\nRunning batch attribution analysis...")
    batch_results = _batch_inference_with_attribution(
        model=model,
        modality=modality,
        test_loader=test_loader,
        attribution_methods=['attention'],  # 只用attention方法进行批量分析，速度更快
        save_results=True,
        output_dir='./batch_attribution_results'
    )
    
    print("\nAttribution analysis complete!")
    return single_result, batch_results
def plot_km_curve_from_results(patient_results, dataset_name, save_path=None):
    """
    从模型结果绘制Kaplan-Meier曲线
    
    Args:
        patient_results: _summary函数返回的患者结果字典
        dataset_name: 数据集名称
        save_path: 保存路径
    
    Returns:
        p_value: Log-rank检验的p值
    """
    # 提取数据
    case_ids = list(patient_results.keys())
    times = [patient_results[case_id]["time"] for case_id in case_ids]
    risks = [patient_results[case_id]["risk"] for case_id in case_ids]
    censorships = [patient_results[case_id]["censorship"] for case_id in case_ids]
    
    # 转换为DataFrame
    df = pd.DataFrame({
        'case_id': case_ids,
        'time': times,
        'risk': risks,
        'censorship': censorships
    })
    
    # 按风险评分的中位数分组
    median_risk = np.median(df['risk'])
    df['risk_group'] = np.where(df['risk'] >= median_risk, 'High Risk', 'Low Risk')
    df['event'] = 1 - df['censorship']  # 转换删失状态
    
    # 分别拟合高风险组和低风险组
    high_risk_data = df[df['risk_group'] == 'High Risk']
    low_risk_data = df[df['risk_group'] == 'Low Risk']
    
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    kmf_high.fit(high_risk_data['time'], high_risk_data['event'], label='High Risk')
    kmf_low.fit(low_risk_data['time'], low_risk_data['event'], label='Low Risk')
    
    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 绘制生存曲线
    ax = kmf_high.plot_survival_function(color='red', linewidth=2.5, ci_show=False)
    kmf_low.plot_survival_function(ax=ax, color='blue', linewidth=2.5, ci_show=False)
    
    # Log-rank检验
    results = logrank_test(high_risk_data['time'], low_risk_data['time'], 
                          high_risk_data['event'], low_risk_data['event'])
    p_value = results.p_value
    
    # 设置图形属性
    plt.xlabel('Time (Years)', fontsize=12, fontweight='bold')
    plt.ylabel('Survival Probability', fontsize=12, fontweight='bold')
    plt.title(f'{dataset_name}\nKaplan-Meier Survival Curve\nLog-Rank P-value = {p_value:.3e}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend([f'High Risk (n={len(high_risk_data)})', 
                f'Low Risk (n={len(low_risk_data)})'], 
               loc='upper right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return p_value