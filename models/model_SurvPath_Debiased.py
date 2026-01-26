import torch
import numpy as np
import torch.nn as nn
from torch import nn
from einops import reduce
from torch.nn import ReLU
from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb
import math
import pandas as pd

def exists(val):
    return val is not None

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


class SurvPathDebiased(nn.Module):
    def __init__(
        self, 
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names=[],
        debias_type="modality",  # "modality", "pathway", "both"
        debias_strength=1.0,
        device="cuda"
        ):
        super(SurvPathDebiased, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.debias_type = debias_type
        self.debias_strength = debias_strength
        self.device = device

        #---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> omics props
        self.init_per_path_model(omic_sizes)

        #---> cross attention props
        self.identity = nn.Identity() # use this layer to calculate ig
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways=self.num_pathways
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # when both top and bottom blocks 
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        
        # Counterfactual components
        if self.debias_type in ["modality", "both"]:
            # For modality debiasing: WSI-only and Omics-only branches
            self.wsi_only_branch = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/2)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/2), self.num_classes)
            )
            
            self.omics_only_branch = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/2)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/2), self.num_classes)
            )
        
        if self.debias_type in ["pathway", "both"]:
            # For pathway debiasing: learnable uniform distribution parameter
            self.uniform_pathway_param = nn.Parameter(torch.zeros(self.num_pathways, self.wsi_projection_dim))
            nn.init.xavier_normal_(self.uniform_pathway_param)
        
        self.single_modality_logit = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/2)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim/2), self.num_classes)
        )

        
    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)    

    def create_counterfactual_inputs(self, wsi_embed, h_omic_bag, mode="wsi_only"):
        """
        Create counterfactual inputs for debiasing
        
        Args:
            wsi_embed: WSI embeddings
            h_omic_bag: pathway embeddings
            mode: "wsi_only", "omics_only", "uniform_pathway"
        """
        if mode == "wsi_only":
            # Block omics information, use uniform distribution
            uniform_omics = self.uniform_pathway_param.unsqueeze(0).expand(
                h_omic_bag.shape[0], -1, -1
            )
            return torch.cat([uniform_omics, wsi_embed], dim=1)
        
        elif mode == "omics_only":
            # Block WSI information, use zeros or uniform distribution
            uniform_wsi = torch.zeros_like(wsi_embed)
            return torch.cat([h_omic_bag, uniform_wsi], dim=1)
        
        elif mode == "uniform_pathway": # 需要修改
            # Use uniform pathway distribution instead of real pathways
            uniform_omics = self.uniform_pathway_param.unsqueeze(0).expand(
                h_omic_bag.shape[0], -1, -1
            )
            return torch.cat([uniform_omics, wsi_embed], dim=1)
        
        else:
            # Original input
            return torch.cat([h_omic_bag, wsi_embed], dim=1)

    def forward_with_counterfactual(self, tokens, single_mod=False, mask=None):
        """
        Forward pass with counterfactual reasoning
        """
        # Original forward pass
        if not single_mod:
            # print("==="*50)
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)
            mm_embed = self.feed_forward(mm_embed)
            mm_embed = self.layer_norm(mm_embed)
            
            # Aggregate embeddings
            paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
            paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

            wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
            wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

            embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
            logits = self.to_logits(embedding)
            
            return logits, paths_postSA_embed, wsi_postSA_embed
        else:
            # Single modality forward pass
            single_modelity = tokens[:, self.num_pathways:, :]
            single_logit = self.single_modality_logit(single_modality)
            return single_logit, single_modelity, None

    def forward(self, **kwargs):
        wsi = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways+1)]
        # print("x_omic: ", x_omic)
        mask = None
        return_attn = kwargs.get("return_attn")
        training = kwargs.get("training", True)
        
        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)]
        # print("h_omic: ", h_omic)
        # print("h_omic[0].shape: ", h_omic[0].shape)
        h_omic_bag = torch.stack(h_omic).unsqueeze(0) # [1, 331, 256]

        #---> project wsi to smaller dimension
        wsi_embed = self.wsi_projection_net(wsi)

        # Original tokens
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)
        
        # Main forward pass
        main_logits, paths_embed, wsi_embed_processed = self.forward_with_counterfactual(tokens, mask)
        
        if not training:
            return main_logits
        
        # Counterfactual forward passes for debiasing
        counterfactual_results = {}
        
        if self.debias_type in ["modality", "both"]:
            # WSI-only counterfactual
            wsi_only_tokens = self.create_counterfactual_inputs(wsi_embed, h_omic_bag, "wsi_only")
            wsi_only_logits, _, _ = self.forward_with_counterfactual(wsi_only_tokens, mask)
            counterfactual_results["wsi_only"] = wsi_only_logits
            
            # Omics-only counterfactual  
            omics_only_tokens = self.create_counterfactual_inputs(wsi_embed, h_omic_bag, "omics_only")
            omics_only_logits, _, _ = self.forward_with_counterfactual(omics_only_tokens, mask)
            counterfactual_results["omics_only"] = omics_only_logits
        
        if self.debias_type in ["pathway", "both"]:
            # Uniform pathway counterfactual
            uniform_pathway_tokens = self.create_counterfactual_inputs(wsi_embed, h_omic_bag, "uniform_pathway")
            uniform_pathway_logits, _, _ = self.forward_with_counterfactual(uniform_pathway_tokens, mask)
            counterfactual_results["uniform_pathway"] = uniform_pathway_logits
        
        if return_attn:
            # For attention visualization (simplified)
            mm_embed = self.cross_attender(x=tokens, mask=mask, return_attention=True)
            return main_logits, counterfactual_results, mm_embed
        else:
            return main_logits, counterfactual_results

    def compute_debiased_prediction(self, main_logits, counterfactual_results):
        """
        Compute debiased prediction using counterfactual inference
        Following CF-VQA: TIE = TE - NDE
        """
        if self.debias_type == "modality":
            # Total effect (TE) = main prediction
            # Natural direct effect (NDE) = average of single modality predictions
            nde = (counterfactual_results["wsi_only"] + counterfactual_results["omics_only"]) / 2
            tie = main_logits - self.debias_strength * nde
            return tie
            
        elif self.debias_type == "pathway":
            # Reduce pathway bias
            nde = counterfactual_results["uniform_pathway"]
            tie = main_logits - self.debias_strength * nde
            return tie
            
        elif self.debias_type == "both":
            # Combined debiasing
            modality_nde = (counterfactual_results["wsi_only"] + counterfactual_results["omics_only"]) / 2
            pathway_nde = counterfactual_results["uniform_pathway"]
            combined_nde = (modality_nde + pathway_nde) / 2
            tie = main_logits - self.debias_strength * combined_nde
            return tie
        
        else:
            return main_logits

    def captum(self, omics_list, wsi):
        """Captum method for interpretability (simplified)"""
        # Implementation similar to original SurvPath
        mask = None
        return_attn = False
        
        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(omics_list)]
        h_omic_bag = torch.stack(h_omic, dim=1)
        
        #---> project wsi to smaller dimension
        wsi_embed = self.wsi_projection_net(wsi)

        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)

        main_logits, _, _ = self.forward_with_counterfactual(tokens, mask)

        hazards = torch.sigmoid(main_logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        return risk