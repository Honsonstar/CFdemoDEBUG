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
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class SurvPathCounterfactual(nn.Module):
    def __init__(
        self, 
        omic_sizes=[100, 200, 300, 400, 500, 600],
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names = [],
        counterfactual_lambda=1.0,
        ):
        super(SurvPathCounterfactual, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.counterfactual_lambda = counterfactual_lambda

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
        self.identity = nn.Identity()
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # Regular prediction head (total effect)
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
        
        # Counterfactual prediction head (direct genomic effect)
        # This uses only genomic features without WSI interaction
        self.counterfactual_head = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
        )
        
        # Learnable parameter for no-treatment condition (similar to VQA paper)
        self.no_treatment_param = nn.Parameter(torch.randn(1))
        
    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)    
    
    def forward(self, **kwargs):
        wsi = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
        mask = None
        return_attn = kwargs.get("return_attn", False)
        use_counterfactual = kwargs.get("use_counterfactual", True) 
        
        #---> get pathway embeddings 
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)

        #---> project wsi to smaller dimension
        wsi_embed = self.wsi_projection_net(wsi)

        # Total Effect (TE): Regular forward pass with both modalities
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(
                x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        #---> feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        #---> aggregate for total effect
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed_total = torch.mean(paths_postSA_embed, dim=1)
        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed_total = torch.mean(wsi_postSA_embed, dim=1)
        embedding_total = torch.cat([paths_postSA_embed_total, wsi_postSA_embed_total], dim=1)
        
        # Get total effect logits
        logits_total = self.to_logits(embedding_total)
        
        # Natural Direct Effect (NDE): Genomic effect with WSI blocked
        if use_counterfactual:
            # Create no-treatment WSI embedding (similar to c* in VQA paper)
            no_treatment_wsi = torch.ones_like(wsi_embed) * self.no_treatment_param
            
            # Counterfactual tokens: genomics with no-treatment WSI
            tokens_counterfactual = torch.cat([h_omic_bag, no_treatment_wsi], dim=1)
            
            # Process through attention (but WSI effect is blocked)
            mm_embed_cf = self.cross_attender(x=tokens_counterfactual, mask=mask, return_attention=False)
            mm_embed_cf = self.feed_forward(mm_embed_cf)
            mm_embed_cf = self.layer_norm(mm_embed_cf)
            
            # Only use genomic embeddings for counterfactual
            paths_cf_embed = mm_embed_cf[:, :self.num_pathways, :]
            paths_cf_embed = torch.mean(paths_cf_embed, dim=1)
            
            # Counterfactual prediction (direct genomic effect)
            logits_nde = self.counterfactual_head(paths_cf_embed)
            
            # Total Indirect Effect (TIE) = TE - NDE
            # This represents the debiased prediction
            logits_tie = logits_total - self.counterfactual_lambda * logits_nde
            
            if return_attn:
                return logits_tie, logits_total, logits_nde, attn_pathways, cross_attn_pathways, cross_attn_histology
            else:
                return logits_tie, logits_total, logits_nde
        else:
            if return_attn:
                return logits_total, attn_pathways, cross_attn_pathways, cross_attn_histology
            else:
                return logits_total