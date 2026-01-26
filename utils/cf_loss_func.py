import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.loss_func import nll_loss
import pdb


class CFSurvivalLoss(nn.Module):
    """
    Counterfactual survival loss combining main survival loss with counterfactual regularization
    反事实生存损失，结合主要生存损失和反事实正则化
    
    This loss function implements the CF-VQA style training for survival prediction:
    L = L_main(TE) + λ_g * L_genomic(NDE_g) + λ_i * L_imaging(NDE_i)
    
    Where:
    - TE: Total Effect (multimodal prediction)
    - NDE_g: Natural Direct Effect of genomics (genomic bias)
    - NDE_i: Natural Direct Effect of imaging (imaging bias)
    """
    
    def __init__(self, main_loss_fn, cf_weight=0.1, debias_mode='genomic', 
                 genomic_weight=None, imaging_weight=None, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.cf_weight = cf_weight
        self.debias_mode = debias_mode
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction
        
        # Allow different weights for different modalities
        self.genomic_weight = genomic_weight if genomic_weight is not None else cf_weight
        self.imaging_weight = imaging_weight if imaging_weight is not None else cf_weight
        
    def forward(self, main_logits, cf_outputs, y, t, c):
        """
        Compute counterfactual survival loss
        
        Args:
            main_logits: Main branch predictions (Total Effect) [batch_size, n_classes]
            cf_outputs: Dictionary containing counterfactual branch outputs
                - 'genomic_nde': Genomic-only predictions [batch_size, n_classes]
                - 'imaging_nde': Imaging-only predictions [batch_size, n_classes]
            y: Discrete time bins [batch_size, 1]
            t: Event times [batch_size, 1]
            c: Censorship indicators [batch_size, 1]
        
        Returns:
            total_loss: Combined loss value
        """
        
        # Main survival loss (Total Effect)
        if hasattr(self.main_loss_fn, '__call__'):
            main_loss = self.main_loss_fn(h=main_logits, y=y, t=t, c=c)
        else:
            main_loss = nll_loss(h=main_logits, y=y, c=c, alpha=self.alpha, eps=self.eps, reduction=self.reduction)
        
        total_loss = main_loss
        loss_components = {'main_loss': main_loss.item()}
        
        # Counterfactual regularization losses
        if self.debias_mode in ['genomic', 'both'] and 'genomic_nde' in cf_outputs:
            if hasattr(self.main_loss_fn, '__call__'):
                genomic_loss = self.main_loss_fn(h=cf_outputs['genomic_nde'], y=y, t=t, c=c)
            else:
                genomic_loss = nll_loss(h=cf_outputs['genomic_nde'], y=y, c=c, 
                                      alpha=self.alpha, eps=self.eps, reduction=self.reduction)
            
            weighted_genomic_loss = self.genomic_weight * genomic_loss
            total_loss += weighted_genomic_loss
            loss_components['genomic_loss'] = genomic_loss.item()
            loss_components['weighted_genomic_loss'] = weighted_genomic_loss.item()
            
        if self.debias_mode in ['imaging', 'both'] and 'imaging_nde' in cf_outputs:
            if hasattr(self.main_loss_fn, '__call__'):
                imaging_loss = self.main_loss_fn(h=cf_outputs['imaging_nde'], y=y, t=t, c=c)
            else:
                imaging_loss = nll_loss(h=cf_outputs['imaging_nde'], y=y, c=c,
                                      alpha=self.alpha, eps=self.eps, reduction=self.reduction)
            
            weighted_imaging_loss = self.imaging_weight * imaging_loss
            total_loss += weighted_imaging_loss
            loss_components['imaging_loss'] = imaging_loss.item()
            loss_components['weighted_imaging_loss'] = weighted_imaging_loss.item()
        
        # Store loss components for monitoring
        self.last_loss_components = loss_components
        
        return total_loss
    
    def get_loss_components(self):
        """Get individual loss components from last forward pass"""
        return getattr(self, 'last_loss_components', {})


class AdaptiveCFSurvivalLoss(CFSurvivalLoss):
    """
    Adaptive counterfactual loss that adjusts weights based on training progress
    自适应反事实损失，根据训练进度调整权重
    """
    
    def __init__(self, main_loss_fn, cf_weight=0.1, debias_mode='genomic', 
                 warmup_epochs=5, max_epochs=20, min_cf_weight=0.01, 
                 adaptive_strategy='linear', **kwargs):
        super().__init__(main_loss_fn, cf_weight, debias_mode, **kwargs)
        
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_cf_weight = min_cf_weight
        self.adaptive_strategy = adaptive_strategy
        self.base_cf_weight = cf_weight
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        """Update the current epoch for adaptive weighting"""
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warm-up phase: gradually increase CF weight
            progress = epoch / self.warmup_epochs
            if self.adaptive_strategy == 'linear':
                self.cf_weight = progress * self.base_cf_weight
            elif self.adaptive_strategy == 'cosine':
                self.cf_weight = 0.5 * (1 - np.cos(np.pi * progress)) * self.base_cf_weight
        else:
            # Main training phase: gradually decrease CF weight
            remaining_epochs = self.max_epochs - self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / remaining_epochs
            
            if self.adaptive_strategy == 'linear':
                self.cf_weight = self.base_cf_weight - progress * (self.base_cf_weight - self.min_cf_weight)
            elif self.adaptive_strategy == 'exponential':
                self.cf_weight = self.base_cf_weight * (self.min_cf_weight / self.base_cf_weight) ** progress
            elif self.adaptive_strategy == 'constant':
                self.cf_weight = self.base_cf_weight
        
        # Update individual modality weights
        self.genomic_weight = self.cf_weight
        self.imaging_weight = self.cf_weight


class FocalCFSurvivalLoss(CFSurvivalLoss):
    """
    Focal loss variant for counterfactual survival prediction
    专注于困难样本的反事实生存预测损失
    """
    
    def __init__(self, main_loss_fn, cf_weight=0.1, debias_mode='genomic',
                 focal_alpha=0.25, focal_gamma=2.0, **kwargs):
        super().__init__(main_loss_fn, cf_weight, debias_mode, **kwargs)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        """
        Compute focal loss for survival prediction
        """
        # Convert to probabilities
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        
        # Get target probabilities
        target_probs = torch.gather(survival, 1, targets.long())
        
        # Compute focal weights
        focal_weights = alpha * (1 - target_probs) ** gamma
        
        # Compute weighted cross-entropy
        ce_loss = F.cross_entropy(logits, targets.squeeze().long(), reduction='none')
        focal_loss = focal_weights.squeeze() * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, main_logits, cf_outputs, y, t, c):
        """Override to use focal loss for main branch"""
        
        # Use focal loss for main branch
        main_loss = self.focal_loss(main_logits, y, self.focal_alpha, self.focal_gamma)
        total_loss = main_loss
        loss_components = {'main_focal_loss': main_loss.item()}
        
        # Standard NLL loss for counterfactual branches
        if self.debias_mode in ['genomic', 'both'] and 'genomic_nde' in cf_outputs:
            genomic_loss = nll_loss(h=cf_outputs['genomic_nde'], y=y, c=c,
                                  alpha=self.alpha, eps=self.eps, reduction=self.reduction)
            weighted_genomic_loss = self.genomic_weight * genomic_loss
            total_loss += weighted_genomic_loss
            loss_components['genomic_loss'] = genomic_loss.item()
            
        if self.debias_mode in ['imaging', 'both'] and 'imaging_nde' in cf_outputs:
            imaging_loss = nll_loss(h=cf_outputs['imaging_nde'], y=y, c=c,
                                  alpha=self.alpha, eps=self.eps, reduction=self.reduction)
            weighted_imaging_loss = self.imaging_weight * imaging_loss
            total_loss += weighted_imaging_loss
            loss_components['imaging_loss'] = imaging_loss.item()
        
        self.last_loss_components = loss_components
        return total_loss


class ContrastiveCFSurvivalLoss(CFSurvivalLoss):
    """
    Contrastive counterfactual loss to encourage diverse representations
    对比学习的反事实损失，鼓励多样化的表示
    """
    
    def __init__(self, main_loss_fn, cf_weight=0.1, debias_mode='genomic',
                 contrastive_weight=0.05, temperature=0.1, **kwargs):
        super().__init__(main_loss_fn, cf_weight, debias_mode, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
    
    def contrastive_loss(self, main_logits, cf_logits):
        """
        Compute contrastive loss between main and counterfactual predictions
        """
        # Normalize logits
        main_norm = F.normalize(main_logits, dim=1)
        cf_norm = F.normalize(cf_logits, dim=1)
        
        # Compute similarity
        similarity = torch.mm(main_norm, cf_norm.T) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = main_logits.size(0)
        labels = torch.arange(batch_size).to(main_logits.device)
        
        # Compute contrastive loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def forward(self, main_logits, cf_outputs, y, t, c):
        """Override to add contrastive terms"""
        
        # Standard CF loss
        total_loss = super().forward(main_logits, cf_outputs, y, t, c)
        
        # Add contrastive terms
        contrastive_losses = []
        
        if self.debias_mode in ['genomic', 'both'] and 'genomic_nde' in cf_outputs:
            genomic_contrastive = self.contrastive_loss(main_logits, cf_outputs['genomic_nde'])
            contrastive_losses.append(genomic_contrastive)
            
        if self.debias_mode in ['imaging', 'both'] and 'imaging_nde' in cf_outputs:
            imaging_contrastive = self.contrastive_loss(main_logits, cf_outputs['imaging_nde'])
            contrastive_losses.append(imaging_contrastive)
        
        if contrastive_losses:
            total_contrastive = sum(contrastive_losses) / len(contrastive_losses)
            total_loss += self.contrastive_weight * total_contrastive
            
            # Update loss components
            loss_components = self.get_loss_components()
            loss_components['contrastive_loss'] = total_contrastive.item()
            self.last_loss_components = loss_components
        
        return total_loss


# Utility functions for loss analysis
def analyze_loss_components(loss_fn, dataloader, model, device, max_batches=10):
    """
    Analyze loss components across batches for debugging
    分析各批次的损失组件用于调试
    """
    if not isinstance(loss_fn, CFSurvivalLoss):
        print("Loss function is not a CFSurvivalLoss instance")
        return
    
    model.eval()
    all_components = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
                
            # Process batch (simplified)
            try:
                # You'll need to adapt this based on your data loading
                wsi, omics, y, t, c = batch[:5]  # Simplified
                
                # Forward pass
                input_args = {"x_path": wsi.to(device)}
                for j in range(len(omics)):
                    input_args[f'x_omic{j+1}'] = omics[j].to(device)
                input_args["training_mode"] = "counterfactual"
                
                main_logits, cf_outputs = model(**input_args)
                
                # Compute loss
                loss = loss_fn(main_logits, cf_outputs, y.to(device), t.to(device), c.to(device))
                
                # Get components
                components = loss_fn.get_loss_components()
                components['total_loss'] = loss.item()
                all_components.append(components)
                
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                continue
    
    # Aggregate results
    if all_components:
        avg_components = {}
        for key in all_components[0].keys():
            values = [comp.get(key, 0) for comp in all_components]
            avg_components[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        print("\nLoss Component Analysis:")
        print("-" * 50)
        for key, stats in avg_components.items():
            print(f"{key}:")
            print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        return avg_components
    
    return None