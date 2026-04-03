from collections import OrderedDict
from os.path import join
import pdb
from transformers import AutoTokenizer, AutoModel
import numpy as np
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE, global_mean_pool, global_max_pool
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.data import Data, Batch
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data
import pickle
import hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class HypergraphPathwayEncoder(nn.Module):
    """
    完整的超图通路编码器
    """
    def __init__(self, 
                 pathway_file_path,
                 cache_dir="./hypergraph_cache",
                 input_dim=1, 
                 hidden_dim=128, 
                 output_dim=256, 
                 num_layers=1):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # 加载并构建固定的超图结构
        self.hypergraph_structure = self._load_or_build_hypergraph_structure(pathway_file_path)
        self.num_genes = self.hypergraph_structure['num_genes']  # 4999
        self.num_pathways = self.hypergraph_structure['num_pathways']  # 331
        
        # 注册为buffer，不参与梯度计算但会保存到模型状态
        self.register_buffer('incident_matrix', 
                           torch.tensor(self.hypergraph_structure['incident_matrix'], 
                                      dtype=torch.float32))
        
        # 超图卷积层
        self.hypergraph_convs = nn.ModuleList([
            HypergraphConv(input_dim if i == 0 else hidden_dim, 
                          hidden_dim if i < num_layers-1 else output_dim)
            for i in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers-1 else output_dim)
            for i in range(num_layers)
        ])
        
        self.pathway_self_attention = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=1, 
            batch_first=True
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(output_dim, output_dim)

    def _get_cache_filename(self, pathway_file_path):
        """
        根据原始文件路径和内容生成缓存文件名
        """
        # 获取文件的修改时间和大小作为唯一标识
        file_stat = os.stat(pathway_file_path)
        file_info = f"{pathway_file_path}_{file_stat.st_mtime}_{file_stat.st_size}"
        
        # 生成MD5哈希作为缓存文件名
        hash_object = hashlib.md5(file_info.encode())
        cache_filename = f"hypergraph_structure_{hash_object.hexdigest()}.pkl"
        
        return self.cache_dir / cache_filename
    
    def _load_or_build_hypergraph_structure(self, pathway_file_path, force_rebuild=False):
        """
        加载或构建超图结构（带缓存机制）
        """
        cache_file = self._get_cache_filename(pathway_file_path)
        
        # 检查是否需要重新构建
        if not force_rebuild and cache_file.exists():
            try:
                print(f"🔄 从缓存加载超图结构: {cache_file}")
                with open(cache_file, 'rb') as f:
                    hypergraph_structure = pickle.load(f)
                
                # 验证缓存数据完整性
                required_keys = ['genes', 'pathways', 'num_genes', 'num_pathways', 
                               'incident_matrix', 'hyperedges', 'hyperedge_index']
                if all(key in hypergraph_structure for key in required_keys):
                    print(f"✅ 缓存加载成功!")
                    return hypergraph_structure
                else:
                    print("⚠️  缓存文件损坏，重新构建...")
            except Exception as e:
                print(f"⚠️  缓存加载失败: {e}，重新构建...")
        
        # 重新构建超图结构
        print(f"🔨 构建超图结构: {pathway_file_path}")
        hypergraph_structure = self._build_hypergraph_structure_from_file(pathway_file_path)
        
        # 保存到缓存
        try:
            print(f"💾 保存超图结构到缓存: {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump(hypergraph_structure, f)
            print(f"✅ 缓存保存成功!")
        except Exception as e:
            print(f"⚠️  缓存保存失败: {e}")
        
        return hypergraph_structure
    
    def _build_hypergraph_structure_from_file(self, pathway_file_path):
        """
        从CSV文件构建超图结构（实际处理逻辑）
        """
        print("📊 读取通路数据文件...")
        
        # 读取数据
        df = pd.read_csv(pathway_file_path)
        print(f"   - 数据形状: {df.shape}")
        
        # 基因列表
        genes = df['gene'].tolist()
        print(f"   - 基因数量: {len(genes)}")
        
        # 通路列表 (排除gene列)
        pathways = df.columns[1:].tolist()
        print(f"   - 通路数量: {len(pathways)}")
        
        # 基因-通路关联矩阵 (4999 x 331)
        print("🔗 构建基因-通路关联矩阵...")
        incident_matrix = df.iloc[:, 1:].values.astype(np.float32)
        
        # 统计信息
        total_connections = int(incident_matrix.sum())
        avg_genes_per_pathway = incident_matrix.sum(axis=0).mean()
        avg_pathways_per_gene = incident_matrix.sum(axis=1).mean()
        
        print(f"   - 总连接数: {total_connections}")
        print(f"   - 平均每个通路包含基因数: {avg_genes_per_pathway:.1f}")
        print(f"   - 平均每个基因参与通路数: {avg_pathways_per_gene:.1f}")
        
        # 构建超边列表和索引
        print("  构建超边结构...")
        hyperedges = []
        hyperedge_index = []
        
        for pathway_idx, pathway in enumerate(pathways):
            # 找到属于该通路的基因
            gene_indices = np.where(incident_matrix[:, pathway_idx] == 1)[0]
            
            if len(gene_indices) > 0:
                hyperedges.append({
                    'pathway_idx': pathway_idx,
                    'pathway_name': pathway,
                    'gene_indices': gene_indices.tolist(),
                    'size': len(gene_indices)
                })
                
                # 构建超边索引 (用于PyTorch Geometric)
                for gene_idx in gene_indices:
                    hyperedge_index.append([gene_idx, pathway_idx])
        
        hyperedge_index = np.array(hyperedge_index).T  # [2, num_connections]
        
        print(f"   - 有效超边数量: {len(hyperedges)}")
        print(f"   - 超边索引形状: {hyperedge_index.shape}")
        
        # 分析通路大小分布
        pathway_sizes = [he['size'] for he in hyperedges]
        print(f"   - 通路大小范围: {min(pathway_sizes)} - {max(pathway_sizes)}")
        print(f"   - 通路大小中位数: {np.median(pathway_sizes)}")
        
        structure = {
            'genes': genes,
            'pathways': pathways,
            'num_genes': len(genes),
            'num_pathways': len(pathways),
            'incident_matrix': incident_matrix,
            'hyperedges': hyperedges,
            'hyperedge_index': hyperedge_index,
            # 添加元数据
            'metadata': {
                'total_connections': total_connections,
                'avg_genes_per_pathway': float(avg_genes_per_pathway),
                'avg_pathways_per_gene': float(avg_pathways_per_gene),
                'pathway_size_range': (int(min(pathway_sizes)), int(max(pathway_sizes))),
                'pathway_size_median': float(np.median(pathway_sizes))
            }
        }
        
        print("✅ 超图结构构建完成!")
        return structure
    
    def forward(self, patient_expression_batch):
        """
        完整的前向传播
        
        Args:
            patient_expression_batch: [batch_size, num_genes] 患者基因表达数据
            
        Returns:
            pathway_tokens: [batch_size, num_pathways, output_dim] 通路tokens
        """
        batch_size = patient_expression_batch.size(0)
        device = patient_expression_batch.device
        
        # 确保输入维度正确
        assert patient_expression_batch.size(1) == self.num_genes, \
            f"Expected {self.num_genes} genes, got {patient_expression_batch.size(1)}"
        
        # 存储所有患者的pathway tokens
        all_pathway_tokens = []
        
        for batch_idx in range(batch_size):
            patient_expression = patient_expression_batch[batch_idx]  # [num_genes]
            
            # === 方法1: 超图神经网络编码 ===
            hypergraph_tokens_x = self._encode_with_hypergraph(patient_expression, device)
            all_pathway_tokens.append(hypergraph_tokens_x)
        
        # 合并批次
        pathway_tokens = torch.stack(all_pathway_tokens, dim=0).squeeze(2)
        # print('pathway_tokens.shape', pathway_tokens.shape) # torch.Size([1, 4999])
        return pathway_tokens
    
    def _encode_with_hypergraph(self, patient_expression, device):
        """
        使用超图神经网络编码
        """
        # 构建节点特征 [num_genes, 1]
        node_features = patient_expression.unsqueeze(1).float()
        
        # 获取超边索引
        hyperedge_index = torch.tensor(
            self.hypergraph_structure['hyperedge_index'], 
            dtype=torch.long, 
            device=device
        )
        
        # 多层超图卷积
        x = node_features
        for i, (conv, norm) in enumerate(zip(self.hypergraph_convs, self.layer_norms)):
            x = conv(x, hyperedge_index)
            x = norm(x)
            if i < len(self.hypergraph_convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        # 将基因级别特征聚合到通路级别
        # print('x 在超图后的 shape:', x.shape)
        # exit(0)
        pathway_embeddings = self._aggregate_genes_to_pathways(x, device)
        # 通路间自注意力
        pathway_embeddings = pathway_embeddings.unsqueeze(0)  # [1, num_pathways, dim]
        attended_pathways, _ = self.pathway_self_attention(
            pathway_embeddings, pathway_embeddings, pathway_embeddings
        )
        
        return attended_pathways.squeeze(0)  # [num_pathways, dim]
    
    def _aggregate_genes_to_pathways(self, gene_embeddings, device):
        """
        将基因级别嵌入聚合到通路级别
        """
        pathway_embeddings = torch.zeros(
            self.num_pathways, 
            gene_embeddings.size(1), 
            device=device
        )
        
        incident_matrix = self.incident_matrix.to(device)
        
        for pathway_idx in range(self.num_pathways):
            # 获取该通路的基因mask
            gene_mask = incident_matrix[:, pathway_idx] == 1
            
            if gene_mask.sum() > 0:
                # 对该通路的基因嵌入求平均
                pathway_embeddings[pathway_idx] = gene_embeddings[gene_mask].mean(dim=0)
        
        return pathway_embeddings
    
    def _encode_with_sparse_mlp(self, patient_expression, device):
        """
        使用稀疏MLP编码 (原SURVPATH方法)
        """
        sparse_pathway_tokens = []
        incident_matrix = self.incident_matrix.to(device)
        
        for pathway_idx in range(self.num_pathways):
            mlp = self.sparse_mlps[pathway_idx]
            
            if mlp is not None:
                # 获取该通路的基因表达
                gene_mask = incident_matrix[:, pathway_idx] == 1
                pathway_genes = patient_expression[gene_mask]
                
                # 通过MLP编码
                pathway_token = mlp(pathway_genes)
                sparse_pathway_tokens.append(pathway_token)
            else:
                # 空通路用零向量
                sparse_pathway_tokens.append(
                    torch.zeros(self.hypergraph_convs[-1].out_channels, device=device)
                )
        
        return torch.stack(sparse_pathway_tokens)  # [num_pathways, dim]
    
    def _fuse_features(self, hypergraph_tokens, sparse_tokens):
        """
        融合超图特征和稀疏MLP特征
        """
        # 拼接两种特征
        concatenated = torch.cat([hypergraph_tokens, sparse_tokens], dim=-1)
        
        # 通过融合层
        fused = self.feature_fusion(concatenated)
        
        return fused

class GatedFusion(nn.Module):
    """
    门控融合模块 - 让网络自动学习多模态融合权重
    【已改】实现自适应权重学习，替代简单相加
    """
    def __init__(self, feature_dim: int, num_modalities: int = 3):
        """
        Args:
            feature_dim: 每个模态的特征维度
            num_modalities: 模态数量 (默认3: 基因/通路/文本)
        """
        super(GatedFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # 【已改】为每个模态创建可学习的门控权重
        self.gate_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)

        # 【已改】创建门控网络，基于输入特征动态调整权重
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_modalities),
            nn.Sigmoid()  # 每个模态独立决定权重，允许多模态同时保持高激活
        )

        # 【已改】融合后的特征投影
        self.fusion_projection = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, modality_features):
        """
        Args:
            modality_features: List of tensors, each [batch_size, feature_dim]
                             长度应该等于num_modalities

        Returns:
            fused_features: [batch_size, feature_dim]
            gate_weights: [batch_size, num_modalities] - 学习到的权重
        """
        batch_size = modality_features[0].size(0)
        device = modality_features[0].device

        # 确保输入数量正确
        assert len(modality_features) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(modality_features)}"

        # 【已改】方法1: 使用可学习的静态权重
        static_weights = F.softmax(self.gate_weights, dim=0)

        # 【已改】方法2: 使用动态门控网络
        concatenated = torch.cat(modality_features, dim=1)  # [batch_size, feature_dim * num_modalities]
        dynamic_weights = self.gate_network(concatenated)  # [batch_size, num_modalities]

        # 【已改】结合静态和动态权重 (80%动态 + 20%静态)
        final_weights = 0.8 * dynamic_weights + 0.2 * static_weights.unsqueeze(0).expand(batch_size, -1)

        # 【已改】应用门控权重进行加权融合
        weighted_features = []
        for i, features in enumerate(modality_features):
            weighted_features.append(features * final_weights[:, i:i+1])

        # 【已改】融合所有加权特征
        fused = torch.cat(weighted_features, dim=1)  # [batch_size, feature_dim * num_modalities]

        # 【已改】通过投影层得到最终融合特征
        output = self.fusion_projection(fused)  # [batch_size, feature_dim]

        return output, final_weights


class FeatureGating(nn.Module):
    """
    针对 100+ 高维不稳定基因特征的动态软掩码门控
    """
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, max(32, input_dim // 2)),
            nn.SELU(),
            nn.Linear(max(32, input_dim // 2), input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [Batch, input_dim]
        attention_weights = self.gate(x)
        return x * attention_weights


class SNNOmics(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4,
                 n_stage_classes: int=4, enable_multitask: bool=False, multitask_weight: float=0.3,
                 graph_data_path: str="graph_data.pkl", ab_model: int=3, encoder_dropout: float=0.25):
        """
        SNNOmics多模态生存预测模型

        Args:
            omic_input_dim: 基因/组学数据维度
            model_size_omic: 基因网络大小 ('small' 或 'big')
            n_classes: 生存预测类别数
            n_stage_classes: 分期预测类别数
            enable_multitask: 是否启用多任务学习（生存+分期）
            multitask_weight: 多任务权重
            graph_data_path: 图数据路径
            ab_model: 【新增】运行模式控制
                      1 = 仅文本模式 (only_text)
                      2 = 仅基因模式 (only_omic)
                      3 = 多模态融合模式 (基因+文本)【默认】
        """
        super(SNNOmics, self).__init__()

        # 【新增】存储运行模式
        self.ab_model = ab_model
        self.encoder_dropout = float(encoder_dropout)
        print(f"🚀 [Model Config] 运行模式: {ab_model} "
              f"({'仅文本' if ab_model == 1 else '仅基因' if ab_model == 2 else '多模态融合'})")

        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        # self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}
        # self.size_dict_omic = {'small': [768, 768], 'big': [1024, 1024, 1024, 256]}
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        self.target_dim = hidden[-1]

        # 通路编码器
        # === [修改] 判断是否使用通路编码器 ===
        # 原有的 combine 数据维度通常是 4999。如果你筛选后只有几十/几百，说明是自定义数据。
        # 我们设定一个阈值（比如 2000），小于这个数就不加载通路图。
        if omic_input_dim > 2000:
            self.use_pathway = True
            self.pathway_encoder = HypergraphPathwayEncoder(
                pathway_file_path='./datasets_csv/pathway_compositions/combine_comps.csv',
                output_dim=self.target_dim,
                num_layers=2
            )
            # 只有在使用通路时才初始化 Cross Attention
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.target_dim,
                num_heads=1,
                batch_first=True
            )
            self.query_projection = nn.Linear(self.target_dim + hidden[-1], self.target_dim)
        else:
            print(f"🔥 [Model Info] 检测到筛选后的特征 (Dim={omic_input_dim})，自动禁用通路编码器，改用直接拼接模式。")
            self.use_pathway = False
            self.pathway_encoder = None
            self.cross_attention = None
            self.query_projection = None
            # 【新增】仅在需要时初始化分类器
            if self.ab_model != 1:  # 非仅文本模式需要分类器
                self.survival_classifier = nn.Linear(1024, n_classes)
            #原代码存在逻辑失误
            # 256*2 + 256 = 768 (如果是 small)
            # 或者 cat 了 text_embeddings (768) ? 原代码里 cat_embeddings = cat([gene, cross, text])
            # 256 + 256 + 768 = 1280
        # ==========================================

        
        if self.ab_model == 1: # only text
            self.use_debias = False
            self.use_align_loss = False
            self.only_omic = False
            self.only_text = True
            self.multi_model = False
            self.use_resnet = True
            self.NOT_hygraph = True
        if self.ab_model == 2: # only omic
            self.use_debias = False
            self.use_align_loss = False
            self.only_omic = True
            self.only_text = False
            self.multi_model = False
            self.use_resnet = False
            self.NOT_hygraph = True
        if self.ab_model == 3: # 多模态。
            self.use_debias = False
            self.use_align_loss = False
            self.only_omic = False
            self.only_text = False
            self.multi_model = True
            self.use_resnet = True
            self.NOT_hygraph = True # 不使用超图
        if self.ab_model == 4: # 多模态。残差 + 对齐损失 + 去偏 (强度，) 
            self.use_debias = True
            self.use_align_loss = False
            self.only_omic = False
            self.only_text = False
            self.multi_model = True
            self.use_resnet = True
            self.NOT_hygraph = False # 使用超图
            self.weight_debiase = 0.4 # 如果为 False，表示用可学习的参数
        if self.ab_model == 5: # 多模态。去偏，基因不用我们的方法
            self.use_debias = True
            self.use_align_loss = False
            self.only_omic = False  
            self.only_text = False
            self.multi_model = True
            self.use_resnet = True
            self.NOT_hygraph = True # 不使用超图
            self.weight_debiase = 0.1 # 如果为 False，表示用可学习的参数
        
        self.clinical_bert_tokenizer = AutoTokenizer.from_pretrained("biobert")
        self.clinical_bert_model = AutoModel.from_pretrained("biobert")

        self.n_classes = n_classes      
        self.n_stage_classes = n_stage_classes
        self.enable_multitask = enable_multitask
        self.multitask_weight = multitask_weight
        
        

        # 构建动态的 Genomic SNN 网络
        fc_omic = []
        target_dim = hidden[-1]  # 通常为 256

        # 针对输入维度的不同范围，设计平滑的过渡层 (Buffer Layers)
        if omic_input_dim <= 64:
            # 极低维因果特征：使用窄漏斗防止严重过拟合
            fc_omic.append(SNN_Block(dim1=omic_input_dim, dim2=32, dropout=min(0.1, self.encoder_dropout)))
            fc_omic.append(SNN_Block(dim1=32, dim2=target_dim, dropout=self.encoder_dropout))

        elif omic_input_dim < target_dim:
            # 100+ 的中等维度特征 (如 105, 140 等)：使用 128 维作为平滑过渡
            # 这样既不会丢失 100+ 基因的信息，也不会一步到位升维到 256 放大噪声
            mid_dim = 128
            # 在输入端应用较低的 dropout 容忍不稳定的特征波动
            fc_omic.append(SNN_Block(dim1=omic_input_dim, dim2=mid_dim, dropout=min(0.15, self.encoder_dropout)))
            fc_omic.append(SNN_Block(dim1=mid_dim, dim2=target_dim, dropout=self.encoder_dropout))

        else:
            # 兼容未筛选的高维全集 (如原始的 4999 维或 1024 维)
            fc_omic.append(SNN_Block(dim1=omic_input_dim, dim2=hidden[0], dropout=self.encoder_dropout))
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=self.encoder_dropout))

        self.fc_omic = nn.Sequential(*fc_omic)

        # 【新增】防守型表征学习模块
        self.feature_gating = FeatureGating(input_dim=omic_input_dim)
        self.gene_fusion_dropout = nn.Dropout(p=0.3)
        
        # 【修复】根据运行模式初始化survival_classifier
        if self.ab_model == 1:  # 仅文本模式
            self.survival_classifier = nn.Sequential(
                nn.Linear(768, 768 // 4),
                nn.ReLU(),
                nn.Linear(768 // 4, n_classes)
            )
        elif self.ab_model == 2:  # 仅基因模式
            self.survival_classifier = nn.Sequential(
                nn.Linear(hidden[-1], hidden[-1] // 2),
                nn.ReLU(),
                nn.Linear(hidden[-1] // 2, n_classes)
            )
        else:  # 多模态融合模式 (ab_model == 3)
            if self.use_pathway:
                # 通路模式：Gene(256) + Cross(256) + Text(768) = 1280维
                self.survival_classifier = nn.Linear(1280, n_classes)
            else:
                # 无通路模式：使用门控融合 (256维)，不再直接拼接
                self.survival_classifier = nn.Linear(self.target_dim, n_classes)
            self.only_text = False
        self.text_classifier = nn.Linear(self.target_dim, n_classes)
        self.gene_classifier = nn.Linear(hidden[-1], n_classes)
        
        # Stage prediction head (only if multitask is enabled)
        # 【修复】根据运行模式初始化stage_classifier
        if self.enable_multitask:
            if self.ab_model == 1:  # 仅文本模式
                self.stage_classifier = nn.Sequential(
                    nn.Linear(768, 768 // 4),
                    nn.ReLU(),
                    nn.Linear(768 // 4, n_stage_classes)
                )
            elif self.ab_model == 2:  # 仅基因模式
                self.stage_classifier = nn.Linear(hidden[-1], n_stage_classes)
            else:  # 多模态融合模式 (ab_model == 3)
                if self.use_pathway:
                    # 通路模式：Gene + Text 拼接
                    self.stage_classifier = nn.Sequential(
                        nn.Linear(hidden[-1] + 768, hidden[-1]),
                        nn.ReLU(),
                        nn.Linear(hidden[-1], n_stage_classes)
                    )
                else:
                    # 无通路模式：使用门控融合 (256维)
                    self.stage_classifier = nn.Sequential(
                        nn.Linear(self.target_dim, self.target_dim // 2),
                        nn.ReLU(),
                        nn.Linear(self.target_dim // 2, n_stage_classes)
                    )
            self.stage_classifier_text = nn.Linear(768, n_stage_classes)

        self.constant = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.debiase_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.lin_weight = nn.Linear(768, 1)
        
        self.debiase_weight2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        
        self.dropout_omic = nn.Dropout(0.3)
        init_max_weights(self)
        self.text_projection = nn.Linear(768, self.target_dim)
        self.query_projection = nn.Linear(self.target_dim + hidden[-1], self.target_dim)

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.target_dim,
            num_heads=1,
            batch_first=True  # 确保输入形状为 [batch, seq, dim]
        )

        # 【已改】初始化门控融合模块
        # 对于 survival 预测的融合 (基因 + 交叉注意力 + 文本)
        if self.use_pathway:
            self.gated_fusion_survival = GatedFusion(
                feature_dim=self.target_dim,
                num_modalities=3  # 基因 + 交叉注意力 + 文本
            )
        else:
            # 如果没有通路，使用基因 + 文本的融合
            self.gated_fusion_survival = GatedFusion(
                feature_dim=self.target_dim,
                num_modalities=2  # 基因 + 文本
            )

        # 【已改】对于 stage 预测的融合 (基因 + 文本)
        self.gated_fusion_stage = GatedFusion(
            feature_dim=self.target_dim,
            num_modalities=2  # 基因 + 文本
        )
        
    def aggregate_subword_attentions(self, tokens, scores):
        """
        将子词(subword)的注意力分数合并到它们所属的完整单词上。
        """
        word_tokens = []
        word_attentions = []
        
        current_word = ""
        current_attention = 0.0
        subword_count = 0

        # 过滤掉特殊token [CLS], [SEP], [PAD]
        special_tokens = {self.clinical_bert_tokenizer.cls_token, 
                            self.clinical_bert_tokenizer.sep_token, 
                            self.clinical_bert_tokenizer.pad_token}

        for token, score in zip(tokens, scores):
            if token in special_tokens:
                continue

            if token.startswith("##"):
                current_word += token[2:]
                current_attention += score
                subword_count += 1
            else:
                # 遇到新词，先保存上一个词的结果
                if current_word:
                    # 使用平均分作为整个词的注意力
                    word_tokens.append(current_word)
                    word_attentions.append(current_attention / subword_count)
                
                # 开始记录新词
                current_word = token
                current_attention = score
                subword_count = 1
        
        # 保存最后一个词
        if current_word:
            word_tokens.append(current_word)
            word_attentions.append(current_attention / subword_count)

        return word_tokens, torch.tensor(word_attentions)


    def generate_highlighted_text(self, words, scores):
        """根据分数生成带背景色的高亮文本"""
        scores = scores.numpy()
        min_s, max_s = scores.min(), scores.max()
        normalized_scores = (scores - min_s) / (max_s - min_s)
        
        final_text = ""
        for word, score in zip(words, normalized_scores):
                intensity = int(255 - score * 150)
                bg_color = f"\033[48;2;255;255;{intensity}m" # 黄色背景
                text_color = "\033[30m" # 黑色字体
                reset_color = "\033[0m"
                final_text += f"{bg_color}{text_color} {word} {reset_color}"

        print("--- [Attention Visualization (Words)] ---")
        print("(颜色越黄代表模型关注度越高)")
        print(final_text.lstrip()) # lstrip()移除开头的空格
        print("-----------------------------------------\n")


    def visualize_keywords_in_context(self, original_words, original_scores, stop_words, punctuation, save_path=None):
        """
        在原文中只高亮关键词。
        
        Args:
            original_words: 单词列表
            original_scores: attention分数列表
            stop_words: 停用词集合
            punctuation: 标点符号集合
            save_path: 保存HTML文件的路径（新增）。如果为None，则打印到终端
        """
        # 创建一个新的"可视化分数列表"，将停用词和标点符号的分数设为0
        visualization_scores = []
        for word, score in zip(original_words, original_scores):
            if word.lower() in stop_words or word in punctuation:
                visualization_scores.append(0.0)
            else:
                visualization_scores.append(score.item())
        
        # 归一化可视化分数以便映射到颜色
        scores_tensor = torch.tensor(visualization_scores)
        min_s, max_s = scores_tensor.min(), scores_tensor.max()
        
        # 防止所有词都是停用词导致除以零
        if max_s == 0:
            normalized_scores = scores_tensor
        else:
            normalized_scores = (scores_tensor - min_s) / (max_s - min_s)

        # ========== 如果指定了save_path，生成HTML文件 ==========
        if save_path:
            html_content = """<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>文本关键词可视化</title>
        <style>
            body { 
                font-family: 'Arial', 'Microsoft YaHei', sans-serif;
                padding: 30px; 
                max-width: 1200px; 
                margin: 0 auto;
                background-color: #f9f9f9;
            }
            h2 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            .legend {
                margin: 20px 0;
                padding: 15px;
                background-color: #e8f5e9;
                border-left: 4px solid #4CAF50;
                border-radius: 5px;
            }
            .highlighted-text { 
                line-height: 2.2; 
                font-size: 16px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .word { 
                padding: 3px 6px; 
                margin: 0 2px;
                border-radius: 4px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h2>📊 文本关键词重要性可视化</h2>
        <div class="legend">
            <strong>说明：</strong>颜色越黄（越亮）代表模型越关注该词，灰色词为停用词或标点
        </div>
        <div class="highlighted-text">
    """
            
            # 生成每个词的HTML
            for word, score in zip(original_words, normalized_scores):
                score_val = score.item()
                
                if score_val == 0:
                    # 停用词或标点 - 灰色
                    bg_color = 'rgba(200, 200, 200, 0.3)'
                else:
                    # 关键词 - 黄色渐变（根据重要性）
                    intensity = int(255 - score_val * 150)
                    bg_color = f'rgb(255, 255, {intensity})'
                
                html_content += f'<span class="word" style="background-color: {bg_color};">{word}</span>\n'
            
            html_content += """
        </div>
    </body>
    </html>
    """
            
            # 保存HTML文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
        # ========== 否则，打印到终端（保持原有行为）==========
        else:
            final_text = ""
            for word, score in zip(original_words, normalized_scores):
                    # score 为0的词（停用词等）intensity会最大，背景色最浅
                    intensity = int(255 - score * 150)
                    bg_color = f"\033[48;2;255;255;{intensity}m" # 黄色背景
                    text_color = "\033[30m" # 黑色字体
                    reset_color = "\033[0m"
                    final_text += f"{bg_color}{text_color} {word} {reset_color}"

            print("--- [Keyword Visualization in Context] ---")
            print("(仅关键词被高亮，颜色越黄代表模型关注度越高)")
            print(final_text.lstrip())
            print("------------------------------------------\n")


    def visualize_top_pathways(self, attention_scores, pathway_names, top_k=15, save_path=None, save_figure=None):
        """
        可视化最重要的Top-K个通路。

        Args:
            attention_scores (torch.Tensor): 单个样本的注意力权重，形状为 [num_pathways]。
            pathway_names (list): 包含所有通路名称的列表。
            top_k (int): 要显示的最重要通路的数量。
            save_path (str): 保存CSV文件的路径（新增）。如果为None，则打印到终端
            save_figure (str): 保存图片的路径（新增）。如果为None，则不保存图片
        """
        # 将Tensor转为numpy数组
        scores = attention_scores.cpu().detach().numpy()
        
        # 创建DataFrame并排序
        df = pd.DataFrame({
            'Pathway': pathway_names,
            'Attention_Score': scores
        })
        df_sorted = df.sort_values(by='Attention_Score', ascending=False).head(top_k)
        
        # ========== 如果指定了save_path，保存CSV ==========
        if save_path:
            df_sorted.to_csv(save_path, index=False)
        # ========== 【修改】只有在既不保存CSV也不保存图片时，才打印到终端 ==========
        elif not save_figure:
            print(f"\n--- [模型通路解释] ---")
            print(f"Top {top_k} Most Important Pathways:")
            print(df_sorted)
            print("--------------------------\n")

        # ========== 如果指定了save_figure，生成并保存图片 ==========
        if save_figure:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Attention_Score', y='Pathway', data=df_sorted, palette='viridis')
            
            # 美化图表
            plt.title(f'Top {top_k} Important Pathways identified by Cross-Attention', fontsize=16)
            plt.xlabel('Attention Score', fontsize=12)
            plt.ylabel('Pathway', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            
            # 确保y轴标签完全显示
            plt.tight_layout()
            
            # 保存图片
            plt.savefig(save_figure, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭图形，避免占用内存

    def fusion(self, z_main, z_text, z_gene, main_fact=False, text_fact=False, gene_fact=False):
        # Apply counterfactual transformations
        if not main_fact:
            z_main = self.constant * torch.ones_like(z_main).cuda()
        if not text_fact:
            z_text = self.constant * torch.ones_like(z_text).cuda()
        if not gene_fact:
            z_gene = self.constant * torch.ones_like(z_gene).cuda()
        
        # cfvqa
        # z = z_main * torch.sigmoid(z_text)
        
        z = z_main + z_text
        if z_gene is not None and gene_fact:
            z = z + z_gene
        z = torch.log(torch.sigmoid(z) + 1e-9)
        # z = torch.nn.functional.leaky_    relu(z) 
        # z = torch.log(torch.nn.functional.relu(z) + 1e-9)  # ReLU activation
        
        
        # # cat + linear fusion
        # z = torch.cat((z_main, z_text), dim=1)
        # if z_gene is not None and gene_fact:
        #     z = torch.cat((z, z_gene), dim=1)
        # z = self.fusion_lin(z)
        
        # # add + linear fusion
        # z = z_main + z_text
        # if z_gene is not None and gene_fact:
        #     z = z + z_gene
        # z = self.fusion_lin_add(z)
        
        return z
    
    def fusion_stage(self, z_main, z_text, z_gene, main_fact=False, text_fact=False, gene_fact=False):
        # Apply counterfactual transformations

        if not main_fact:
            z_main = self.constant * torch.ones_like(z_main).cuda()
        if not text_fact:
            z_text = self.constant * torch.ones_like(z_text).cuda()
        if not gene_fact:
            z_gene = self.constant * torch.ones_like(z_gene).cuda()
            
        # cfvqa
        z = z_main + z_text
        if z_gene is not None and gene_fact:
            z = z + z_gene
        # z = torch.log(torch.sigmoid(z) + 1e-9)
        z = torch.nn.functional.leaky_relu(z) 
        
        # # cat + linear fusion
        # z = torch.cat((z_main, z_text), dim=1)
        # if z_gene is not None and gene_fact:
        #     z = torch.cat((z, z_gene), dim=1)
        # z = self.fusion_lin(z)
        
        # # add + linear fusion
        # z = z_main + z_text
        # if z_gene is not None and gene_fact:
        #     z = z + z_gene
        # z = self.fusion_lin_add(z)
        
        return z
    #主要修改的地方
    def forward(self, return_feats=False, **kwargs):
        # 【基因级别的表征 ， 通路级别的表征， 文本级别的表征】
        x = kwargs['data_omics'] # 基因数据 [Batch, omic_input_dim]

        # [策略1] FeatureGating：动态噪声过滤，压制不稳定基因
        x_gated = self.feature_gating(x)

        # 1. 基因级别的表征 [Batch, dim]
        gene_level_rep = self.fc_omic(x_gated)

        # [策略2] 显式 Input Dropout：在进入多模态融合前，随机掩码部分基因表征，
        # 迫使后续的 GatedFusion / CrossAttention 不敢"死磕"不稳定的基因分支，增强对文本特征的依赖
        gene_level_rep = self.gene_fusion_dropout(gene_level_rep)
        # print('gene_level_rep.shape:', gene_level_rep.shape)

        #修改通路逻辑，到后面。
        #pathway_level_rep = self.pathway_encoder(x) # 2.通路级别的表征  [batch_size, num_pathways, dim]
        x_text_report = kwargs['x_text_report'][-1] # 文本数据
        text_inputs = self.clinical_bert_tokenizer( # 对文本进行编码
            x_text_report, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(x.device)
        
        # outputs = self.clinical_bert_model(**text_inputs)
        outputs = self.clinical_bert_model(**text_inputs, output_attentions=True)
        # 使用 [CLS] token 的嵌入
        text_embeddings = outputs.last_hidden_state[:, 0, :] # 3.文本级别的表征 [batch_size, dim=768] 
        #======================================================

        #修改后通路逻辑
        # 注意：需确保 __init__ 中已定义 self.use_pathway
        use_pathway = getattr(self, 'use_pathway', True) # 默认 True 以兼容旧代码

        if use_pathway:
            # 如果开启了超图通路，也让超图接收过滤后的高质量信号
            pathway_level_rep = self.pathway_encoder(x_gated) 
            
            # 投影文本特征用于 Attention
            text_emb_proj = self.text_projection(text_embeddings) 
            
            combined_context = torch.cat([gene_level_rep, text_emb_proj], dim=1)
            query_rep = self.query_projection(combined_context)
            query = query_rep.unsqueeze(1)
            
            attn_output, attn_weights = self.cross_attention(
                query=query, key=pathway_level_rep, value=pathway_level_rep
            )
            cross_modal_rep = attn_output.squeeze(1)
            
            # 原有的拼接逻辑 (Gene + Cross + Text)
            cat_embeddings = torch.cat([gene_level_rep, cross_modal_rep, text_embeddings], dim=1)
        else:
            # === 修复的多模态融合：带残差的独立门控 ===
            text_emb_proj = self.text_projection(text_embeddings)  # 768 -> 256

            # 使用 Sigmoid 门控融合
            fused_embeddings, survival_gate_weights = self.gated_fusion_survival([gene_level_rep, text_emb_proj])

            # 引入文本特征的残差连接（Residual Connection），确保即使融合权重坍缩，文本的基线性能也不会丢失
            cat_embeddings = fused_embeddings + text_emb_proj

            # 设置空的 attention weights 防止后面报错
            attn_weights = None




        #0117修改
        # # print('text_embeddings.shape:', text_embeddings.shape)
        # text_embeddings = self.text_projection(text_embeddings) # 变为 dim
        # # print('text_embeddings.shape:', text_embeddings.shape)

        # combined_context = torch.cat([gene_level_rep, text_embeddings], dim=1) # Shape: [B, 512]
        # query_rep = self.query_projection(combined_context) # Shape: [B, 256]
        # # 为MultiheadAttention准备Query (需要增加一个序列长度的维度)
        # query = query_rep.unsqueeze(1) # Shape: [B, 1, 256]
        # attn_output, attn_weights = self.cross_attention(
        #     query=query,
        #     key=pathway_level_rep,
        #     value=pathway_level_rep
        # ) # attn_output Shape: [B, 1, 256]， attn_weights Shape: [B, 1, num_pathways]

        # cross_modal_rep = attn_output.squeeze(1) # Shape: [B, 256]
        # # # 1. 拼接多模态的表征做预测
        # cat_embeddings = torch.cat([gene_level_rep, cross_modal_rep, text_embeddings], dim=1) # Shape: [B, 768]

        # ============ 【修改】收集解释性数据，不直接保存 ============
        # 提取文本attention数据
        attentions = outputs.attentions
        last_layer_attentions = attentions[-1]
        avg_attentions = last_layer_attentions.mean(dim=1).squeeze(0)
        cls_attention_scores = avg_attentions[0, :]
        tokens = self.clinical_bert_tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])
        word_tokens, word_scores = self.aggregate_subword_attentions(tokens, cls_attention_scores.cpu().detach())

        # 定义停用词和标点（用于后续过滤）
        # stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        #                  'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
        #                  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
        #                  'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        #                  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        #                  'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        #                  'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        #                  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
        #                  'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        #                  'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
        #                  'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        #                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        #                  'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

        stop_words = set(['I'])

        import string
        punctuation = set(string.punctuation)

        # 提取pathway attention数据（做修改0117）
        #pathway_names = self.pathway_encoder.hypergraph_structure['pathways']
        #sample_attn_weights = attn_weights[0].squeeze()  # Shape: [num_pathways]
        # === [修改] 安全获取通路解释数据 ===
        if use_pathway and attn_weights is not None:
            pathway_names = self.pathway_encoder.hypergraph_structure['pathways']
            sample_attn_weights = attn_weights[0].squeeze()
        else:
            # 如果没用通路，给空列表，防止报错
            pathway_names = []
            sample_attn_weights = torch.tensor([])

        # 【修改】将解释数据打包成字典，供core_util.py使用
        explanation_data = {
            'word_tokens': word_tokens,
            'word_scores': word_scores,
            'stop_words': stop_words,
            'punctuation': punctuation,
            'pathway_names': pathway_names,
            'pathway_attention_weights': sample_attn_weights,
            # 【已改】添加门控融合权重到解释数据
            'survival_gate_weights': survival_gate_weights if 'survival_gate_weights' in locals() else None,
            'stage_gate_weights': stage_gate_weights if self.enable_multitask and 'stage_gate_weights' in locals() else None
        }
        # =========================================



        # 【修复】根据运行模式选择正确的输入特征
        if self.ab_model == 1:  # 仅文本模式
            cat_embeddings = text_embeddings
            # 仅在第一个batch打印，避免日志刷屏
            if not hasattr(self, '_print_debug'):
                print(f"[Debug] 仅文本模式: cat_embeddings shape = {cat_embeddings.shape}")
                self._print_debug = True
        elif self.ab_model == 2:  # 仅基因模式
            cat_embeddings = gene_level_rep
            # 仅在第一个batch打印，避免日志刷屏
            if not hasattr(self, '_print_debug'):
                print(f"[Debug] 仅基因模式: cat_embeddings shape = {cat_embeddings.shape}")
                self._print_debug = True
        else:  # 多模态融合模式 (ab_model == 3)
            if self.use_pathway:
                # 多模态：基因 + 交叉注意力 + 文本
                cat_embeddings = torch.cat([gene_level_rep, cross_modal_rep, text_embeddings], dim=1)
            else:
                # 多模态：基因 + 文本，使用门控融合 + 残差连接
                text_emb_proj = self.text_projection(text_embeddings)  # 768 -> 256
                fused_embeddings, _ = self.gated_fusion_survival([gene_level_rep, text_emb_proj])
                # 引入文本特征的残差连接
                cat_embeddings = fused_embeddings + text_emb_proj
            # 仅在第一个batch打印，避免日志刷屏
            if not hasattr(self, '_print_debug'):
                print(f"[Debug] 多模态模式: cat_embeddings shape = {cat_embeddings.shape}")
                self._print_debug = True

        # 【已改】临时禁用门控融合，恢复简单相加
        text_pred = self.text_classifier(self.text_projection(text_embeddings))
        gene_pred = self.gene_classifier(gene_level_rep)

        # Survival prediction - 使用选择的特征进行预测
        survival_logits = self.survival_classifier(cat_embeddings)

        assert len(survival_logits.shape) == 2 and survival_logits.shape[1] == self.n_classes

        if self.use_debias:
        # TE 
            logit_te = self.fusion(survival_logits, text_pred, gene_pred,
                                    main_fact=True, text_fact=True, gene_fact=False)
            logits_nde = self.fusion(survival_logits.clone().detach(), text_pred.clone().detach(), gene_pred,
                                main_fact=False, text_fact=True, gene_fact=False) # NDE
            if self.weight_debiase: # 指定 去偏强度
                logits_tie = logit_te - self.weight_debiase * logits_nde
            else: # 可学习的去偏强度
                nde_weight = torch.sigmoid(self.debiase_weight)
                logits_tie = logit_te - nde_weight * logits_nde
            
            logits_text = text_pred
        if self.enable_multitask:
            # Stage prediction - 根据运行模式选择正确的输入特征
            if self.ab_model == 1:  # 仅文本模式
                stage_input = text_embeddings
                print(f"[Debug] 仅文本模式(Stage): stage_input shape = {stage_input.shape}")
            elif self.ab_model == 2:  # 仅基因模式
                stage_input = gene_level_rep
                print(f"[Debug] 仅基因模式(Stage): stage_input shape = {stage_input.shape}")
            else:  # 多模态融合模式 (ab_model == 3)
                if self.use_pathway:
                    stage_input = torch.cat((gene_level_rep, text_embeddings), dim=1)
                else:
                    # 无通路模式：使用门控融合 + 残差连接
                    text_emb_proj = self.text_projection(text_embeddings)
                    fused_stage, _ = self.gated_fusion_stage([gene_level_rep, text_emb_proj])
                    stage_input = fused_stage + text_emb_proj
                print(f"[Debug] 多模态模式(Stage): stage_input shape = {stage_input.shape}")

            # 输入进分类器
            stage_logits = self.stage_classifier(stage_input)
            assert len(stage_logits.shape) == 2 and stage_logits.shape[1] == self.n_stage_classes
            #logits_text_stage = self.stage_classifier_text(text_embeddings)
            if 'outputs' in locals():
                raw_bert_cls = outputs.last_hidden_state[:, 0, :] # Shape: [Batch, 768]
                logits_text_stage = self.stage_classifier_text(raw_bert_cls)
            else:
                # 万一 outputs 不存在 (理论上不会)，回退到旧逻辑
                logits_text_stage = self.stage_classifier_text(text_embeddings)
        else:
            stage_logits = None

        # 【修改】所有返回值后面都添加explanation_data
        if self.use_debias and self.enable_multitask and not self.use_align_loss:
            return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, explanation_data
        if not self.use_debias and self.enable_multitask:
            return survival_logits, stage_logits, explanation_data
        if not self.use_debias and not self.enable_multitask:
            return survival_logits, explanation_data
        if self.use_debias and self.enable_multitask and self.use_align_loss:
            return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss, explanation_data
        if self.use_debias and not self.enable_multitask:
            return survival_logits, logits_text, logit_te, logits_tie, logits_nde, explanation_data

    def forward_0(self, return_feats=False, **kwargs):
        # print('kwargs: ', kwargs.keys())
        # phrase = kwargs['phrase']
        # 【基因级别的表征 ， 通路级别的表征， 文本级别的表征】
        x = kwargs['data_omics']
        gene_level_rep = self.fc_omic(x) # 1.基因级别的表征 [1, dim]
        # print('gene_level_rep.shape:', gene_level_rep.shape)

        pathway_level_rep = self.pathway_encoder(x) # 2.通路级别的表征  [batch_size, num_pathways, dim]
        # print('pathway_level_rep.shape:', pathway_level_rep.shape)
        # pathway_level_rep = pathway_level_rep.

        x_text_report = kwargs['x_text_report'][-1] 
        text_inputs = self.clinical_bert_tokenizer(
            x_text_report, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(x.device)
        
        # outputs = self.clinical_bert_model(**text_inputs)
        outputs = self.clinical_bert_model(**text_inputs, output_attentions=True)
        # 使用 [CLS] token 的嵌入
        text_embeddings = outputs.last_hidden_state[:, 0, :] # 3.文本级别的表征 [batch_size, dim] 
        # print('text_embeddings.shape:', text_embeddings.shape)
        text_embeddings = self.text_projection(text_embeddings) # 变为 dim
        # print('text_embeddings.shape:', text_embeddings.shape)

        combined_context = torch.cat([gene_level_rep, text_embeddings], dim=1) # Shape: [B, 512]
        query_rep = self.query_projection(combined_context) # Shape: [B, 256]
        # 为MultiheadAttention准备Query (需要增加一个序列长度的维度)
        query = query_rep.unsqueeze(1) # Shape: [B, 1, 256]
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=pathway_level_rep,
            value=pathway_level_rep
        ) # attn_output Shape: [B, 1, 256]， attn_weights Shape: [B, 1, num_pathways]

        cross_modal_rep = attn_output.squeeze(1) # Shape: [B, 256]
        # # 1. 拼接多模态的表征做预测
        cat_embeddings = torch.cat([gene_level_rep, cross_modal_rep, text_embeddings], dim=1) # Shape: [B, 768]
        # print('cat_embeddings.shape:', cat_embeddings.shape)

        # # 2. 直接用 cross_modal_rep 做预测
        # cat_embeddings = cross_modal_rep
        # survival_logits = self.survival_classifier(cat_embeddings)

        # 3. 残差 链接多模态
        # cat_embeddings = cross_modal_rep + gene_level_rep + text_embeddings
        # survival_logits = self.survival_classifier(cat_embeddings)




        # if phrase == 'test':
        print('test mode')
        attentions = outputs.attentions 
        last_layer_attentions = attentions[-1]
        avg_attentions = last_layer_attentions.mean(dim=1).squeeze(0)
        cls_attention_scores = avg_attentions[0, :]
        tokens = self.clinical_bert_tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])
        word_tokens, word_scores = self.aggregate_subword_attentions(tokens, cls_attention_scores.cpu().detach())

        stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

        punctuation = set(string.punctuation)

        filtered_words = []
        filtered_scores = []
        
        for word, score in zip(word_tokens, word_scores):
            # 过滤条件：单词小写后不在停用词列表，且单词本身不在标点符号列表
            if word.lower() not in stop_words and word not in punctuation:
                filtered_words.append(word)
                filtered_scores.append(score.item()) # .item() 将0维张量转为数字
        # --- 过滤结束 ---


        # 打印最重要的Top-15个【经过滤的完整单词】
        import pandas as pd
        df = pd.DataFrame({'word': filtered_words, 'attention': filtered_scores})
        df = df.sort_values(by='attention', ascending=False)
        # print("\n--- [模型文本解释 (已过滤)] ---")
        # print("Top 15 Most Attended Keywords:")
        # print(df.head(15))
        # print("-------------------------------------\n")

        # 9. 生成高亮文本进行可视化 (基于完整单词)
        # self.visualize_keywords_in_context(word_tokens, word_scores, stop_words, punctuation)


        pathway_names = self.pathway_encoder.hypergraph_structure['pathways']
        sample_attn_weights = attn_weights[0].squeeze() # 形状变为 [num_pathways]
        
        # 调用可视化函数
        self.visualize_top_pathways(sample_attn_weights, pathway_names, top_k=15)

        
        if self.only_omic:
            # h_omic = self.fc_omic(x)
            # cat_embeddings = h_omic
            cat_embeddings = gene_level_rep
        if self.only_text:
            cat_embeddings = text_embeddings
        
        # Survival prediction
        # cat_embeddings = self.dropout_omic(cat_embeddings)

        survival_logits = self.survival_classifier(cat_embeddings)
        
        # survival_logits = self.survival_classifier(h_omic)
        assert len(survival_logits.shape) == 2 and survival_logits.shape[1] == self.n_classes
        text_pred = self.text_classifier(text_embeddings)
        gene_pred = self.gene_classifier(gene_level_rep)
        
        if self.use_debias:
        # TE 
            logit_te = self.fusion(survival_logits, text_pred, gene_pred,
                                    main_fact=True, text_fact=True, gene_fact=False)
            # logits_text = self.fusion(survival_logits, text_pred, gene_pred,
            #                     main_fact=False, text_fact=True, gene_fact=False) 
            logits_nde = self.fusion(survival_logits.clone().detach(), text_pred.clone().detach(), gene_pred,
                                main_fact=False, text_fact=True, gene_fact=False) # NDE
            if self.weight_debiase: # 指定 去偏强度
                logits_tie = logit_te - self.weight_debiase * logits_nde
            else: # 可学习的去偏强度
                nde_weight = torch.sigmoid(self.debiase_weight)
                logits_tie = logit_te - nde_weight * logits_nde
            
            logits_text = text_pred
            # nde_weight 是通过 text_embeddings 得到的一个可学习参数
            # nde_weight = torch.sigmoid(self.lin_weight(text_embeddings))
            # logits_tie = logit_te - nde_weight * logits_nde
                
        if self.enable_multitask:
            # Stage prediction
            stage_logits = self.stage_classifier(cat_embeddings)
            assert len(stage_logits.shape) == 2 and stage_logits.shape[1] == self.n_stage_classes
            logits_text_stage = self.text_classifier_stage(text_embeddings)
            # gene_logits_stage = self.gene_classifier_stage(h_omic)
            # logit_te_stage = self.fusion_stage(stage_logits, text_logits_stage, gene_logits_stage,
            #                                     main_fact=True, text_fact=True, gene_fact=False)
            # logits_nde_stage = self.fusion_stage(stage_logits.clone().detach(), text_logits_stage.clone().detach(), gene_logits_stage,
            #                                     main_fact=False, text_fact=True, gene_fact=False) # NDE
            # nde_weight_stage2 = torch.sigmoid(self.debiase_weight2)
            # logit_te_stage = logit_te_stage + nde_weight_stage2 * logits_nde_stage
            
        else:
            stage_logits = None
            
        if self.use_debias and self.enable_multitask and not self.use_align_loss:
            return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde
        if not self.use_debias and self.enable_multitask:
            return survival_logits, stage_logits
        if not self.use_debias and not self.enable_multitask:
            return survival_logits
        if self.use_debias and self.enable_multitask and self.use_align_loss:
            # print('return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss: ')
            return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss
        if self.use_debias and not self.enable_multitask:
            return survival_logits, logits_text, logit_te, logits_tie, logits_nde
    
    def forward00(self, return_feats=False, **kwargs):
        x = kwargs['data_omics']
        # print('x shape: ', x.shape)
        Hy_gen = self.pathway_encoder(x) # [num_gen, hidden_dim=1]
        # print('Hy_gen.shape:', Hy_gen.shape)
        x_text_report = kwargs['x_text_report'][-1]
        text_inputs = self.clinical_bert_tokenizer(
            x_text_report, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(x.device)
        
        outputs = self.clinical_bert_model(**text_inputs)
        # 使用 [CLS] token 的嵌入
        text_embeddings = outputs.last_hidden_state[:, 0, :] # [batch_size, hidden_dim]
        
        hint_emb = Hy_gen
        
        # 残差
        if self.use_resnet:
            hint_emb = hint_emb + x
        hint_emb = self.fc_omic(hint_emb)
        if self.NOT_hygraph:
            # 直接将原始的 x 输入 SNN ，得到 hint_emb
            hint_emb = self.fc_omic(x)
        # hint_emb = self.dropout_omic(hint_emb)
        
        if self.use_align_loss:
            # text_embeddings_align = self.lin_text_emb(text_embeddings)
            cos_align_loss = self.alignment_model.alignment_loss(hint_emb, text_embeddings)
            aligned_gene, aligned_wsi = self.alignment_model.forward(hint_emb, text_embeddings)
            hint_emb = aligned_gene 
            text_embeddings = aligned_wsi
        cat_embeddings = torch.cat((hint_emb, text_embeddings), dim=1)
        if self.only_omic:
            # h_omic = self.fc_omic(x)
            # cat_embeddings = h_omic
            cat_embeddings = hint_emb
        if self.only_text:
            cat_embeddings = text_embeddings
        
        # Survival prediction
        # cat_embeddings = self.dropout_omic(cat_embeddings)
        survival_logits = self.survival_classifier(cat_embeddings)
        
        # survival_logits = self.survival_classifier(h_omic)
        assert len(survival_logits.shape) == 2 and survival_logits.shape[1] == self.n_classes
        text_pred = self.text_classifier(text_embeddings)
        gene_pred = self.gene_classifier(hint_emb)
        
        if self.use_debias:
        # TE 
            logit_te = self.fusion(survival_logits, text_pred, gene_pred,
                                    main_fact=True, text_fact=True, gene_fact=False)
            # logits_text = self.fusion(survival_logits, text_pred, gene_pred,
            #                     main_fact=False, text_fact=True, gene_fact=False) 
            logits_nde = self.fusion(survival_logits.clone().detach(), text_pred.clone().detach(), gene_pred,
                                main_fact=False, text_fact=True, gene_fact=False) # NDE
            if self.weight_debiase: # 指定 去偏强度
                logits_tie = logit_te - self.weight_debiase * logits_nde
            else: # 可学习的去偏强度
                nde_weight = torch.sigmoid(self.debiase_weight)
                logits_tie = logit_te - nde_weight * logits_nde
            
            logits_text = text_pred
            # nde_weight 是通过 text_embeddings 得到的一个可学习参数
            # nde_weight = torch.sigmoid(self.lin_weight(text_embeddings))
            # logits_tie = logit_te - nde_weight * logits_nde
                
        if self.enable_multitask:
            # Stage prediction
            stage_logits = self.stage_classifier(cat_embeddings)
            assert len(stage_logits.shape) == 2 and stage_logits.shape[1] == self.n_stage_classes
            logits_text_stage = self.text_classifier_stage(text_embeddings)
            # gene_logits_stage = self.gene_classifier_stage(h_omic)
            # logit_te_stage = self.fusion_stage(stage_logits, text_logits_stage, gene_logits_stage,
            #                                     main_fact=True, text_fact=True, gene_fact=False)
            # logits_nde_stage = self.fusion_stage(stage_logits.clone().detach(), text_logits_stage.clone().detach(), gene_logits_stage,
            #                                     main_fact=False, text_fact=True, gene_fact=False) # NDE
            # nde_weight_stage2 = torch.sigmoid(self.debiase_weight2)
            # logit_te_stage = logit_te_stage + nde_weight_stage2 * logits_nde_stage
            
        else:
            stage_logits = None
            
        if self.use_debias and self.enable_multitask and not self.use_align_loss:
            return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde
        if not self.use_debias and self.enable_multitask:
            return survival_logits, stage_logits
        if not self.use_debias and not self.enable_multitask:
            return survival_logits
        if self.use_debias and self.enable_multitask and self.use_align_loss:
            # print('return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss: ')
            return survival_logits, stage_logits, logits_text, logit_te, logits_tie, logits_nde, cos_align_loss
        if self.use_debias and not self.enable_multitask:
            return survival_logits, logits_text, logit_te, logits_tie, logits_nde
    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
        else:
            self.fc_omic = self.fc_omic.to(device)

        self.survival_classifier = self.survival_classifier.to(device)
        if self.enable_multitask:
            self.stage_classifier = self.stage_classifier.to(device)


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()
