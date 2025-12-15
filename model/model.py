# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio
from torchaudio.models import Conformer



class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_weights = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # x: (batch, length, dim)
        attention_scores = self.attention_weights(x)  # (batch, length, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, length, 1)
        weighted_sum = torch.sum(x * attention_weights, dim=1)  # (batch, dim)
        return weighted_sum

class Classifier(nn.Module):
    """分类器模型"""
    def __init__(self,d_model, n_spks, nhead ,num_layers, 
                dim_feedforward,dropout,use_self_attention_pool):
        super().__init__()
        
        # 输入投影层
        self.prenet = nn.Linear(40, d_model)
        
        # Conformer编码层
        self.encoder_layer = torchaudio.models.Conformer(
            input_dim=d_model, 
            num_heads=nhead,
            ffn_dim=dim_feedforward,
            num_layers=num_layers,
            depthwise_conv_kernel_size=15, 
            dropout=dropout
        )
        self.use_self_attention_pool = use_self_attention_pool
        if self.use_self_attention_pool:
            self.pooling = SelfAttentionPooling(d_model)
        
        
        # 预测层
        self.pred_layer = nn.Sequential(
            nn.BatchNorm1d(d_model),
           
            nn.Linear(d_model, n_spks),
        )
    
    def forward(self, mels):
        """
        Args:
            mels: (batch size, length, 40)
        Return:
            out: (batch size, n_spks)
        """
        # 投影到d_model维度
        out = self.prenet(mels)  # (batch, length, d_model)
        lengths = torch.full((out.size(0),), out.size(1), dtype=torch.long, device=out.device)
    
        # Conformer编码
        out,_ = self.encoder_layer(out, lengths) # (batch, length, d_model)
        
        

        
        # 使用自注意力池化或者平均池化
        stats = self.pooling(out) if self.use_self_attention_pool else out.mean(dim=1)

        # 分类输出
        out = self.pred_layer(stats)  # (batch, n_spks)
        return out





# class Classifier(nn.Module):
#     """分类器模型"""
#     def __init__(self, d_model, n_spks, nhead ,num_layers, dim_feedforward,dropout,use_self_attention_pool):
#         super().__init__()
        
#         # 输入投影层
#         self.prenet = nn.Linear(40, d_model)
        
#         # Transformer编码层
#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, 
#             dim_feedforward=dim_feedforward, 
#             nhead=nhead,
#             dropout=dropout,
            
#             batch_first=True,

#         )
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
#         # 预测层
#         self.pred_layer = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.ReLU(),
#             nn.Linear(d_model, n_spks),
#         )
    
#     def forward(self, mels):
#         """
#         Args:
#             mels: (batch size, length, 40)
#         Return:
#             out: (batch size, n_spks)
#         """
#         # 投影到d_model维度
#         out = self.prenet(mels)  # (batch, length, d_model)
        
#         # 调整维度以适应Transformer
#         out = out.permute(1, 0, 2)  # (length, batch, d_model)
        
#         # Transformer编码
#         out = self.encoder(out)
        
#         # 恢复维度
#         out = out.transpose(0, 1)  # (batch, length, d_model)
        
#         # 平均池化
#         stats = out.mean(dim=1)
        
#         # 分类输出
#         out = self.pred_layer(stats)  # (batch, n_spks)
#         return out
