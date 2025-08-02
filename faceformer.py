import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec import Wav2Vec2Model
import random

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
import os
class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        
        # 双流音频编码器
        # Wav2Vec2 编码器
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            os.environ["WAV2VEC_PATH"]
        )
        self.audio_encoder.feature_extractor._freeze_parameters()
        
        # HuBERT 编码器
        self.hubert_encoder = Wav2Vec2Model.from_pretrained(
            os.environ["HUBERT_PATH"]  # 需要设置环境变量或直接指定路径
        )
        self.hubert_encoder.feature_extractor._freeze_parameters()
        
        # 特征映射层
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        self.hubert_feature_map = nn.Linear(768, args.feature_dim)
        
        # 注意力融合模块
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=args.feature_dim,
            num_heads=4,
            batch_first=True
        )
        self.fusion_layer_norm = nn.LayerNorm(args.feature_dim)
        
        # 运动编码器
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        
        # 周期性位置编码
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period=args.period)
        
        # 时序偏置掩码
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=600, period=args.period)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.feature_dim,
            nhead=4,
            dim_feedforward=2*args.feature_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # 运动解码器
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        
        # 风格嵌入
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        
        # 跨模态对齐投影层
        self.cross_modal_proj = nn.Linear(args.feature_dim, args.feature_dim)
        
        # 初始化
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True):
        template = template.unsqueeze(1)  # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        frame_num = vertice.shape[1]
        
        # 双流特征提取
        # Wav2Vec2特征
        wav2vec_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        # HuBERT特征
        hubert_states = self.hubert_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        
        # 数据集特定长度调整
        if self.dataset == "BIWI":
            if wav2vec_states.shape[1] < frame_num * 2:
                frame_num = wav2vec_states.shape[1] // 2
                vertice = vertice[:, :frame_num]
            # 确保两个特征长度一致
            hubert_states = hubert_states[:, :wav2vec_states.shape[1]]
        
        # 特征映射
        wav2vec_feat = self.audio_feature_map(wav2vec_states)
        hubert_feat = self.hubert_feature_map(hubert_states)
        
        # 注意力融合
        fused_feat, _ = self.fusion_attention(
            wav2vec_feat,
            hubert_feat,
            hubert_feat
        )
        fused_feat = self.fusion_layer_norm(fused_feat + wav2vec_feat)
        
        # 跨模态对齐损失计算
        # 投影到相同空间
        vertice_proj = self.cross_modal_proj(self.vertice_map(vertice))
        audio_proj = self.cross_modal_proj(fused_feat)
        
        # 对比损失（正样本：相同时间步，负样本：随机采样）
        contrastive_loss = 0
        for t in range(frame_num):
            pos_sim = F.cosine_similarity(vertice_proj[:, t], audio_proj[:, t], dim=-1)
            # 随机选择负样本
            neg_idx = random.randint(0, frame_num-1)
            while neg_idx == t:
                neg_idx = random.randint(0, frame_num-1)
            neg_sim = F.cosine_similarity(vertice_proj[:, t], audio_proj[:, neg_idx], dim=-1)
            
            contrastive_loss += F.relu(neg_sim - pos_sim + 0.2)  # margin=0.2
        
        contrastive_loss = contrastive_loss / frame_num
        
        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
            style_emb = vertice_emb
            vertice_input = torch.cat((template, vertice[:, :-1]), 1)  # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], fused_feat.shape[1])
            
            vertice_out = self.transformer_decoder(vertice_input, fused_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            vertice_emb = obj_embedding.unsqueeze(1)
            style_emb = vertice_emb
            
            for i in range(frame_num):
                vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], fused_feat.shape[1])
                
                vertice_out = self.transformer_decoder(vertice_input, fused_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                
                new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)
        
        vertice_out = vertice_out + template
        reconstruction_loss = criterion(vertice_out, vertice)
        reconstruction_loss = torch.mean(reconstruction_loss)
        
        # 总损失 = 重建损失 + 跨模态对齐损失
        total_loss = reconstruction_loss + 0.5 * contrastive_loss
        
        return total_loss
    def predict(self, audio, template, one_hot):
        template = template.unsqueeze(1)  # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)
        
        # 双流特征提取
        wav2vec_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        hubert_states = self.hubert_encoder(audio, self.dataset).last_hidden_state
        
        if self.dataset == "BIWI":
            frame_num = wav2vec_states.shape[1] // 2
            # 对齐特征长度
            hubert_states = hubert_states[:, :wav2vec_states.shape[1]]
        elif self.dataset == "vocaset":
            frame_num = wav2vec_states.shape[1]
        
        # 特征映射
        wav2vec_feat = self.audio_feature_map(wav2vec_states)
        hubert_feat = self.hubert_feature_map(hubert_states)
        
        # 注意力融合
        fused_feat, _ = self.fusion_attention(
            wav2vec_feat,
            hubert_feat,
            hubert_feat
        )
        fused_feat = self.fusion_layer_norm(fused_feat + wav2vec_feat)
        
        vertice_emb = obj_embedding.unsqueeze(1)
        style_emb = vertice_emb
        
        for i in range(frame_num):
            vertice_input = self.PPE(vertice_emb)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], fused_feat.shape[1])
            
            vertice_out = self.transformer_decoder(vertice_input, fused_feat, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
            
            new_output = self.vertice_map(vertice_out[:, -1, :]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
        
        vertice_out = vertice_out + template
        return vertice_out
