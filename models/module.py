import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class CrossModalEncoder(nn.Module):
    def __init__(self, hidden_size, n_head, dropout, kernel_size: list =[9, 1]):
        super(CrossModalEncoder, self).__init__()
        self.MHA = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_head, batch_first=True)
        self.MHA_dropout = nn.Dropout(dropout)
        self.MHA_norm = nn.LayerNorm(hidden_size)

        # FFN
        self.w_1 = nn.Conv1d(
            hidden_size, 
            hidden_size*4,
            kernel_size=kernel_size[0],
            padding=(kernel_size[0]-1)//2
        )
        self.w_2 = nn.Conv1d(
            hidden_size*4, 
            hidden_size,
            kernel_size=kernel_size[1],
            padding=(kernel_size[1]-1)//2
        )
        self.dropout = nn.Dropout(dropout)
        self.FFN_norm = nn.LayerNorm(hidden_size)
        # 
    def forward(self, query, memory):
        x = query
        x = self.MHA_norm(x + self.attn_block(x, memory))
        x = self.FFN_norm(x + self.ffn_block(x))
        return x 

    def attn_block(self, query, memory):
        x = self.MHA(query, memory, memory, need_weights=False)[0]
        return self.MHA_dropout(x)
    
    def ffn_block(self, x):
        x = self.w_2(F.relu(self.w_1(x.transpose(1, 2)))).transpose(1, 2)
        return self.dropout(x)