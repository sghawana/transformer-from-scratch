import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import SDPA, MHA, DEVICE, DTYPE

class EncoderLayer(nn.Module):
    def __init__(self, in_dimension, out_dimension, kq_dimension,
                 num_heads=8, linear_stretch=2,
                 device=DEVICE, dtype=DTYPE
                ):
        super(EncoderLayer, self).__init__()
        
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        self.num_heads = num_heads
        self.linear_stretch = linear_stretch
        
        self.device = device
        self.dtype = dtype
        
        self.mha = MHA(self.in_dimension, self.out_dimension, self.kq_dimension, self.num_heads,
                       self.device, self.dtype)
        
        self.ff1 = nn.Linear(self.out_dimension, self.linear_stretch*self.out_dimension,
                             device=self.device, dtype=self.dtype
                            )
        
        self.ff2 = nn.Linear(self.linear_stretch*self.out_dimension, self.out_dimension,
                             device=self.device, dtype=self.dtype
                            )
        
        self.layernorm1 = nn.LayerNorm(self.out_dimension, device=self.device, dtype=self.dtype)
        self.layernorm2 = nn.LayerNorm(self.out_dimension, device=self.device, dtype=self.dtype)
        
    def forward(self, x):
        residual = x
        Y = self.mha(x)
        Y = self.layernorm1(Y + residual) 
        
        residual = Y
        Z = self.ff1(Y)
        Z = F.relu(Z)
        Z = self.ff2(Z)
        Z = self.layernorm2(Z + residual)
        return Z
    

class ENCODER(nn.Module):
    def __init__(self, encoder_dimension, kq_dimension, vocab_size, seq_len,
                 num_heads=8, linear_stretch=2, num_layers=6, padding_index=0,
                 use_pos_enc=True, device=DEVICE, dtype=DTYPE
                ):
        super(ENCODER, self).__init__()
        
        self.encoder_dim = encoder_dimension
        self.kq_dimension = kq_dimension
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        self.num_heads = num_heads
        self.linear_stretch = linear_stretch
        self.num_layers = num_layers
        self.use_pos_enc = use_pos_enc
        self.padding_index = padding_index
        
        self.device = device
        self.dtype = dtype
        
        self.embd = nn.Embedding(self.vocab_size, self.encoder_dim, padding_idx=self.padding_index,
                                dtype=self.dtype, device=self.device)
        
        self.layer_list = nn.ModuleList([])
        for _ in range(self.num_layers):
            layer = EncoderLayer(self.encoder_dim, self.encoder_dim, self.kq_dimension,
                 num_heads=self.num_heads, linear_stretch=self.linear_stretch,
                 device=self.device, dtype=self.dtype
                )
            self.layer_list.append(layer)
        
    
    def positional_encoding(self, d_model):
        position = torch.arange(0, self.seq_len, dtype=self.dtype, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=self.dtype, device=self.device) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(self.seq_len, d_model, dtype=self.dtype, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        Z = self.embd(x) * torch.sqrt(torch.tensor(self.encoder_dim, dtype=self.dtype, device=self.device))
        if self.use_pos_enc:
            pe = self.positional_encoding(self.encoder_dim)
            Z = Z + pe
        for layer in self.layer_list:
            Z = layer(Z)
        return Z