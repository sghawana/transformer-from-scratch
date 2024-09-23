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
    def __init__(self, model_dimension, kq_dimension, vocab_size,
                 num_heads=8, linear_stretch=2, num_layers=6,
                 use_pos_enc=True, device=DEVICE, dtype=DTYPE
                ):
        super(ENCODER, self).__init__()
        
        self.model_dim = model_dimension
        self.kq_dimension = kq_dimension
        
        self.num_heads = num_heads
        self.linear_stretch = linear_stretch
        self.num_layers = num_layers
        self.use_pos_enc = use_pos_enc
        
        self.device = device
        self.dtype = dtype
        
        self.embd = nn.Embedding(vocab_size, self.model_dim, padding_idx=0,
                                dtype=self.dtype, device=self.device)
        
        self.layer_list = nn.ModuleList([])
        for _ in range(self.num_layers):
            layer = EncoderLayer(self.model_dim, self.model_dim, self.kq_dimension,
                 num_heads=self.num_heads, linear_stretch=self.linear_stretch,
                 device=self.device, dtype=self.dtype
                )
            self.layer_list.append(layer)
        
    
    def positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len, dtype=self.dtype, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=self.dtype, device=self.device) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model, dtype=self.dtype, device=self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x):
        Z = self.embd(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=self.dtype, device=self.device))
        if self.use_pos_enc:
            seq_len = x.size(1) 
            pe = self.positional_encoding(seq_len, self.model_dim)
            Z = Z + pe
        print(Z.shape)
        for layer in self.layer_list:
            Z = layer(Z)
        return Z
        
        
        
        
          