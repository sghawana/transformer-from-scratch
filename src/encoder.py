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
    
    