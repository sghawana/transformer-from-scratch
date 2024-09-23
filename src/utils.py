import torch
import torch.nn as nn
import torch.nn.functional as F

import math

DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
DTYPE = torch.float32

### Scaled dot Product Attention
class SDPA(nn.Module):
    def __init__(self, in_dimension, out_dimension, kq_dimension,
                masked=False, device=DEVICE, dtype=DTYPE):
        super(SDPA, self).__init__()
        
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        
        self.masked = masked
        self.device = device
        self.dtype = dtype

        self.query = nn.Linear(in_features=self.in_dimension, out_features=self.kq_dimension,
                               device=self.device, dtype=self.dtype
        )
        
        self.key = nn.Linear(in_features=self.in_dimension, out_features=self.kq_dimension,
                               device=self.device, dtype=self.dtype
        )
        
        self.value = nn.Linear(in_features=self.in_dimension, out_features=self.out_dimension,
                               device=self.device, dtype=self.dtype
        )

    def forward(self,x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        kq = math.sqrt(self.kq_dimension)
        attn = torch.matmul(Q,K.permute(0,2,1))/kq
        if self.masked:
            attn_mask = torch.triu(torch.ones_like(attn), diagonal=1).bool()
            attn.masked_fill_(attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        Z = torch.matmul(attn,V)
        return Z
    
    
### Multihead attention
class MHA(nn.Module):
    def __init__(self, in_dimension, out_dimension, kq_dimension, num_heads=8,
                 masked=False, device=DEVICE, dtype=DTYPE
                ):
        super(MHA, self).__init__()
        
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        self.num_heads = num_heads
        
        self.masked = masked
        self.device = device
        self.dtype = dtype
        
        self.sdpa_head_list = nn.ModuleList([])
        for _ in range(num_heads):
            sdpa_head = SDPA(self.in_dimension, self.out_dimension, self.kq_dimension,
                            masked=self.masked, device=DEVICE, dtype=DTYPE)
            self.sdpa_head_list.append(sdpa_head)
            
        self.out = nn.Linear(in_features=num_heads*out_dimension, out_features=out_dimension,
                             device=self.device, dtype=self.dtype
                            )
            
    def forward(self, x):
        y_list = []
        for head in self.sdpa_head_list:
            y = head(x)
            y_list.append(y)
        
        Y = torch.cat(y_list, dim=-1)
        Z = self.out(Y)
        return Z
    
    
### Cross Scaled dot Product Attention
class Cross_SDPA(nn.Module):
    def __init__(self, enc_inp_dim, dec_inp_dim, out_dimension, kq_dimension,
                device=DEVICE, dtype=DTYPE):
        super(Cross_SDPA, self).__init__()
        
        self.enc_inp_dim = enc_inp_dim
        self.dec_inp_dim = dec_inp_dim
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        
        self.device = device
        self.dtype = dtype

        self.query = nn.Linear(in_features=self.dec_inp_dim, out_features=self.kq_dimension,
                               device=self.device, dtype=self.dtype
                            )
        
        self.key = nn.Linear(in_features=self.enc_inp_dim, out_features=self.kq_dimension,
                               device=self.device, dtype=self.dtype
                            )
        
        self.value = nn.Linear(in_features=self.enc_inp_dim, out_features=self.out_dimension,
                               device=self.device, dtype=self.dtype
                            )

    def forward(self, z, y):
        Q = self.query(y)
        K = self.key(z)
        V = self.value(z)
        
        kq = math.sqrt(self.kq_dimension)
        attn = torch.matmul(Q,K.permute(0,2,1))/kq
        attn = F.softmax(attn, dim=-1)
        
        Y = torch.matmul(attn,V)
        return Y
    
### Cross Multi head Attention
class Cross_MHA(nn.Module):
    def __init__(self, enc_inp_dim, dec_inp_dim, out_dimension, kq_dimension, num_heads=8,
                 device=DEVICE, dtype=DTYPE
                ):
        super(Cross_MHA, self).__init__()
        
        self.enc_inp_dim = enc_inp_dim
        self.dec_inp_dim = dec_inp_dim
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        self.num_heads = num_heads
        
        self.device = device
        self.dtype = dtype
        
        self.sdpa_head_list = nn.ModuleList([])
        for _ in range(num_heads):
            sdpa_head = Cross_SDPA(self.enc_inp_dim, self.dec_inp_dim, self.out_dimension,
                                   self.kq_dimension, device=self.device, dtype=self.dtype
                                )
            self.sdpa_head_list.append(sdpa_head)
            
        self.out = nn.Linear(in_features=num_heads*out_dimension, out_features=out_dimension,
                             device=self.device, dtype=self.dtype
                            )
            
    def forward(self, z, y):
        w_list = []
        for head in self.sdpa_head_list:
            w = head(z, y)
            w_list.append(w)
        
        W = torch.cat(w_list, dim=-1)
        W = self.out(W)
        return W
    

