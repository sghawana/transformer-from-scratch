import torch
import torch.nn as nn
import torch.nn.functional as F

import math

DEVICE = torch.device('cuda', 1)
DTYPE = torch.float32

### Scaled dot Product Attention
class SDPA(nn.Module):
    def __init__(self, in_dimension, out_dimension, kq_dimension, device=DEVICE, dtype=DTYPE):
        super(SDPA, self).__init__()
        
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
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
        attn = F.softmax(attn, dim=0)
        
        Z = torch.matmul(attn,V)
        return Z
    
    
### Multihead attention
class MHA(nn.Module):
    def __init__(self, in_dimension, out_dimension, kq_dimension, num_heads=8,
                 device=DEVICE, dtype=DTYPE
                ):
        super(MHA, self).__init__()
        
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        self.num_heads = num_heads
        
        self.device = device
        self.dtype = dtype
        
        self.sdpa_head_list = nn.ModuleList([])
        for _ in range(num_heads):
            sdpa_head = SDPA(self.in_dimension, self.out_dimension, self.kq_dimension,
                            device=DEVICE, dtype=DTYPE)
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