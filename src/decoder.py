import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from utils import SDPA, MHA, Cross_MHA, Cross_SDPA, DEVICE, DTYPE

class DecoderLayer(nn.Module):
    def __init__(self, enc_inp_dim, dec_inp_dim, out_dimension, kq_dimension,
                 num_heads=8, linear_stretch=2, dropout=0.1,
                 device=DEVICE, dtype=DTYPE
                ):
        super(DecoderLayer, self).__init__()
        
        self.enc_inp_dim = enc_inp_dim
        self.dec_inp_dim = dec_inp_dim
        
        self.out_dimension = out_dimension
        self.kq_dimension = kq_dimension
        self.num_heads = num_heads
        self.linear_stretch = linear_stretch
        self.dropout = dropout
        
        self.device = device
        self.dtype = dtype
        
        self.masked_mha = MHA(self.dec_inp_dim, self.out_dimension, self.kq_dimension, self.num_heads,
                                True, self.device, self.dtype
                            )
        
        self.cross_mha = Cross_MHA(self.enc_inp_dim, self.dec_inp_dim, self.out_dimension,
                                   self.kq_dimension, self.num_heads, device=self.device,
                                   dtype=self.dtype
                                )
        
        self.ff1 = nn.Linear(self.out_dimension, self.linear_stretch*self.out_dimension,
                             device=self.device, dtype=self.dtype
                            )
        
        self.ff2 = nn.Linear(self.linear_stretch*self.out_dimension, self.out_dimension,
                             device=self.device, dtype=self.dtype
                            )
        
        self.layernorm1 = nn.LayerNorm(self.out_dimension, device=self.device, dtype=self.dtype)
        self.layernorm2 = nn.LayerNorm(self.out_dimension, device=self.device, dtype=self.dtype)
        self.layernorm3 = nn.LayerNorm(self.out_dimension, device=self.device, dtype=self.dtype)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, e):
        residual = x
        Y = self.masked_mha(x)
        Y = self.layernorm1(Y + residual) 
        Y = self.dropout(Y)
        
        residual = Y
        Z = self.cross_mha(e, Y)
        Z = self.layernorm2(Z + residual)
        Z = self.dropout(Z)
        
        residual = Z
        W = self.ff1(Z)
        W = F.relu(W)
        W = self.ff2(W)
        W = self.layernorm3(W + residual)
        W = self.dropout(W)
        return W
    
class DECODER(nn.Module):
    def __init__(self, decoder_dimension, encoder_dimension, kq_dimension, vocab_size,
                 max_seq_len, num_heads=8, linear_stretch=2, num_layers=6, padding_index=0,
                 use_pos_enc=True, dropout=0.1, device=DEVICE, dtype=DTYPE):
        super(DECODER, self).__init__()
        self.decoder_dim = decoder_dimension
        self.encoder_dim = encoder_dimension
        self.kq_dimension = kq_dimension
        self.vocab_size = vocab_size
        
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.linear_stretch = linear_stretch
        self.num_layers = num_layers
        self.use_pos_enc = use_pos_enc
        self.padding_index = padding_index
        
        self.device = device
        self.dtype = dtype

        self.embd = nn.Embedding(self.vocab_size, self.decoder_dim, padding_idx=self.padding_index,
                                 dtype=self.dtype, device=self.device)
        
        self.layer_list = nn.ModuleList([])
        for _ in range(self.num_layers):
            layer = DecoderLayer(self.encoder_dim, self.decoder_dim, self.decoder_dim, self.kq_dimension,
                                 self.num_heads, self.linear_stretch, self.dropout,
                                 self.device, self.dtype)
            self.layer_list.append(layer)
        
        self.final = nn.Linear(self.decoder_dim, self.vocab_size,
                               device=self.device, dtype=self.dtype)

    def positional_encoding(self):
        pe = torch.zeros(self.max_seq_len, self.decoder_dim, device=self.device, dtype=self.dtype)
        position = torch.arange(0, self.max_seq_len, dtype=self.dtype, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.decoder_dim, 2, dtype=self.dtype, device=self.device) *
                             -(math.log(10000.0) / self.decoder_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, encoder_output):
        Z = self.embd(x) * torch.sqrt(torch.tensor(self.decoder_dim, dtype=self.dtype, device=self.device))
        if self.use_pos_enc:
            pe = self.positional_encoding()
            Z = Z + pe[:, :Z.size(1), :]
        
        for layer in self.layer_list:
            Z = layer(Z, encoder_output)
        
        Y = self.final(Z)
        Y = F.softmax(Y, dim=-1)
        return Y