import torch
import torch.nn as nn
from torch.Functional as F




class MultiHead(nn.Module):
    def __init__(self,d_model,n_head,dropout):
        assert d_model%n_head == 0

        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout

        self.proj_k = nn.linear(d_model,d_model)
        
