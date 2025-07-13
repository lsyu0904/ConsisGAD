import torch
import dgl
import math
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import Linear as TypedLinear


class CustomLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        nn.init.zeros_(self.bias)
