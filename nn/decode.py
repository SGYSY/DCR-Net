import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationDecoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_dar: int,
                 num_sc: int):
        super().__init__()
        self.dar_linear = nn.Linear(hidden_dim, num_dar)
        self.sc_linear = nn.Linear(hidden_dim, num_sc)

    def forward(self, DL, SL):
        # Apply linear transformation followed by softmax
        dar_linear = self.dar_linear(DL)
        sc_linear = self.sc_linear(SL)

        # Softmax for probability distributions
        dar_prob = F.softmax(dar_linear)
        sc_prob = F.softmax(sc_linear)

        return dar_prob, sc_prob
