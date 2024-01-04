import torch.nn as nn
import torch.nn.functional as F

from nn.relation import StackedRelation


class RelationDecoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_dar: int,
                 num_sc: int,
                 dropout_rate: float,
                 input_dim: int,
                 num_layer: int):
        super().__init__()

        self._num_layer = num_layer

        self._sent_layer_dict = nn.ModuleDict()
        self._act_layer_dict = nn.ModuleDict()

        self._relate_layer = StackedRelation(input_dim, hidden_dim, dropout_rate, num_layer)

        self.dar_linear = nn.Linear(hidden_dim, num_dar)
        self.sc_linear = nn.Linear(hidden_dim, num_sc)

    def add_missing_arg(self, layer=3):
        self._relate_layer.add_missing_arg(layer)

    def forward(self, DL, SL):
        # Apply linear transformation followed by softmax
        dar_linear = self.dar_linear(DL)
        sc_linear = self.sc_linear(SL)

        # Softmax for probability distributions
        dar_prob = F.softmax(dar_linear, dim=-1)
        sc_prob = F.softmax(sc_linear, dim=-1)

        return dar_prob, sc_prob
