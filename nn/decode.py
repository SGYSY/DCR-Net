import torch.nn as nn
import torch.nn.functional as F

from nn.relation import StackedRelation


class RelationDecoder(nn.Module):
    def __init__(self,
                 num_sc: int,
                 num_dar: int,
                 hidden_dim: int,
                 num_layer: int,
                 dropout_rate: float,
                 input_dim: int,
                 ):
        super().__init__()

        self._num_layer = num_layer

        self._relate_layer = StackedRelation(hidden_dim, hidden_dim, dropout_rate, num_layer)

        self.dar_linear = nn.Linear(hidden_dim, num_dar, bias=True)
        self.sc_linear = nn.Linear(hidden_dim, num_sc, bias=True)

    def add_missing_arg(self, layer=3):
        self._relate_layer.add_missing_arg(layer)

    def forward(self, S, D):
        SL, DL = self._relate_layer(S, D)

        # Apply linear transformation followed by softmax
        dar_linear = self.dar_linear(DL)
        sc_linear = self.sc_linear(SL)

        # Softmax for probability distributions
        dar_prob = F.softmax(dar_linear, dim=-1)
        sc_prob = F.softmax(sc_linear, dim=-1)

        return dar_prob, sc_prob


class LinearDecoder(nn.Module):
    def __init__(self,
                 num_sent: int,
                 num_act: int,
                 hidden_dim: int):
        super().__init__()

        self._sent_linear = nn.Linear(hidden_dim, num_sent)
        self._act_linear = nn.Linear(hidden_dim, num_act)

    def forward(self, input_h):
        return self._sent_linear(input_h), self._act_linear(input_h)
