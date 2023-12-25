import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super().__init__()
        self._mlp1_cell = nn.Linear(input_dim, hidden_dim)
        self._mlp2_cell = nn.Linear(hidden_dim, input_dim)

    def forward(self, S):
        relu_S = F.relu(self._mlp1_cell(S))
        output_S = self._mlp2_cell(relu_S)
        return output_S


class BiLSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout_rate: float):
        super().__init__()
        self._rnn_cell = nn.LSTM(input_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

    def forward(self, D):
        output_D, _= self._rnn_cell(D)
        return output_D



