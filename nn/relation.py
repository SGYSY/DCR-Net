import torch
import torch.nn as nn
import torch.nn.functional as F


class CoInteractiveRelation(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        """
        Use MLP + Co-Attention to Encode
        """

        super().__init__()
        self._mlp = MLP(input_dim, hidden_dim)
        self._lstm = BiLSTM(input_dim, hidden_dim)
        self._mlp_layer = MLPLayer(input_dim, hidden_dim)
        self._co_attention = CoAttention()

    def forward(self, S, D):
        S = self._mlp(S)
        D = self._lstm(D)

        mlp_D = self._mlp_layer(D)
        mlp_S = self._mlp_layer(S)

        co_D, co_S = self._co_attention(S, D)

        return mlp_D, mlp_S, co_D, co_S


class CoAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, S, D):
        attention_D = F.softmax(torch.matmul(D, S.transpose(1, 2)), dim=-1)
        attention_S = F.softmax(torch.matmul(S, D.transpose(1, 2)), dim=-1)

        co_D = D + torch.matmul(attention_D, S)
        co_S = S + torch.matmul(attention_S, D)

        return co_D, co_S


class MLPLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.mlp_layer = MLP(input_dim, hidden_dim)

    def forward(self, prime_S, prime_D):
        cat_D = torch.cat((prime_S, prime_D), dim=-1)
        cat_S = torch.cat((prime_S, prime_D), dim=-1)

        mlp_D = self.mlp_layer(cat_D)
        mlp_S = self.mlp_layer(cat_S)

        return mlp_D, mlp_S


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
                 hidden_dim: int):
        super().__init__()
        self._rnn_cell = nn.LSTM(input_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

    def forward(self, D):
        output_D, _ = self._rnn_cell(D)
        return output_D
