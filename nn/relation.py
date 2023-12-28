import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedRelation(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout_rate: float,
                 n_layer=3):
        super().__init__()

        self.layer = n_layer
        self.dropout = dropout_rate
        self.first_layer = CoInteractiveRelation(input_dim, hidden_dim, dropout_rate)
        self.second_layer = CoInteractiveRelation(hidden_dim, hidden_dim, dropout_rate)
        self.third_layer = CoInteractiveRelation(hidden_dim, hidden_dim, dropout_rate)

    def add_missing_arg(self, layer=3):
        self.layer = layer

    def forward(self, S, D):
        S, D = self.first_layer(S, D)

        if self.layer > 1:
            S, D = self.second_layer(S, D)
        if self.layer > 2:
            S, D = self.third_layer(S, D)

        return S, D


class CoInteractiveRelation(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout_rate: float,
                 n_layer=3):  # 还有一些其他参数
        """
        Use MLP + Co-Attention to Encode
        """

        super().__init__()

        self.dropout = dropout_rate
        self.layer = n_layer

        self._mlp = MLP(input_dim, hidden_dim, dropout=dropout_rate)
        self._lstm = BiLSTM(input_dim, hidden_dim, dropout=dropout_rate)
        self._mlp_layer = MLPLayer(2 * hidden_dim, hidden_dim, dropout=dropout_rate)
        self._co_attention = CoAttentionLayer(dropout=dropout_rate)  # 输入维度 输出维度 alpha concat这些都没有处理

    def forward(self, S, D):
        pre_s = self._mlp(S)
        pre_d = self._lstm(D)

        mlp_s, mlp_d = self._mlp_layer(pre_s, pre_d)

        co_s, co_d = self._co_attention(mlp_s, mlp_d)  # 最后输出需要用elu激活函数吗

        return co_s, co_d


class CoAttentionLayer(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Define dimensions for the linear layers based on your architecture
        # Example: self.attention = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, S, D):  # 可能不需要dropout
        drop_s = self.dropout(S)
        drop_d = self.dropout(D)

        attention_d = F.softmax(torch.matmul(drop_d, drop_s.transpose(1, 2)), dim=-1)
        attention_s = F.softmax(torch.matmul(drop_s, drop_d.transpose(1, 2)), dim=-1)

        co_d = drop_d + torch.matmul(attention_d, drop_s)
        co_s = drop_s + torch.matmul(attention_s, drop_d)  # 最后输出需要用elu激活函数吗

        return co_s, co_d


class MLPLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout):
        super().__init__()
        self.mlp_layer = MLP(input_dim, hidden_dim, dropout=dropout)

    def forward(self, prime_s, prime_d):
        cat_d = torch.cat((prime_s, prime_d), dim=-1)
        cat_s = torch.cat((prime_s, prime_d), dim=-1)

        mlp_d = self.mlp_layer(cat_d)
        mlp_s = self.mlp_layer(cat_s)

        return mlp_s, mlp_d


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout):
        super().__init__()
        self._mlp1_cell = nn.Linear(input_dim, hidden_dim)
        self._mlp2_cell = nn.Linear(hidden_dim, input_dim)
        self._dropout = nn.Dropout(dropout)

    def forward(self, S):
        relu_s = F.relu(self._mlp1_cell(S))
        drop_s = self._dropout(relu_s)
        output_s = self._mlp2_cell(drop_s)
        return output_s


class BiLSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 dropout):
        super().__init__()
        self._rnn_cell = nn.LSTM(input_dim, hidden_dim // 2, bidirectional=True,
                                 batch_first=True, dropout=dropout)

    def forward(self, D):
        output_d, _ = self._rnn_cell(D)
        return output_d
