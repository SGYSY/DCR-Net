from transformers import BertModel, RobertaModel, XLNetModel, AlbertModel, ElectraModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiRNNEncoder(nn.Module):
    def __init__(self,
                 word_embedding: nn.Embedding,
                 hidden_dim: int,
                 dropout_rate: float):
        super().__init__()

        _, embedding_dim = word_embedding.weight.size()
        # 声明一个类成员变量，对象内部的状态不能从外部直接更改
        self._word_embedding = word_embedding

        self._rnn_cell = nn.LSTM(embedding_dim, hidden_dim // 2,
                                 batch_first=True, bidirectional=True)

        self._drop_layer = nn.Dropout(dropout_rate)

        self._hidden_dim = hidden_dim

    def forward(self, input_w):
        embed_w = self._word_embedding(input_w)
        dropout_w = self._drop_layer(embed_w)

        hidden_states, _ = self._rnn_cell(dropout_w)
        last_hidden_states = hidden_states[:, -1, :]
        #
        cat_hidden_states = torch.cat((last_hidden_states[:, :self._hidden_dim // 2],
                                       last_hidden_states[:, self._hidden_dim // 2:]), dim=-1)
