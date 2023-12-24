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

    def forward(self, input_w):
        embed_w = self._word_embedding(input_w)
        dropout_w = self._drop_layer(embed_w)

        # output shape: (batch_size, seq_len, num_directions * hidden_size)
        # h_n shape: (num_layers * num_directions, batch_size, hidden_size)
        output, (h_n, _) = self._rnn_cell(dropout_w)

        batch_size = input_w.size(0)

        # 将h_n重新排列成一个4D张量，这里的1表示只有一层（考虑到只使用了一个LSTM层），2表示双向（前向和后向）,
        # -1表示自动计算hidden_size的大小。
        h_n = h_n.view(1, 2, batch_size, -1).transpose(0, 2)
        utterance_representation = torch.cat((h_n[:, -1, 0, :], h_n[:, -1, 1, :]), dim=1)
        return utterance_representation
