from transformers import BertModel

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtteranceEncoder(nn.Module):
    def __init__(self,
                 word_embedding: nn.Embedding,
                 hidden_dim: int,
                 dropout_rate: float,
                 pretrained_model: str):
        """
        Use BiLSTM + Self_Attention to Encode
        """
        super().__init__()

        if pretrained_model != "none":
            self._utt_encoder = UtterancePretrainedModel(hidden_dim, pretrained_model)
        else:
            self._utt_encoder = BiRNNEncoder(word_embedding, hidden_dim, dropout_rate)
        self._pretrained_model = pretrained_model

        self.self_attention = SelfAttention(hidden_dim, dropout_rate)

    # Add for loading best model
    def add_missing_arg(self, pretrained_model):
        self._pretrained_model = pretrained_model

    def forward(self, dialogues, mask=None):
        """
        dialogues: Tensor representing a batch of dialogues, shape [batch_size, T, K_t]
        seq_lens: List representing the actual lengths of each dialogue in the batch
        """
        batch_size, T, K_t = dialogues.size()
        H = torch.zeros(batch_size, T, self._utt_encoder.rnn_cell.hidden_size * 2).to(dialogues.device)

        # 用BiRNNEncoder来编码每一个utterance
        for i in range(T):
            utt = dialogues[:, i, :]
            if self._pretrained_model != "none":
                utt_representation = self._utt_encoder(utt, mask)
            else:
                utt_representation = self._utt_encoder(utt)
            H[:, i, :] = utt_representation

        C = self.self_attention(H)
        return C


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
        # 这里做了对双向和单向LSTM的不同处理，但是经过测试这里的的确是BiLSTM
        num_layers = self._rnn_cell.num_layers
        num_directions = 2 if self._rnn_cell.bidirectional else 1
        hidden_size = self._rnn_cell.hidden_size

        # 重新排列h_n以使其符合(batch_size, num_layers, num_directions, hidden_size)
        h_n = h_n.view(num_layers, num_directions, batch_size, hidden_size)
        h_n = h_n.transpose(0, 2).transpose(1, 2)

        # 对于双向，连接最后一层的两个方向的隐藏状态
        # 对于单向，直接使用最后一层的隐藏状态
        if self._rnn_cell.bidirectional:
            utterance_representation = torch.cat((h_n[:, -1, 0, :], h_n[:, -1, 1, :]), dim=-1)
        else:
            utterance_representation = h_n[:, -1, 0, :]
        return utterance_representation

    @property
    def rnn_cell(self):
        return self._rnn_cell


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float):
        super().__init__()

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.scale = hidden_dim ** 0.5

    def forward(self, H):
        Q = self.Wq(H)
        K = self.Wk(H)
        V = self.Wv(H)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_scores = F.softmax(attention_scores, dim=-1)
        drop_attention_scores = self.dropout(attention_scores)
        C = torch.matmul(drop_attention_scores, V)

        return C


class UtterancePretrainedModel(nn.Module):
    HIDDEN_DIM = 768

    def __init__(self,
                 hidden_dim: int,
                 pretrained_model):
        super().__init__()
        self._pretrained_model = pretrained_model

        if pretrained_model == "bert":
            self._encoder = BertModel.from_pretrained("bert-base-uncased")
        else:
            assert False, "Unsupported pretrained_model argument, expected 'bert' but got: {}".format(pretrained_model)

        self._linear = nn.Linear(UtterancePretrainedModel.HIDDEN_DIM, hidden_dim)

    def forward(self, input_p, mask):
        outputs = self._encoder(input_p, attention_mask=mask)

        # 获取[CLS]标记的输出，这通常是Transformer模型的第一个输出，用于下游任务
        cls_output = outputs.last_hidden_state[:, 0, :]

        transformed_output = self._linear(cls_output)

        return transformed_output
