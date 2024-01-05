import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.encode import UtteranceEncoder
from nn.decode import RelationDecoder, LinearDecoder

from utils.dict import PieceAlphabet
from utils.load import WordAlphabet, LabelAlphabet
from utils.help import expand_list, noise_augment
from utils.help import nest_list, iterable_support


class TaggingAgent(nn.Module):

    def __init__(self,
                 word_vocab: WordAlphabet,
                 piece_vocab: PieceAlphabet,
                 sent_vocab: LabelAlphabet,
                 act_vocab: LabelAlphabet,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layer: int,
                 dcr_layer: int,
                 dcr_dropout_rate: float,
                 dropout_rate: float,
                 use_linear_decoder: bool,
                 pretrained_model: str,
                 input_dim: int):

        super().__init__()

        self._piece_vocab = piece_vocab
        self._pretrained_model = pretrained_model

        self._word_vocab = word_vocab
        self._sent_vocab = sent_vocab
        self._act_vocab = act_vocab

        self._encoder = UtteranceEncoder(
            nn.Embedding(len(word_vocab), embedding_dim),
            hidden_dim, dropout_rate, pretrained_model
        )

        if use_linear_decoder:
            self._decoder = LinearDecoder(len(sent_vocab), len(act_vocab), hidden_dim)
        else:
            self._decoder = RelationDecoder(
                len(word_vocab), len(act_vocab)
                , hidden_dim, num_layer, dropout_rate, input_dim
            )

        # Loss function
        self._criterion = nn.NLLLoss(reduction="sum")

    def set_load_best_missing_arg(self, pretrained_model):
        self._pretrained_model = pretrained_model
        self._encoder.add_missing_arg(pretrained_model)

    def set_load_best_missing_arg_mastodon(self, pretrained_model, layer=3):
        self._pretrained_model = pretrained_model
        self._encoder.add_missing_arg(pretrained_model)
        self._decoder.add_missing_arg(layer)

    def forward(self, input_h, len_list, mask=None):
        encode_h = self._encoder(input_h, mask)
        return self._decoder(encode_h, len_list)

    @property
    def sent_vocab(self):
        return self._sent_vocab

    @property
    def act_vocab(self):
        return self._act_vocab

    def _wrap_paddding(self, dial_list, use_noise):
        # 处理对话列表中最长的对话长度
        dial_len_list = [len(d) for d in dial_list]
        max_dial_len = max(dial_len_list)

        # 处理对话中最长的句子长度
        turn_len_list = [[len(u) for u in d] for d in dial_list]
        max_turn_len = max(expand_list(turn_len_list))

        # 储存填充后的对话列表和填充符号
        pad_w_list, pad_sign = [], self._word_vocab.PAD_SIGN
        for dial_i in range(0, len(dial_list)):
            pad_w_list.append([])

            for turn in dial_list[dial_i]:
                if use_noise:
                    noise_turn = noise_augment(self._word_vocab, turn, 5.0)
                else:
                    noise_turn = turn
                pad_utt = noise_turn + [pad_sign] * (max_turn_len - len(turn))
                # iterable_support生成可迭代对话列表对象，用于迭代填充后的句子
                pad_w_list[-1].append(iterable_support(self._word_vocab.index, pad_utt))

                if len(dial_list[dial_i]) < max_dial_len:
                    pad_dial = [[pad_sign] * max_turn_len] * (max_dial_len - len(dial_list[dial_i]))
                    # extend可以把每个句子添加到句子列表中 而不是将pad_dial作为一个元素添加
                    pad_w_list[-1].extend(iterable_support(self._word_vocab.index, pad_dial))






















