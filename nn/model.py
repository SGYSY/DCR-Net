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



