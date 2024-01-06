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

        # 处理分词
        cls_sign = self._piece_vocab.CLS_SIGN  # 存储分词后的开头符号
        # 创建存储分词后的对话列表
        piece_list, sep_sign = [], self._piece_vocab.SEP_SIGN  # 存储分词后的分隔符号

        for dial_i in dial_list(0, len(dial_list)):
            piece_list.append([])

            for turn in dial_list[dial_i]:
                # 使用_piece_vocab.tokenize函数对句子进行分词，得到分词后的句子列表
                seg_list = self._piece_vocab.tokenize(turn)
                piece_list[-1].append([cls_sign] + seg_list + [sep_sign])

            if len(dial_list[dial_i]) < max_dial_len:
                pad_dial = [[cls_sign, sep_sign]] * (max_dial_len - len(dial_list[dial_i]))
                piece_list[-1].extend(pad_dial)

        # 处理分词后句子长度和填充的过程
        p_len_list = [[len(u) for u in d] for d in piece_list]
        max_p_len = max(expand_list(p_len_list))  # expand_list将嵌套的长度列表展开为一维列表

        # 存储填充后分词的句子列表和填充掩码
        pad_p_list, mask = [], []
        for dial_i in range(0, len(piece_list)):
            pad_p_list.append([])
            mask.append([])

            for turn in piece_list[dial_i]:
                pad_t = turn + [pad_sign] * (max_p_len - len(turn))
                pad_p_list[-1].append(self._piece_vocab.index(pad_t))  # 转换为分词后的词汇表索引
                mask[-1].append([1] * len(turn) + [0] * (max_p_len - len(turn)))  # 创建掩码列表

        # 转换为torch张量
        var_w_dial = torch.LongTensor(pad_w_list)
        var_p_dial = torch.LongTensor(pad_p_list)
        var_mask = torch.LongTensor(mask)

        # 移动到GPU上计算
        if torch.cuda.is_available():
            var_w_dial = var_w_dial.cuda()
            var_p_dial = var_p_dial.cuda()
            var_mask = var_mask.cuda()

        return var_w_dial, var_p_dial, var_mask, turn_len_list, p_len_list

    def predict(self, utt_list):
        var_utt, var_p, mask, len_list, _ = self._wrap_paddding(utt_list, False)
        if self._pretrained_model != "none":
            pred_act, pred_sent = self.forward(var_p, len_list, mask)
        else:
            pred_act, pred_sent = self.forward(var_utt, len_list, mask=None)

        trim_list = [len(l) for l in len_list]
        # [i, :trim_list[i], :] 第一个i是对话索引 第二个是选择前trim_list[i]个元素 第三个是表示选择所有维度
        flat_sent = torch.cat(
            [pred_sent[i, :trim_list[i], :]
             for i in range(0, len(trim_list))], dim=0
        )
        flat_act = torch.cat(
            [pred_act[i, :trim_list[i], :]
             for i in range(0, len(trim_list))], dim=0
        )

        # 提取最大预测
        _, top_sent = flat_sent.topk(1, dim=-1)
        _, top_act = flat_act.topk(1, dim=-1)

        # 索引转换成Python列表
        sent_list = top_sent.cpu().numpy().flatten().tolist()
        act_list = top_act.cpu().numpy().flatten().tolist()

        # 嵌套列表
        # 将扁平的列表重新嵌套，以匹配原始输入数据的结构
        nest_sent = nest_list(sent_list, trim_list)
        nest_act = nest_list(act_list, trim_list)

        # 词汇表转换，将索引转换为可读的字符串
        string_sent = iterable_support(self._sent_vocab.get, nest_sent)
        string_act = iterable_support(self._act_vocab.get, nest_act)

        return string_sent, string_act







