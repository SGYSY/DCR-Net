import os
from copy import deepcopy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.dict import WordAlphabet
from utils.dict import LabelAlphabet
from utils.help import load_json_file
from utils.help import iterable_support


class DataHub(object):

    def __init__(self):
        self._word_vocab = WordAlphabet("word")

        self._sent_vocab = LabelAlphabet("sentiment")
        self._act_vocab = LabelAlphabet("act")

        # using a dict to store the train, dev, test data
        self._data_collection = {}

    @property
    def word_vocab(self):
        return deepcopy(self._word_vocab)

    @property
    def sent_vocab(self):
        return deepcopy(self._sent_vocab)

    @property
    def act_vocab(self):
        return deepcopy(self._act_vocab)

    @classmethod
    def from_dir_addadj(cls, dir_path):
        house = DataHub()

        # 读取指定目录下的训练、验证和测试数据文件
        house._data_collection["train"] = house._read_data(
            os.path.join(dir_path, "train.json"), True
        )
        house._data_collection["dev"] = house._read_data(
            os.path.join(dir_path, "dev.json"), False
        )
        house._data_collection["test"] = house._read_data(
            os.path.join(dir_path, "test.json"), False
        )
        return house

    # 读取特定格式的数据文件并可能构建词汇表
    def _read_data(self,
                   file_path: str,
                   build_vocab: bool = False):
        """
        On train, set build_vocab=True, will build alphabet
        """

        utt_list, sent_list, act_list = [], [], []
        dialogue_list = load_json_file(file_path)

        for session in dialogue_list:
            utt, emotion, act = [], [], []

            for interact in session:
                act.append(interact["act"])
                emotion.append(interact["sentiment"])

                word_list = interact["utterance"].split()
                utt.append(word_list)

            utt_list.append(utt)
            sent_list.append(emotion)
            act_list.append(act)

        if build_vocab:
            iterable_support(self._word_vocab.add, utt_list)
            iterable_support(self._sent_vocab.add, sent_list)
            iterable_support(self._act_vocab.add, act_list)

        # The returned list is based on dialogue, with three levels of nesting.
        return utt_list, sent_list, act_list

    # 获取数据集
    def get_iterator(self, data_name, batch_size, shuffle):
        data_set = _GeneralDataSet(*self._data_collection[data_name])

        data_loader = DataLoader(
            data_set, batch_size, shuffle, collate_fn=_collate_func
        )
        return data_loader


# 加载和处理数据
class _GeneralDataSet(Dataset):

    def __init__(self, utt, sent, act):
        self._utt = utt
        self._sent = sent
        self._act = act

    # 通过索引获取数据集中的单个项
    def __getitem__(self, item):
        return self._utt[item], self._sent[item], self._act[item]

    # 返回情感标签列表的长度
    def __len__(self):
        return len(self._sent)


# 一个自定义的批处理函数，它指定了如何从数据集中抽取多个元素，并将它们组合成一个批次
def _collate_func(instance_list):
    """
    As a function parameter to instantiate the DataLoader object.
    """

    n_entity = len(instance_list[0])
    scatter_b = [[] for _ in range(0, n_entity)]

    for idx in range(0, len(instance_list)):
        for jdx in range(0, n_entity):
            scatter_b[jdx].append(instance_list[idx][jdx])
    return scatter_b
