import os
import torch
import argparse

from utils import DataHub
from utils import evaluate

# 处理命令行参数
parser = argparse.ArgumentParser()
# 指定预训练模型的类型
parser.add_argument("--pretrained_model", "-pm", type=str, default="none",
                    choices=["none", "bert"],
                    help="choose pretrained model, default is none.")
# 指定数据集的目录
parser.add_argument("--data_dir", "-dd", type=str, default="dataset/mastodon")
# 指定模型和其他任何输出应该被保存到的目录
parser.add_argument("--save_dir", "-sd", type=str, default="./save/mastodon")
# 指定处理数据时使用的批处理大小
parser.add_argument("--batch_size", "-bs", type=int, default=16)
# 解析参数
args = parser.parse_args()

# 加载和预处理数据
data_house = DataHub.from_dir_addadj(args.data_dir)
# 加载了之前保存的模型
model = torch.load(os.path.join(args.save_dir, "model.pt"))
if args.data_dir == "dataset/mastodon":
    metric = False
    model.set_load_best_missing_arg_mastodon(args.pretrained_model)
else:
    metric = True
    model.set_load_best_missing_arg(args.pretrained_model)
if torch.cuda.is_available():
    model = model.cuda()

test_sent_f1, sent_r, sent_p, test_act_f1, act_r, act_p, test_time = evaluate(
    model, data_house.get_iterator("test", args.batch_size, False), metric)
print("On test, sentiment f1: {:.4f} (r: {:.4f}, p: {:.4f}), act f1: {:.4f} (r: {:.4f}, p: {:.4f})"
      .format(test_sent_f1, sent_r, sent_p, test_act_f1, act_r, act_p))
