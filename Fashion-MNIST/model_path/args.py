import argparse
from argparse import ArgumentParser
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Loc Training")
    # General
    parser.add_argument("--lr", type=float, default=0.1) #学习率
    parser.add_argument("--n_hidden", type=int, default=10) #隐藏层
    parser.add_argument("--sample_idx", type=int, default=1)  #图片索引，可视化，可设置为None不进行可视化输出 

    parser.add_argument("--lambda", type=float, default=0.01) # L正则化强度
    parser.add_argument("--activation_function", type=str, default="sigmoid") #激活函数
    parser.add_argument("--checkpoint_path", type=str, default='best_model.npz') # 测试时读取的checkpoint路径  'best_model.npz'
    
    args = parser.parse_args()
    
    if bool(args.checkpoint_path):
        assert osp.isfile(args.checkpoint_path)

    assert args.activation_function in ("sigmoid", "relu", "tanh")
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
