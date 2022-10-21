#  在这里我们需要实现的是argprase的相关功能,prase是解析的意思，这里用于解析命令行
import argparse
# 官方文档: https://docs.python.org/dev/library/argparse.html#help
def prase_args():
    parser = argparse.ArgumentParser(description='Args for Training') # 创建解析器

    # General
    parser.add_argument('-path', '--data', default='data', type=str, metavar='datapath', help='the location of the path')
    parser.add_argument('-dfn', '--data_file_name', default='50_raw.mat', type=str, metavar='mat file name',
                        help='the name of the data file')
    parser.add_argument('-ckp', '--checkpoint_path', default='checkpoint', type=str, metavar='checkpoint path',
                        help='the path of checkpoint dir')
    # General

    # Train
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='0.001',
                        help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.001, type=float, metavar='0.001',
                        help='weight decay')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='batch size',
                        help='batch size')
    parser.add_argument('-e', '--epoch_num', default=200, type=int, metavar='epoch num',
                        help='epoch num')
    parser.add_argument('-skpe', '--sckp_epoch', default=20, type=int, metavar='save checkpoint epoch',
                        help='epoch num')
    # Train

    args = parser.parse_args()
    return args