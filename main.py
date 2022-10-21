from common.arguments import *
from common.models import *
from common.Datasets import *
from common.Trainer import *
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import random, os
import numpy as np

args = prase_args()
print(args)
# 确定有关GPU使用的相关问题
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# 固定随机种子
seed = 1024
random.seed(seed)     # python的随机性
np.random.seed(seed)  # np的随机性
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)            # torch的CPU随机性，为CPU设置随机种子
torch.cuda.manual_seed(seed)       # torch的GPU随机性，为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # torch的GPU随机性，为所有GPU设置随机种子
# 固定随机种子


def run():
    data_dist = torch.load(os.path.join(args.data, args.data_file_name))
    X_train, X_test, y_train, y_test = train_test_split(data_dist['data'], data_dist['target'], test_size=0.1, random_state=seed)
    train_dataset = SEMGDatasets(X_train, y_train)
    test_dataset = SEMGDatasets(X_test, y_test)
    model = raw_CNN(50,160)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size)
    loss_function = nn.CrossEntropyLoss()
    Train(args, model, args.epoch_num, train_iter, test_iter, loss_function, optimizer)
if __name__ =='__main__':
    run()