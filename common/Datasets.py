import torch
from torch.utils.data import Dataset
class SEMGDatasets(Dataset):
    def __init__(self, x,y):
        '''
        :param packed_data_path:the path of the packed data,it should be a npy file.
        '''
        super().__init__()
        self.data = torch.Tensor(x)
        self.target = torch.LongTensor(y)

    def __len__(self):
        assert len(self.data) == len(self.target)
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.target[item]

