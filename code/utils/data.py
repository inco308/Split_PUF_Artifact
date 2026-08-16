import torch
import numpy as np
from random import randint

class DataLoader():
    def __init__(self, init_data: np.ndarray, batch_size=4096, data_type="PR",
                 shuffle=True, drop_last=False, device='cuda'):
        self.init_data = init_data.copy()
        self.dataline = torch.from_numpy(init_data).to(torch.float32)
        self.size = self.dataline.shape[0]
        
        self.data_type = data_type
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.device = device
        
    def __iter__(self):
        self.pos = -1
        if self.shuffle is True:
            np.random.shuffle(self.init_data)
            self.dataline = torch.from_numpy(self.init_data).to(torch.float32)
        self.index = [(x, x + self.batch_size) for x in range(0, self.size, self.batch_size)]
        if self.index[-1][1] > self.size:
            if self.drop_last is True:
                self.index = self.index[:-1]
                self.size = self.index[-1][1]
            else:
                self.index[-1] = (self.index[-1][0], self.size)
        return self
    
    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index):
            raise StopIteration
        (l, r) = self.index[self.pos]
        return self.split(self.dataline[l:r])
    
    def split(self, dataline):
        if self.data_type == "PR": # phi, response
            return dataline[:, :-1].cuda(), dataline[:, -1:].cuda()
    
def makeData(PUF_sample, data_size, filename=None):
    dataset = []
    length = PUF_sample.PUF_length
    for _ in range(data_size):
        phi = [randint(0, 1) * 2 - 1 for _ in range(length)]
        res = PUF_sample.getResponse(phi)
        
        dataline = phi
        dataline.append(res)
        dataset.append(dataline)
        
    dataset = np.asarray(dataset)
    if filename is not None:
        np.savetxt(filename, dataset, fmt='%d', delimiter=',')
    return dataset

def splitDataLoader(dataset, loader=DataLoader):
    data_size = dataset.shape[0]
    train_size = int(data_size * 0.8)
    valid_size = int(data_size * 0.02)
    
    np.random.shuffle(dataset)
    train_loader = loader(dataset[:train_size])
    valid_loader = loader(dataset[train_size:train_size+valid_size])
    test_loader = loader(dataset[train_size+valid_size:])
    return train_loader, valid_loader, test_loader

def loadData(filename):
    dataset = np.loadtxt(filename, delimiter=',')
    return dataset