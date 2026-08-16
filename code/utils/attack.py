import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

class MLPOnXORAPUF():
    class NNModel(nn.Module):
        def __init__(self, XOR_num, PUF_length):
            super().__init__()
            
            # => (MLP) => R
            input_dim = PUF_length

            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 2 ** (XOR_num - 1), bias=True), nn.Tanh(),
                nn.Linear(2 ** (XOR_num - 1), 2 ** XOR_num), nn.Tanh(),
                nn.Linear(2 ** XOR_num, 2 ** (XOR_num - 1)), nn.Tanh(),
                nn.Linear(2 ** (XOR_num - 1), 1), nn.Sigmoid()
            )
        
        def forward(self, x):
            res = self.mlp(x)
            return res
    
    def launchingLog(self):
        print("Launching MLP attack on XOR APUF...")
        
    def __init__(self, train_loader, valid_loader, test_loader,
                 XOR_num, PUF_length, log=False):
        if log is True:
            self.launchingLog()
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        self.XOR_num = XOR_num
        self.PUF_length = PUF_length
        
        self.model = None
    
    def train(self, epochs=500, device='cuda', log=None):
        model = self.NNModel(self.XOR_num, self.PUF_length).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = F.binary_cross_entropy
        
        stTime = time()
        batch_epTime = stTime
        for epoch in range(1, epochs + 1):
            model.train()
            for (phi, res) in self.train_loader:
                predict = model(phi)
                loss = criterion(predict, res)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval()
            acc_count = 0
            for (phi, res) in self.valid_loader:
                predict = model(phi).round()
                acc_count += (predict == res).sum().item()
            accuracy = acc_count / self.valid_loader.size
            if log is not None and epoch % log == 0:
                duration = time() - batch_epTime
                batch_epTime = batch_epTime + duration
                print("Epoch %d, Accuracy = %.2f%%, time = %.2f s" % (epoch, accuracy * 100, duration))
        duration = time() - stTime
        print("Train time cost = %.2f s" % (duration))
        
        self.model = model
        return duration
        
    def test(self):
        self.model.eval()
        acc_count = 0
        for (phi, res) in self.test_loader:
            predict = self.model(phi).round()
            acc_count += (predict == res).sum().item()
        accuracy = acc_count / self.test_loader.size
        print("Test Accuracy = %.2f%%" % (accuracy * 100))
        return accuracy
    