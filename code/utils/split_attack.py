import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

from utils.attack import MLPOnXORAPUF
    
class BMM_Layer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, in_features, out_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(2, 1, out_features))

    def forward(self, x):
        return torch.bmm(x, self.weight) + self.bias
            
class splitXORAPUF(MLPOnXORAPUF):
    class NNModel_baseline(nn.Module):
        def __init__(self, XOR_num, PUF_length):
            super().__init__()
            
            # => (MLP) => R
            input_dim = PUF_length
            model_k = XOR_num

            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 2 ** (model_k - 1), bias=True), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 2 ** model_k), nn.Tanh(),
                nn.Linear(2 ** model_k, 2 ** (model_k - 1)), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 1), nn.Sigmoid()
            )
            
            print("Launch NNModel_baseline, level up 0")

        def forward(self, x):
            res = self.mlp(x)
            return res
    
    class NNModel(nn.Module):
        def __init__(self, XOR_num, PUF_length):
            super().__init__()
            
            # => (MLP) => R
            input_dim = PUF_length
            div_num = XOR_num // 2
            model_k = div_num

            # MLP
            self.mlp_up = nn.Sequential(
                nn.Linear(input_dim, 2 ** (model_k - 1), bias=True), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 2 ** model_k), nn.Tanh(),
                nn.Linear(2 ** model_k, 2 ** (model_k - 1)), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 1), nn.Sigmoid()
            )
            self.mlp_down = nn.Sequential(
                nn.Linear(input_dim, 2 ** (model_k - 1), bias=True), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 2 ** model_k), nn.Tanh(),
                nn.Linear(2 ** model_k, 2 ** (model_k - 1)), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 1), nn.Sigmoid()
            )
            
            print("Launch NNModel, level up 0")

        def forward(self, x):
            p_up = self.mlp_up(x)
            p_down = self.mlp_down(x)
            res = p_up * (1.0 - p_down) + p_down * (1.0 - p_up)
            return res

    class NNModel_BMM(nn.Module):
        def __init__(self, XOR_num, PUF_length):
            super().__init__()
            input_dim = PUF_length
            model_k = XOR_num // 2
            
            h1_dim = 2 ** (model_k - 1)
            h2_dim = 2 ** model_k
            
            self.fc1 = nn.Linear(input_dim, h1_dim * 2)
            self.group_fc2 = BMM_Layer(h1_dim, h2_dim)
            self.group_fc3 = BMM_Layer(h2_dim, h1_dim)
            self.group_out = BMM_Layer(h1_dim, 1)

            print("Launch NNModel_BMM, level up 0")

        def forward(self, x):
            x = torch.tanh(self.fc1(x))

            batch_size = x.shape[0]
            x = x.view(batch_size, 2, -1).transpose(0, 1)
            
            x = torch.tanh(self.group_fc2(x))
            x = torch.tanh(self.group_fc3(x))
            x = torch.sigmoid(self.group_out(x))
            
            res = x[0] * (1.0 - x[1]) + x[1] * (1.0 - x[0])
            return res
    
    class NNModel_Masked(nn.Module):
        def __init__(self, XOR_num, PUF_length):
            super().__init__()
            input_dim = PUF_length
            div_num = XOR_num // 2
            model_k = div_num + 1
            
            dims = [input_dim, 2 ** (model_k - 1), 2 ** model_k, 2 ** (model_k - 1), 2]
            
            layers = []
            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i+1])
                # 对中间隐藏层应用掩码逻辑
                if i > 0 and i < len(dims) - 2: 
                    mask = torch.zeros(dims[i+1], dims[i])
                    mid_in, mid_out = dims[i] // 2, dims[i+1] // 2
                    mask[:mid_out, :mid_in] = 1
                    mask[mid_out:, mid_in:] = 1
                    
                    with torch.no_grad():
                        layer.weight *= mask
                    mask = mask.to('cuda')
                    layer.weight.register_hook(lambda grad, m=mask: grad * m)
                
                layers.append(layer)
                layers.append(nn.Tanh() if i < len(dims) - 2 else nn.Sigmoid())
                
            self.mlp = nn.Sequential(*layers)

            print("Launch NNModel_Masked, level up 0")

        def forward(self, x):
            y = self.mlp(x)
            p_up, p_down = y[:, :1], y[:, 1:]
            res = p_up * (1.0 - p_down) + p_down * (1.0 - p_up)
            return res
    
    class NNModel_fc(nn.Module):
        def __init__(self, XOR_num, PUF_length):
            super().__init__()
            
            # => (MLP) => R
            input_dim = PUF_length
            div_num = XOR_num // 2
            model_k = div_num + 1

            # MLP
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 2 ** (model_k - 1), bias=True), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 2 ** model_k), nn.Tanh(),
                nn.Linear(2 ** model_k, 2 ** (model_k - 1)), nn.Tanh(),
                nn.Linear(2 ** (model_k - 1), 2), nn.Sigmoid()
            )

            print("Launch NNModel_fc, level up 0")
            
        def forward(self, x):
            y = self.mlp(x)
            p_up, p_down = y[:, :1], y[:, 1:]
            res = p_up * (1.0 - p_down) + p_down * (1.0 - p_up)
            return res
        
    def launchingLog():
        print("Launching split MLP attack on XOR APUF...")

    def train(self, epochs=500, device='cuda', log=None, model_type="NNModel"):
        if model_type == "NNModel":
            model = self.NNModel(self.XOR_num, self.PUF_length).to(device)
        elif model_type == "NNModel_BMM":
            model = self.NNModel_BMM(self.XOR_num, self.PUF_length).to(device)
        elif model_type == "NNModel_Masked":
            model = self.NNModel_Masked(self.XOR_num, self.PUF_length).to(device)
        elif model_type == "NNModel_fc":
            model = self.NNModel_fc(self.XOR_num, self.PUF_length).to(device)
        else:
            model = self.NNModel_baseline(self.XOR_num, self.PUF_length).to(device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = F.binary_cross_entropy
        
        stTime = time()
        batch_epTime = stTime
        for epoch in range(1, epochs + 1):
            model.train()
            for (phi, res) in self.train_loader:
                predict = model(phi)
                
                loss_main = criterion(predict, res)
                #loss_aux = -torch.mean((p_up - 0.5) ** 2 + (p_down - 0.5) ** 2)
                loss = loss_main #+ 0.001 * loss_aux

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            model.eval()
            acc_count = 0
            for (phi, res) in self.valid_loader:
                predict = model(phi)
                acc_count += (predict.round() == res).sum().item()
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
            predict = self.model(phi)
            acc_count += (predict.round() == res).sum().item()
        accuracy = acc_count / self.test_loader.size
        print("Test Accuracy = %.2f%%" % (accuracy * 100))
        return accuracy