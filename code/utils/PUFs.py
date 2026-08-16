import numpy as np

class XORPUFModel():
    def __init__(self, weight, noise=0.0):
        self.weight = weight
        self.noise = noise
        self.PUF_length = weight.shape[-1] - 1
    
    def getResponse(self, phi):
        weight = self.weight + np.random.normal(0, self.noise, size=self.weight.shape)
        weight, bias = weight[:, :-1], weight[:, -1]
        Douts = np.sum(phi * weight, axis=-1, keepdims=False) + bias
        res = 0
        for i in range(Douts.shape[0]):
            res = res ^ int(Douts[i] >= 0)
        return res
    
    def randomSample(Xnum, length=32, alpha=0.05, noise=0.0):
        weight = np.random.normal(0, alpha, size=(Xnum, length + 1))
        return XORPUFModel(weight, alpha * noise)   

    def save(self, filename):
        np.savetxt(filename, self.weight, fmt='%f', delimiter=',')
        
    def load(self, filename):
        self.weight = np.loadtxt(filename, delimiter=',')
        self.PUF_length = self.weight.shape[-1] - 1