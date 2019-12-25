import numpy as np
# import tensorflow as tf
import torch
from torch import nn as nn

from scipy.stats import truncnorm


device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def swish(x):
    return x*torch.sigmoid(x)

def truncated_norm(size,std):
    val=truncnorm.rvs(a=-2,b=2,size=size,scale=std)
    return torch.tensor(val,dtype=torch.float)

def get_w_b(ensemble_size,input_size,output_size):
    w=truncated_norm(size=(ensemble_size,input_size,output_size),
                     std=1.0/(2.0*np.sqrt(input_size)))
    w=nn.Parameter(w,requires_grad=True)
    b=torch.zeros(ensemble_size,1,output_size,dtype=torch.float)
    b = nn.Parameter(b, requires_grad=True)
    return w,b

def shuffle_rows(arr):
    idx=np.argsort(np.random.uniform(size=arr.shape),axis=-1)
    return arr[np.arange(arr.shape[0])[:,None],idx]

class ensemble(nn.Module):
    def __init__(self,ensemble_size,input_size,output_size):
        super().__init__()
        self.ensemble_size=ensemble_size
        self.input_size=input_size
        self.output_size=output_size

        self.layer0_w,self.layer0_b=get_w_b(ensemble_size,input_size,200)
        self.layer1_w, self.layer1_b = get_w_b(ensemble_size, 200, 200)
        self.layer2_w, self.layer2_b = get_w_b(ensemble_size, 200, 200)
        self.layer3_w, self.layer3_b = get_w_b(ensemble_size, 200, output_size)

        self.inputs_mu=nn.Parameter(torch.zeros(input_size),requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(input_size), requires_grad=False)

        self.max_logvar=nn.Parameter(torch.ones(1,output_size//2,dtype=torch.float32)/2.0,requires_grad=True)
        self.min_logvar = nn.Parameter(-torch.ones(1, output_size // 2, dtype=torch.float32) * 10.0, requires_grad=True)

    def compute_decays(self):
        layer0_d=0.00025*(self.layer0_w**2).sum()/2.0
        layer1_d = 0.0005 * (self.layer1_w ** 2).sum() / 2.0
        layer2_d = 0.0005 * (self.layer2_w ** 2).sum() / 2.0
        layer3_d = 0.00075 * (self.layer3_w ** 2).sum() / 2.0
        decays=layer0_d+layer1_d+layer2_d+layer3_d
        return decays

    def get_input_stats(self,data):
        mu=np.mean(data,axis=0,keepdims=True)
        sigma=np.std(data,axis=0,keepdims=True)
        sigma[sigma<1e-12]=1.0
        self.inputs_mu.data=torch.from_numpy(mu).to(device).float()
        self.input_sigma.data=torch.from_numpy(sigma).to(device).float()

    def forward(self, input,ret_logvar=False):
        x=(input-self.inputs_mu)/self.inputs_sigma
        x=x.matmul(self.layer0_w)+self.layer0_b
        x=swish(x)
        x = x.matmul(self.layer1_w) + self.layer1_b
        x = swish(x)
        x = x.matmul(self.layer2_w) + self.layer2_b
        x = swish(x)
        x = x.matmul(self.layer3_w) + self.layer3_b
        mean=x[:,:,:self.output_size//2]
        logvar=x[:,:,self.output_size//2:]
        logvar=self.max_logvar+nn.functional.softplus(self.max_logvar-logvar)
        logvar=self.max_logvar+nn.functional.softplus(self.max_logvar-logvar)
        if ret_logvar:
            return mean,logvar
        return mean,torch.exp(logvar)