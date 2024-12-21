from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm

def train_Unet(net: nn.Module, loader: DataLoader, optim: Optimizer, loss_func: nn.Module, device):
    torch.cuda.empty_cache() # in order to have enough VRAM to run the model
    loss_values = []

    net.to(device)
    loss_func.to(device)
    for data in tqdm(loader):
        img, mask = data

        img = img.to(device)
        mask = mask.to(device)
        
        optim.zero_grad()
        output, output_mask = net.forward((img, mask))
        loss = loss_func(img * mask, mask, output, img)
        loss.backward()
        optim.step()
        
        loss_values.append(loss.item())
    
    return loss_values


def train_AE(ae_net: nn.Module, loader: DataLoader, optim: Optimizer, loss_func: nn.Module, device):
    
    loss_values = []
    ae_net.to(device)
    loss_func.to(device)
    for data in tqdm(loader):
        input, mask = data
        input = input.to(device)
        mask = mask.to(device)
        optim.zero_grad() 
        output = ae_net.forward(input * mask)
        loss = loss_func(output, input)
        loss.backward() 
        optim.step()
        
        loss_values.append(loss.item())

    
    return loss_values

loss_functions = nn.MSELoss()
def getOptim(net: nn.Module, lr: float):
    return Adam(net.parameters(), lr)


def validate_AE(net: nn.Module, loader: DataLoader, loss_func: nn.Module, device):
    loss_values = []

    net.to(device)
    loss_func.to(device)

    for data in tqdm(loader):
        input, mask = data
        input = input.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            output = net.forward(input  * mask)
            loss = loss_func(output, input)
            loss_values.append(loss.item())
    
    return loss_values
