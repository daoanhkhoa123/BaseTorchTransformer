import time 
from torch import nn, optim, Tensor
from torch.optim import Adam
from models.model.Transformer import Transformer
import torch

class Data:
    def __init__(self):
        self.source = ...
        self.target=...

class DataLoader:
    def __init__(self, data:Data):
        self.data = list(data)
    
    def generate(self):
        yield from self.data
            
def train(model:Transformer, data_loader:DataLoader, *args, **kwargs):
    model.train()
    epoch_loss = 0
    
    optimizer = Adam(model.parameters(), ...)
    critetion = nn.CrossEntropyLoss()
    clip = ...
    
    iterator = data_loader.generate()
    for i, batch in enumerate(iterator):
        src = batch.source
        trg:Tensor = batch.target
        
        optimizer.zero_grad()
        
        output:Tensor = model(src, trg[:,:-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        
        loss = critetion(output, trg)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()                
        
        epoch_loss += loss.item()
        print("step :", round((i/len(iterator)) * 100, 2),"% , loss :", loss.item())
        
    return epoch_loss/ len(iterator)
        
def evaluation(model:Transformer, data_loader:DataLoader, *args, **kwargs):
    model.eval()
    
    critertion = nn.CrossEntropyLoss()
    epoch_loss = 0
    
    iterator = data_loader.generate()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg:torch.Tensor = batch.trg
            
            output:torch.Tensor = model(src, trg[:,:-1])
            output = output.contiguous().view(-1, output.size(-1))
            trg = trg[:,1:].contiguous().view(-1)
            
            loss = critertion(output, trg)
            epoch_loss += loss.item()
        
        
if __name__ == "__main__":
    model = Transformer(...)
    data_loader = DataLoader()
    train(model, data_loader)