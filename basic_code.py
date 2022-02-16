# Created on Fri Oct  8 10:17:34 2021
# %% 
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
# %% hyper-parameters
BATCH_SIZE = 10
LR = 1e-5
EPOCH = 100
# %% DEVICE setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# %% functions
def calc_s_val(x,model,DEVICE):
    model.eval()
    s_val= np.zeros((1,x.shape[0]))
    with torch.no_grad():
        x_test = x.to(DEVICE)
        y_test = model(x_test)
        s_val = y_test.cpu().numpy()
    return s_val

def data_load():
    np.random.seed(1004)
    
    x_train = torch.randn((100,6,111,111))
    y_train = torch.randn((100,1))
        
    x_test = torch.randn((1,6,111,111))
    y_test = torch.randn((1,1))
        
    train_DS = torch.utils.data.TensorDataset(x_train, y_train)
    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    
    return train_DL, x_test, y_test
# %% MODEL
class sCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Block1 = self.conv_block(6,16)
        self.Block2 = self.conv_block(16,16)
        self.Block3 = self.conv_block(16,32)
        self.Block4 = self.conv_block(32,32)
        self.Block5 = self.conv_block(32,64)
        self.Block6 = self.conv_block(64,64)
        self.FC_Block = self.fc_block()
       
    def conv_block(self, inc, outc):
        b = nn.Sequential(
                nn.Conv2d(inc,outc,3,padding=1),
                nn.ReLU(),
                nn.Conv2d(outc,outc,3,padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                )
        return b
    
    def fc_block(self):
        b = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                )
        return b
    
    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x) 
        x = self.Block3(x) 
        x = self.Block4(x) 
        x = self.Block5(x) 
        x = self.Block6(x) 
        x = torch.flatten(x,1)
        x = self.FC_Block(x)
        return x
    
    def DOtraining(self, train_DL, x_test, y_test):
        optimizer = optim.Adam(self.parameters(), lr = LR)
        
        self.train()
        epoch = EPOCH
        loss_train = np.zeros((1,EPOCH))
        loss_test = np.zeros((1,EPOCH))
        for ep in range(epoch):
            n = 0
            for x_batch, y_batch in train_DL:
                n+=1

                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                y_pred = self(x_batch)
                
                loss = F.mse_loss(y_pred, y_batch)
                loss_train[0,ep] += loss

                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    x_test = x_test.to(DEVICE)
                    y_test = y_test.to(DEVICE)
                    y_pred = self(x_test)
                    loss_test[0,ep] += F.mse_loss(y_pred, y_test)
                
            if ep % 1 == 0:
                loss_train[0,ep] = loss_train[0,ep]/n
                loss_test[0,ep] = loss_test[0,ep]/n
                print('loss_epoch = ' + str(loss_train[0,ep]))
                print('loss_test = ' + str(loss_test[0,ep]))
        return loss_train, loss_test
# %% training
train_DL, x_test, y_test = data_load()
model = sCNN() 
model = model.to(DEVICE)
loss_train, loss_test = model.DOtraining(train_DL, x_test, y_test) # %% TRAINING
# %% testing
s = calc_s_val(x_test,model,DEVICE)
