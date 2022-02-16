import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
# from torchvision import transforms, datasets
# %%
BATCH_SIZE = 32
EPOCHS = 1000
LR = 0.001
# %% DEVICE setting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# %%
# train_DS = datasets.MNIST(root = "data/MNIST", 
#                           train=True, 
#                           transform = transforms.ToTensor())
# test_DS = datasets.MNIST(root = "data/MNIST", 
#                           train=False, 
#                           transform = transforms.ToTensor())
def data_load():
    x_train=torch.linspace(-1,1,201).view(-1,1)
    y_train=x_train**2
    x_test=torch.linspace(-1,1,21).view(-1,1)
    y_test=x_test**2
    train_DS = torch.utils.data.TensorDataset(x_train,y_train)
    train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
    return train_DL, x_test, y_test

def model_out(model,x,DEVICE):
    model.eval()
    out_val= np.zeros((1,x.shape[0]))
    with torch.no_grad():
        x_test = x.to(DEVICE)
        y_test = model(x_test)
        out_val = y_test.cpu().numpy()
    return out_val
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,5)
        # pick 1
        self.fc2 = nn.Linear(5,3)
        self.fc3 = nn.Linear(3,1)
        # pick 2
        # self.conv1=nn.Conv1d(1,1,(1,3))
        # self.fc3 = nn.Linear(3,1)
        # pick 3
        # self.conv1=nn.Conv1d(1,1,(1,5))
        # self.fc3 = nn.Linear(1,1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        # pick 1
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        # pick 2
        # x = torch.reshape(x,(-1,1,1,5))
        # x = self.conv1(x)
        # x = torch.reshape(x,(-1,3))
        # x = torch.sigmoid(x)
        # x = self.fc3(x)
        # pick 3
        # x = torch.reshape(x,(-1,1,1,5))
        # x = self.conv1(x)
        # x = torch.reshape(x,(-1,1))
        # x = torch.sigmoid(x)
        # x = self.fc3(x)
        
        return x
    def DOtraining(self, train_DL, x_test, y_test):
        optimizer = optim.Adam(self.parameters(), lr = LR)
        
        loss_train = np.zeros((1,EPOCHS))
        loss_test = np.zeros((1,EPOCHS))
        for ep in range(EPOCHS):
            self.train()
            iteration = 0
            for x_batch, y_batch in train_DL:
                iteration+=1

                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                optimizer.zero_grad()
                y_pred = self(x_batch)
                
                loss = F.mse_loss(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                loss_train[0,ep] += loss
                
            self.eval()
            with torch.no_grad():
                x_test = x_test.to(DEVICE)
                y_test = y_test.to(DEVICE)
                y_pred = self(x_test)
                loss_test[0,ep] += F.mse_loss(y_pred, y_test)
                
            if ep % 1 == 0:
                loss_train[0,ep] = loss_train[0,ep]/iteration
                print("Epoch : ", str(ep), 
                      " loss_train = ", str(round(loss_train[0,ep],3)), 
                      "\t loss_test = ", str(round(loss_test[0,ep],3)))
        return loss_train, loss_test
# %% data load, model gen, weight check (before training), training
train_DL, x_test, y_test = data_load()
# for x_batch, y_batch in train_DL:
    # print(x_batch)
model = Net().to(DEVICE)
loss_train, loss_test = model.DOtraining(train_DL, x_test, y_test)
torch.save(model.state_dict(), "saved_net/59page.pt")
# %%
out_val=model_out(model,x_test,DEVICE)
plt.close('all')
plt.figure()
plt.plot(x_test,out_val)
plt.grid()
# %% weight check
model_before_training=Net()
# print(model_before_training.fc2.weight.shape)
# print(model_before_training.fc2.weight)
# print(model.fc2.weight)
# print(model.conv1.weight)
# %% output check
# F1=model.fc1.weight
# B1=torch.reshape(model.fc1.bias,(-1,1))
# F2=model.fc2.weight
# B2=torch.reshape(model.fc2.bias,(-1,1))
# F3=model.fc3.weight
# B3=model.fc3.bias
# from torch import sigmoid as sig
# x=1
# print( F3 @ sig(F2 @ sig(F1 * x + B1) + B2)  + B3)
# %% weight sum = 1 ?
# print(torch.sum(F1,dim=0))
# print(torch.sum(F2,dim=1).reshape(-1,1))
# print(torch.sum(F3,dim=1))
# %% 
plt.close('all')
model = Net().to(DEVICE)
# model.load_state_dict(torch.load("saved_net/59page.pt"))
model.eval()
row = np.linspace(-2,2,100)
col = np.linspace(-2,2,100)
val = np.zeros((len(row),len(col)))
for r in range(len(row)):
    for c in range(len(col)):
        with torch.no_grad():
            model.fc2.weight[0,0]=row[r]
            model.fc2.weight[0,1]=col[c]
            
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            y_pred = model(x_batch)
            
            loss = F.mse_loss(y_pred, y_batch)
            val[r,c]=loss

[R,C]=np.meshgrid(row,col)
plt.figure()
ax=plt.axes(projection="3d")
ax.plot_surface(R,C,val,cmap='plasma')
# %% autograd
w=torch.tensor([1, 2, 3, 4], requires_grad=True, dtype=torch.float32)
j=w[0]+2*w[1]
g=j*w[2]
h=j*w[3]
f=2*g+3*h
f.backward()
# print(w.grad)
