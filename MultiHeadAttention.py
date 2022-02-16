import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# %%
seq_len=2000
word_dim=512
j=np.array([i for i in range(seq_len)])
v=np.array([[i+1] for i in range(int(word_dim/2))])
p1=np.sin(j/10000**(2*v/word_dim))
p2=np.cos(j/10000**(2*v/word_dim))
P=np.zeros((word_dim,seq_len))
P[ ::2,:]=p1
P[1::2,:]=p2
plt.imshow(P)
# %%
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim //  n_heads
        
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.reshape(batch_size, self.n_heads, -1, self.head_dim)
        K = K.reshape(batch_size, self.n_heads, -1, self.head_dim)
        V = V.reshape(batch_size, self.n_heads, -1, self.head_dim)
        
        energy = Q @ K.permute(0,1,3,2) / self.scale

        attention = torch.softmax(energy, dim=-1)
        
        x = attention @ V
        
        x = x.permute(0, 2, 1, 3)
        
        x = x.reshape(batch_size, -1, self.hidden_dim)
        
        x - self.fc_o(x)
        
        return x
# %%
HIDDEN_DIM = 512
N_HEADS = 8
BATCH_SIZE = 32
seq_len = 10
HEAD_DIM = HIDDEN_DIM // N_HEADS
# %%
MHA = MultiHeadAttentionLayer(HIDDEN_DIM,N_HEADS)
# %%
Q_test = torch.randn(BATCH_SIZE,seq_len,HIDDEN_DIM)
K_test = torch.randn(BATCH_SIZE,seq_len,HIDDEN_DIM)
V_test = torch.randn(BATCH_SIZE,seq_len,HIDDEN_DIM)
MHA(Q_test, K_test, V_test)


A = torch.randn(1,2,3,2)
Ap = A.permute(0,2,1,3).contiguous()

Are = A.view(1,-1,4)

Apre = Ap.view(1,-1,4)


# %%
# import numpy as np
# import matplotlib.pyplot as plt

# t = np.array([[0.01*i for i in range(100)]])
# x = np.exp(1j*2* np.pi * 1 * t)
# X = np.fft.fft(x)
# xp = np.hstack([x,np.zeros((1,100))])
# Xp = np.fft.fft(xp)
# plt.close('all')
# plt.figure(); plt.plot(np.squeeze(np.real(x)))
# plt.figure(); plt.stem(np.squeeze(np.abs(X)))
# plt.figure(); plt.plot(np.squeeze(np.real(xp)))
# plt.figure(); plt.stem(np.squeeze(np.abs(Xp)))