from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, Dataset
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import loadmat
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = loadmat('datas/P02.mat')
emg = data['emg']
acc = data['acc']
frequency = data['frequency']
glove = data['glove']
glove_calibrated = data['glove_calibrated']
grasp = data['grasp']
gyro = data['gyro']
repetition = data['repetition']
task = data['task']
timestamp = data['timestamp']



def ensure_LC(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x[:, None]
    # If it's (C, L), transpose to (L, C)
    if x.shape[0] < x.shape[1] and x.shape[1] > 1000:
        # heuristic: length usually much bigger than channels
        x = x.T
    return x


emg_np = ensure_LC(emg)
glove_np = ensure_LC(glove_calibrated)


# Sampling frequency
fs = int(np.array(frequency).squeeze()) if np.array(frequency).size == 1 else int(np.array(frequency).ravel()[0])

win_sec = 1.0      
step_sec = 0.5 

win = int(win_sec * fs)
step = int(step_sec * fs)


L = min(len(emg_np), len(glove_np))
emg_np = emg_np[:L]
glove_np = glove_np[:L]

Xs, Ys = [], []
for start in range(0, L - win, step):
    xw = emg_np[start:start+win]        # (win, C_emg)
    yw = glove_np[start:start+win]      # (win, C_glove)

    # Target: mean glove pose over the window (regression target)
    y_target = yw.mean(axis=0)          # (C_glove,)

    Xs.append(xw)
    Ys.append(y_target)

Xs = np.stack(Xs)                        # (N, win, C_emg)
Ys = np.stack(Ys)                        # (N, C_glove)

# Conv1d expects (N, C, L)
Xs = np.transpose(Xs, (0, 2, 1))         # (N, C_emg, win)

# Normalise 
# EMG: normalise per-channel over dataset
Xs = (Xs - Xs.mean(axis=(0, 2), keepdims=True)) / (Xs.std(axis=(0, 2), keepdims=True) + 1e-8)
# Glove: normalise per output dimension (helps)
Ys = (Ys - Ys.mean(axis=0, keepdims=True)) / (Ys.std(axis=0, keepdims=True) + 1e-8)


# Dataset class FIRST

class EMGGloveDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Split into train/val/test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    Xs, Ys, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=42
)

# DataLoaders

train_loader = DataLoader(EMGGloveDataset(X_train, y_train), batch_size=64, shuffle=True)
valid_loader = DataLoader(EMGGloveDataset(X_val, y_val), batch_size=256, shuffle=False)
test_loader  = DataLoader(EMGGloveDataset(X_test, y_test), batch_size=256, shuffle=False)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, N_C, Ch_in, Ch_out, KS, stride, dilation, dropout_p=0.2, Norm=0, Residual=True):
        super(TemporalBlock, self).__init__()
                
        module=[]
        for n in range(N_C):
            
            if n==0:
                input_channels=Ch_in
                
            else:
                input_channels=Ch_out
            
            
            conv = weight_norm(nn.Conv1d(input_channels, Ch_out, KS[n], stride=stride[n], padding=(KS[n]-1)*dilation[n],\
                                  dilation=dilation[n]))
            
            conv.weight.data.normal_(0, 0.01)
            
            chomp = Chomp1d((KS[n]-1)*dilation[n])

            
            relu = nn.ReLU()
            dropout= nn.Dropout(dropout_p)
            
            module.append(conv)
                        
            if stride[n]==1:
                module.append(chomp)

            if Norm == 1:
                
                module.append(nn.BatchNorm1d(Ch_out))
            
            module.append(relu)
            module.append(dropout)
        
        
        self.net = nn.Sequential(*module)

        self.Residual=Residual
        if self.Residual:
            self.downsample = nn.Conv1d(Ch_in, Ch_out, 1) if Ch_in != Ch_out else None
            self.relu = nn.ReLU()
        


    def forward(self, x):

        out = self.net(x)

        if self.Residual:
            res = x if self.downsample is None else self.downsample(x)
            out=self.relu(out + res)
        
        return out


class LoRALinear(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # Low-rank matrices
        self.A = nn.Parameter(torch.zeros(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        base = self.linear(x)
        lora = (x @ self.A.T) @ self.B.T
        return base + self.scaling * lora


class LoRAConv1d(nn.Module):
    def __init__(self, conv, rank, alpha):
        super().__init__()

        self.conv = conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size[0]

        # Low-rank convs
        self.A = nn.Conv1d(
            in_channels,
            rank,
            kernel_size=1,
            bias=False
        )

        self.B = nn.Conv1d(
            rank,
            out_channels,
            kernel_size=kernel_size,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=False
        )

        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):

        base = self.conv(x)

        lora = self.B(self.A(x))

        return base + self.scaling * lora


class TCN(nn.Module):
    def __init__(self, N_C, input_size, output_size, num_channels, K_S, Dil, Str, dropout, F_Ns, dropout_F, \
                 Receptive_F, Norm=0, Residual=True, Loader=[], Load_end=False):
        
        super(TCN, self).__init__()
        
        
        self.F_Ns=F_Ns
        self.Receptive_F=Receptive_F
        
        if Loader!=[]:
            
            tcn=torch.load(Loader)
            self.network=tcn.tcn.network.to(device)
            
                
        
        else:
        
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):

                dilation_size = Dil[i]
                stride_size=Str[i]


                in_channels = input_size if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]


                layers += [TemporalBlock(N_C, in_channels, out_channels, K_S[i], stride=Str[i], dilation=Dil[i],
                                          dropout_p=dropout, Norm=Norm, Residual=Residual)]
                                          
                self.network = nn.Sequential(*layers).to(device)
                
        TCN_out=num_channels[-1]
        
        if Load_end and Loader!=[]:
            
            self.F=tcn.tcn.F.to(device)
                
        else:
            
            module_F=[]
            
            F_Ns[0]=TCN_out
            
            for n in range(1,F_Ns.size()[0]):
                        
                module_F.append(nn.Linear(F_Ns[n-1],F_Ns[n]))

                if n<F_Ns.size()[0]-1:

                    module_F.append(nn.ReLU())
                    module_F.append(nn.Dropout(dropout_F))
                
            self.F=nn.Sequential(*module_F).to(device)
        
        
    def LoRA_Setup(self, rank, alpha, rank_f=0):
        
        for param in self.network.parameters():
            
            param.requires_grad = False
                
        for m in range(len(self.network)):
        
            for n in range(len(self.network[m].net)):

                if isinstance(self.network[m].net[n], nn.Conv1d):
                    
                    self.network[m].net[n]=LoRAConv1d(self.network[m].net[n],rank,alpha)
        
        if rank_f>0:
            
            for m in range(len(self.F)):    

                if isinstance(self.F[m], nn.Linear):
                    
                    self.F[m]=LoRALinear(self.F[m],rank_f,alpha)
                    
                    
    def forward(self, inputs):
        
        """Inputs have to have dimension (N, C_in, L_in)"""
        
        batch_size=inputs.size()[0]
        
        y_temp = self.network(inputs)  # input should have dimension (N, C, L)
        
        y_temp=y_temp[:,:,self.Receptive_F:].clone()
        
        y_temp=y_temp.transpose(1,2)
        
        out=self.F(y_temp.reshape([-1,y_temp.size()[-1]]))
        
        out=out.reshape([batch_size,y_temp.size()[1],-1]).transpose(1,2)
        
        return out

class Learner(nn.Module):
    def __init__(self, N_C, input_size, output_size, num_channels,
                 K_S, Dil, Str, dropout_cnn, F_Ns, dropout_F,
                 Receptive_F, eta=1e-3, Norm=0, Residual=True, Loader=[]):

        super().__init__()

        self.tcn = TCN(
            N_C = N_C,
            input_size = input_size,
            output_size = output_size,
            num_channels = num_channels,
            K_S = K_S,
            Dil = Dil,
            Str = Str,
            dropout = dropout_cnn,
            F_Ns = F_Ns,
            dropout_F = dropout_F,
            Receptive_F = Receptive_F,
            Norm = Norm,
            Residual = Residual,
            Loader = Loader
        )

        self.optim = torch.optim.Adam(self.parameters(), lr=eta)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.tcn(x)

    def step(self, xb, yb):
        pred = self.forward(xb)
        loss = self.loss_fn(pred, yb)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()
N_C=2

CH0=X_train.shape[1]

input_size=CH0
output_size=y_train.shape[1]
eta = 1e-3
Norm = 0
Residual = True


num_channels=[ 24 ]*6

Dil=[ [1,1], [2,2], [4,4], [8,8],  [16,16], [32,32] ]
K_S=[ [4,4] ]*6
Str=[ [1,1] ]*6

print( np.sum((np.array(K_S)-1)*np.array(Dil)) )
Receptive_F=np.sum((np.array(K_S)-1)*np.array(Dil))

F_Ns=torch.tensor([24,output_size])
dropout_F=0.3
dropout_cnn=0.3

model=Learner(N_C, input_size, output_size, num_channels, K_S, Dil, Str, dropout_cnn=dropout_cnn, \
            F_Ns=F_Ns, dropout_F=dropout_F, Receptive_F=Receptive_F, eta=eta, Norm=Norm, Residual=Residual, Loader=[])

def accuracy(model, loader, loss_fn=None):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            pred = pred.mean(dim=2)   # (N, glove_dim)

            loss_sum += loss_fn(pred, yb).item()

    return loss_sum / len(loader)

def train_model(model, train_loader, valid_loader, N_epochs=30):
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=5
    )


    train_losses = []
    valid_losses = []

    for epoch in range(N_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            pred = pred.mean(dim=2)          # (N, glove_dim)

            loss = loss_fn(pred, yb)

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss = accuracy(model, train_loader, loss_fn)
        valid_loss = accuracy(model, valid_loader, loss_fn)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        scheduler.step(valid_loss)


        print(f"Epoch {epoch+1:2d} | Train MSE {train_loss:.5f} | Val MSE {valid_loss:.5f}")

    return train_losses, valid_losses


train_losses, valid_losses = train_model(model, train_loader, valid_loader, N_epochs=70)

plt.figure()
plt.plot(train_losses, label="Train MSE")
plt.plot(valid_losses, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("EMG → Glove Regression")
plt.legend()
plt.show()

xb, yb = next(iter(valid_loader))

xb = xb.to(device)
yb = yb.to(device)

with torch.no_grad():
    pred = model(xb)
    pred = pred[:, :, -1]

plt.figure()
plt.plot(yb[0].cpu(), label="True")
plt.plot(pred[0].cpu(), label="Pred")
plt.legend()
plt.title("One Glove Sample")
plt.show()