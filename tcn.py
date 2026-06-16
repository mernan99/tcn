from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import weight_norm
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=0.5, std=0.5)
])

full_train_data = torchvision.datasets.FashionMNIST('./data/', train=True, download=True, transform=train_transform)

N_train = 10000
train_data = torch.utils.data.Subset(full_train_data, np.arange(N_train))

N_valid = 2000
valid_data = torch.utils.data.Subset(train_data, np.arange(N_train-N_valid, N_train))


train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=256, shuffle=True)

# Utilise the last 20% of the training data for validation
N_valid = 2000
valid_data = torch.utils.data.Subset(train_data, np.arange(N_train-N_valid, N_train))

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


class LoRAConv1d(nn.Module):
    def __init__(self, base_conv, rank, alpha):
        """
        base_conv: nn.Conv1d layer to wrap
        rank: low-rank size
        alpha: scaling factor
        """
        super().__init__()

        self.base = base_conv
        self.rank = rank
        self.alpha = alpha

        in_ch = base_conv.in_channels
        out_ch = base_conv.out_channels
        ks = base_conv.kernel_size
        stride = base_conv.stride
        padding = base_conv.padding
        dilation = base_conv.dilation

        # LoRA low-rank adapters
        self.A = nn.Conv1d(in_ch, rank, ks,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           bias=False)

        self.B = nn.Conv1d(rank, out_ch, 1, bias=False)

        # scale for stability
        self.scale = alpha / rank

        # Freeze original conv
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        # base conv + scaled low-rank update
        return self.base(x) + self.scale * self.B(self.A(x))



class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank, alpha):
        """
        base_linear: nn.Linear layer
        rank: low-rank size
        alpha: scaling factor
        """
        super().__init__()

        self.base = base_linear
        self.rank = rank
        self.alpha = alpha

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        self.A = nn.Linear(in_f, rank, bias=False)
        self.B = nn.Linear(rank, out_f, bias=False)

        self.scale = alpha / rank

        # Freeze original weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


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
        self.loss_fn = nn.CrossEntropyLoss()

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

CH0=1
  
input_size=CH0
output_size=10
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
  correct, total = 0, 0
  loss_sum = 0

  with torch.no_grad():
      for xb, yb in loader:
          # reshape images to (N, 1, 784)
          xb = xb.view(xb.size(0), 1, -1).to(device)
          yb = yb.to(device)

          pred = model(xb)
          pred = pred[:, :, -1]
          if loss_fn:
              loss_sum += loss_fn(pred, yb).item()

          _, predicted = pred.max(1)
          correct += (predicted == yb).sum().item()
          total += yb.size(0)

  if loss_fn:
      return correct / total, loss_sum / len(loader)
  return correct / total


def train_model(model, train_loader, valid_loader, N_epochs=15):
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(N_epochs):
      model.train()
      for xb, yb in train_loader:
          xb = xb.view(xb.size(0), 1, -1).to(device)
          yb = yb.to(device)

          pred = model(xb)
          pred = pred[:, :, -1]
          loss = loss_fn(pred, yb)

          optim.zero_grad()
          loss.backward()
          optim.step()

      train_acc, train_loss = accuracy(model, train_loader, loss_fn)
      valid_acc, valid_loss = accuracy(model, valid_loader, loss_fn)

      print(f"Epoch {epoch+1:2d} | "
            f"Train Loss {train_loss:.4f}, Acc {train_acc:.3f} | "
            f"Val Loss {valid_loss:.4f}, Acc {valid_acc:.3f}")



train_model(model, train_loader, valid_loader, N_epochs=15)