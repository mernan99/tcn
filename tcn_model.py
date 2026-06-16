import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.embed_dim = 128
        self.proj = nn.Linear(TCN_out, self.embed_dim)

        
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
        
        # batch_size=inputs.size()[0]
        
        # y_temp = self.network(inputs)  # input should have dimension (N, C, L)
        
        # y_temp=y_temp[:,:,self.Receptive_F:].clone()
        
        # y_temp=y_temp.transpose(1,2)
        
        # out=self.F(y_temp.reshape([-1,y_temp.size()[-1]]))
        
        # out=out.reshape([batch_size,y_temp.size()[1],-1]).transpose(1,2)
        
        # return out
        feats = self.network(inputs)          # (B, C, T)

    # feats = feats[:, :, self.Receptive_F:]  # <-- remove / keep commented

        tokens = feats.transpose(1, 2)        # (B, T, C)
        tokens = self.proj(tokens)            # (B, T, D)
        return tokens

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
        embed_dim = self.tcn.embed_dim
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_size)
        ).to(device)

        self.optim = torch.optim.Adam(self.parameters(), lr=eta)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.tcn(x)

    def step(self, xb, yb):
        tokens = self.forward(xb)          # (B, T, D)

        # pool over time to get a window representation
        # win_feat = tokens.mean(dim=1)      # (B, D)  (or tokens[:, -1, :] if causal)

        win_feat = tokens[:, -1, :] 

        pred = self.reg_head(win_feat)     # (B, output_size)
        loss = self.loss_fn(pred, yb)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()