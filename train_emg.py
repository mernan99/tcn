import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data_utils import load_emg_glove_windows, EMGGloveDataset
from tcn_model import Learner


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Load data
Xs, Ys, fs = load_emg_glove_windows("datas/P02.mat", win_sec=1.0, step_sec=0.5)

eps = 1e-8
# Xs shape assumed (N, C, T)
mu = Xs.mean(axis=2, keepdims=True)
sd = Xs.std(axis=2, keepdims=True)
Xs = (Xs - mu) / (sd + eps)



#  Split train/val/test (70/15/15)
# X_train, X_tmp, y_train, y_tmp = train_test_split(Xs, Ys, test_size=0.3, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

n = len(Xs)
i1 = int(0.70 * n)
i2 = int(0.85 * n)

gap = 2  # number of windows to skip as buffer (tune: 2–10)
X_train, y_train = Xs[:i1], Ys[:i1]
X_val,   y_val   = Xs[i1+gap:i2], Ys[i1+gap:i2]
X_test,  y_test  = Xs[i2+gap:], Ys[i2+gap:]


y_mu = y_train.mean(axis=0, keepdims=True)
y_sd = y_train.std(axis=0, keepdims=True)

y_train = (y_train - y_mu) / (y_sd + eps)
y_val   = (y_val   - y_mu) / (y_sd + eps)
y_test  = (y_test  - y_mu) / (y_sd + eps)

train_loader = DataLoader(EMGGloveDataset(X_train, y_train), batch_size=64, shuffle=True)
valid_loader = DataLoader(EMGGloveDataset(X_val, y_val), batch_size=256, shuffle=False)
test_loader  = DataLoader(EMGGloveDataset(X_test, y_test), batch_size=256, shuffle=False)

#  Model config
N_C = 2
input_size = X_train.shape[1]
output_size = y_train.shape[1]

num_channels = [64] * 8
Dil = [[1,1],[2,2],[4,4],[8,8],[16,16],[32,32],[64,64],[128,128]]
K_S = [[7,7]] * 8
Str = [[1,1]] * 8

F_Ns = torch.tensor([64, output_size])
dropout_F = 0.1
dropout_cnn = 0.1

model = Learner(
    N_C=N_C, input_size=input_size, output_size=output_size,
    num_channels=num_channels, K_S=K_S, Dil=Dil, Str=Str,
    dropout_cnn=dropout_cnn, F_Ns=F_Ns, dropout_F=dropout_F,
    Receptive_F=np.sum((np.array(K_S)-1)*np.array(Dil)),
    Norm=0, Residual=True
).to(device)

embed_dim = getattr(model.tcn, "embed_dim", 128)  
model.reg_head = nn.Sequential(
    nn.Linear(embed_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(128, output_size)
).to(device)

xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
with torch.no_grad():
    tokens = model(xb)
    print("tokens", tokens.shape)  # expect (B, T, D)
    # pred = model.reg_head(tokens.mean(dim=1))
    win_feat = tokens[:, -1, :]
    pred = model.reg_head(win_feat)

    print("pred", pred.shape, "yb", yb.shape)  # expect (B,out) matches yb

# Train helpers
# def eval_mse(model, loader, loss_fn):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)

#             tokens = model(xb)                 # (B, T, D)  <-- token-first output
#             # win_feat = tokens.mean(dim=1)  # (B, D)
#             win_feat = tokens[:, -1, :]
#             pred = model.reg_head(win_feat)    # (B, out)

#             total += loss_fn(pred, yb).item()
#     return total / len(loader)
    
def eval_mse_seq(model, loader, loss_fn):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)   # xb (B,C,T), yb (B,T,out)
            tokens = model(xb)                      # (B,T,D)
            pred_seq = model.reg_head(tokens)       # (B,T,out)
            total += loss_fn(pred_seq, yb).item()
    return total / len(loader)

# def train_model(model, train_loader, valid_loader, epochs=100):
#     loss_fn = nn.MSELoss()
#     # optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
#     optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optim, mode="min", factor=0.5, patience=3, threshold=1e-4
#     )

#     train_losses, val_losses = [], []

#     for ep in range(epochs):
#         model.train()
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             tokens = model(xb)                 # (B, T, D)
#             # win_feat = tokens.mean(dim=1)      # (B, D)
#             win_feat = tokens[:, -1, :]
#             pred = model.reg_head(win_feat)    # (B, out)
#             loss = loss_fn(pred, yb)

#             optim.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
#             optim.step()

#         tr = eval_mse(model, train_loader, loss_fn)
#         va = eval_mse(model, valid_loader, loss_fn)
#         scheduler.step(va)

#         train_losses.append(tr)
#         val_losses.append(va)
#         print(f"Epoch {ep+1:02d} | Train MSE {tr:.5f} | Val MSE {va:.5f}")

#     return train_losses, val_losses


def train_model_seq(model, train_loader, valid_loader, epochs=100):
    loss_fn = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=3, threshold=1e-4
    )

    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            tokens = model(xb)                 # (B,T,D)
            pred_seq = model.reg_head(tokens)  # (B,T,out)
            loss = loss_fn(pred_seq, yb)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optim.step()

        tr = eval_mse_seq(model, train_loader, loss_fn)
        va = eval_mse_seq(model, valid_loader, loss_fn)
        scheduler.step(va)

        train_losses.append(tr)
        val_losses.append(va)
        print(f"Epoch {ep+1:02d} | Train MSE {tr:.5f} | Val MSE {va:.5f}")

    return train_losses, val_losses

#  Run
train_losses, val_losses = train_model_seq(model, train_loader, valid_loader, epochs=100)

plt.figure()
plt.plot(train_losses, label="Train MSE")
plt.plot(val_losses, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("EMG → Glove Regression")
plt.legend()
plt.show()

#  Final test score
test_mse = eval_mse_seq(model, test_loader, nn.MSELoss())
print("Test MSE:", test_mse)

# #  Plot a sample
# xb, yb = next(iter(valid_loader))
# xb, yb = xb.to(device), yb.to(device)

# with torch.no_grad():
#     tokens = model(xb)                 # (B, T, D)
#     # pred = model.reg_head(tokens.mean(dim=1))  # (B, out)
#     pred = model.reg_head(tokens[:, -1, :])

# # Move to numpy
# pred_np = pred.cpu().numpy()
# true_np = yb.cpu().numpy()

# # Un-normalise
# pred_un = pred_np * (y_sd + eps) + y_mu
# true_un = true_np * (y_sd + eps) + y_mu


# plt.figure()
# plt.plot(true_un[0], label="True")
# plt.plot(pred_un[0], label="Pred")
# plt.legend()
# plt.title("One Glove Sample")
# plt.show()
xb, yb = next(iter(valid_loader))
xb, yb = xb.to(device), yb.to(device)

with torch.no_grad():
    tokens = model(xb)
    pred = model.reg_head(tokens)     # (B,T,Cg)

pred_np = pred.cpu().numpy()
true_np = yb.cpu().numpy()

# Un-normalise if you normalised Ys:
pred_un = pred_np * (y_sd + eps) + y_mu
true_un = true_np * (y_sd + eps) + y_mu

# plot one channel across time
ch = 0
plt.figure()
plt.plot(true_un[0, :, ch], label="True")
plt.plot(pred_un[0, :, ch], label="Pred")
plt.legend()
plt.title(f"Seq2Seq glove channel {ch}")
plt.show()
