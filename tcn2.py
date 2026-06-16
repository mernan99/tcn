import os
import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split
from google.colab import drive
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm


# ============================================================
# Config
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
drive.mount('/content/drive')

DATA_FOLDER = "/content/drive/MyDrive/myoki"   
WINDOW = 1024
BATCH_SIZE = 32
N_EPOCHS = 50
DROPOUT = 0.3
RANDOM_SEED = 42
NUM_ANGLES = 18  # MyoKi calibrated glove DOF

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Temporal Block (TCN components)
# ============================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, N_C, Ch_in, Ch_out, KS, stride, dilation,
                 dropout_p=0.2, Norm=0, Residual=True):
        super().__init__()

        layers = []
        for n in range(N_C):
            in_ch = Ch_in if n == 0 else Ch_out

            conv = weight_norm(
                nn.Conv1d(in_ch, Ch_out, KS[n],
                          stride=stride[n],
                          padding=(KS[n] - 1) * dilation[n],
                          dilation=dilation[n])
            )
            conv.weight.data.normal_(0, 0.01)

            chomp = Chomp1d((KS[n] - 1) * dilation[n])
            relu = nn.ReLU()
            dropout = nn.Dropout(dropout_p)

            layers.append(conv)
            if stride[n] == 1:
                layers.append(chomp)
            if Norm == 1:
                layers.append(nn.BatchNorm1d(Ch_out))
            layers.append(relu)
            layers.append(dropout)

        self.net = nn.Sequential(*layers)
        self.Residual = Residual
        if Residual:
            self.downsample = nn.Conv1d(Ch_in, Ch_out, 1) if Ch_in != Ch_out else None
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) if self.Residual else out


class ModalityTCN(nn.Module):
    def __init__(self, input_ch, num_channels, dropout_p=0.3):
        super().__init__()
        layers = []
        N_C = 2
        KS = [4, 4]

        for i in range(len(num_channels)):
            block = TemporalBlock(N_C,
                                 Ch_in=input_ch if i == 0 else num_channels[i - 1],
                                 Ch_out=num_channels[i],
                                 KS=KS,
                                 stride=[1, 1],
                                 dilation=[2 ** i, 2 ** i],
                                 dropout_p=dropout_p)
            layers.append(block)

        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        y = self.tcn(x)
        return y[:, :, -1]  # final timestep features


class MidFusionTCNRegression(nn.Module):
    def __init__(self, num_angles):
        super().__init__()

        deep_channels = [32] * 6  # 6-layer TCN

        self.emg_tcn = ModalityTCN(12, deep_channels, DROPOUT)
        self.acc_tcn = ModalityTCN(27, deep_channels, DROPOUT)
        self.gyro_tcn = ModalityTCN(27, deep_channels, DROPOUT)

        fusion_in = 32 * 3

        self.fc = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, num_angles)
        )

    def forward(self, emg, acc, gyro):
        feat = torch.cat([
            self.emg_tcn(emg),
            self.acc_tcn(acc),
            self.gyro_tcn(gyro)
        ], dim=1)
        return self.fc(feat)  # (B, 18 angles)


def pad_or_crop(seq, length):
    if seq.shape[0] > length:
        return seq[:length]
    if seq.shape[0] < length:
        pad = np.zeros((length - seq.shape[0], seq.shape[1]))
        return np.concatenate([seq, pad], axis=0)
    return seq


def load_sequences_from_mat(file_path, window=WINDOW):
    mat = sio.loadmat(file_path)

    emg = np.array(mat["emg"])
    acc = np.array(mat["acc"])
    gyro = np.array(mat["gyro"])
    angles = np.array(mat["glove_calibrated"])  # <-- target!

    task = np.array(mat["task"]).flatten()
    rep = np.array(mat["repetition"]).flatten()

    X_e, X_a, X_g, Y = [], [], [], []

    for t, r in sorted(set(zip(task, rep))):
        idx = (task == t) & (rep == r)
        if idx.sum() < 10: continue

        seq_emg = pad_or_crop(emg[idx], window)
        seq_acc = pad_or_crop(acc[idx], window)
        seq_gyro = pad_or_crop(gyro[idx], window)
        seq_ang = pad_or_crop(angles[idx], window)

        X_e.append(seq_emg.T)
        X_a.append(seq_acc.T)
        X_g.append(seq_gyro.T)
        Y.append(seq_ang[-1])  # last timestep

    return X_e, X_a, X_g, Y


def load_dataset(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".mat")]
    print("Files:", files)

    X_e, X_a, X_g, Y = [], [], [], []
    for f in files:
        Xe, Xa, Xg, Yv = load_sequences_from_mat(os.path.join(folder, f))
        X_e.extend(Xe)
        X_a.extend(Xa)
        X_g.extend(Xg)
        Y.extend(Yv)

    return map(np.stack, (X_e, X_a, X_g)), np.array(Y)


def normalize(x):
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True) + 1e-8
    return (x - mean) / std


class MyoKiAnglesDataset(Dataset):
    def __init__(self, X_emg, X_acc, X_gyro, Y):
        self.emg = torch.tensor(X_emg, dtype=torch.float32)
        self.acc = torch.tensor(X_acc, dtype=torch.float32)
        self.gyro = torch.tensor(X_gyro, dtype=torch.float32)
        self.angles = torch.tensor(Y, dtype=torch.float32)

    def __len__(self): return len(self.angles)

    def __getitem__(self, i):
        return self.emg[i], self.acc[i], self.gyro[i], self.angles[i]


def train(model, train_loader, val_loader, epochs, lr=1e-3):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for emg, acc, gyro, y in train_loader:
            emg, acc, gyro, y = emg.to(DEVICE), acc.to(DEVICE), gyro.to(DEVICE), y.to(DEVICE)
            pred = model(emg, acc, gyro)
            loss = criterion(pred, y)
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:2d} | Train MSE: {np.mean(losses):.4f}")



if __name__ == "__main__":
    (X_emg, X_acc, X_gyro), Y = load_dataset(DATA_FOLDER)

    X_emg = normalize(X_emg)
    X_acc = normalize(X_acc)
    X_gyro = normalize(X_gyro)

    idx_train, idx_val = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42)

    train_ds = MyoKiAnglesDataset(X_emg[idx_train], X_acc[idx_train], X_gyro[idx_train], Y[idx_train])
    val_ds = MyoKiAnglesDataset(X_emg[idx_val], X_acc[idx_val], X_gyro[idx_val], Y[idx_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MidFusionTCNRegression(NUM_ANGLES).to(DEVICE)
    train(model, train_loader, val_loader, N_EPOCHS)
