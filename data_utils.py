import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat


def ensure_LC(x):
    x = np.array(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < x.shape[1] and x.shape[1] > 1000:
        x = x.T
    return x


def load_emg_glove_windows(mat_path, win_sec=1.0, step_sec=0.5, lag_sec=0.0):
    data = loadmat(mat_path)

    emg = ensure_LC(data["emg"])
    glove = ensure_LC(data["glove_calibrated"])

    frequency = data["frequency"]
    fs = int(np.array(frequency).squeeze()) if np.array(frequency).size == 1 else int(np.array(frequency).ravel()[0])

    win  = int(win_sec * fs)
    step = int(step_sec * fs)
    lag  = int(lag_sec * fs)

    L = min(len(emg), len(glove))
    emg = emg[:L]
    glove = glove[:L]

    Xs, Ys = [], []
    # we need start+win+lag <= L
    for start in range(0, L - win - lag, step):
        xw = emg[start:start+win]                  # (T, C_emg)
        yw = glove[start+lag:start+lag+win]        # (T, C_glove)  <-- shifted by lag

        Xs.append(xw)
        Ys.append(yw)

    Xs = np.stack(Xs)                              # (N, T, C_emg)
    Ys = np.stack(Ys)                              # (N, T, C_glove)

    # Conv1d wants (N, C, T)
    Xs = np.transpose(Xs, (0, 2, 1))               # (N, C_emg, T)

    return Xs, Ys, fs


class EMGGloveDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
