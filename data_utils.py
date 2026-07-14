import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path


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

def load_all_emg_glove_windows(data_dir="datas", win_sec=1.0, step_sec=0.5, lag_sec=0.0):
    """Load and concatenate windows from every .mat file in a directory."""
    mat_files = sorted(Path(data_dir).glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {Path(data_dir).resolve()}")

    all_X, all_Y = [], []
    expected_fs = None
    expected_x_shape = None
    expected_y_shape = None

    for mat_path in mat_files:
        X, Y, fs = load_emg_glove_windows(
            mat_path,
            win_sec=win_sec,
            step_sec=step_sec,
            lag_sec=lag_sec,
        )

        if expected_fs is None:
            expected_fs = fs
            expected_x_shape = X.shape[1:]
            expected_y_shape = Y.shape[1:]
        else:
            if fs != expected_fs:
                raise ValueError(
                    f"Sampling-frequency mismatch: {mat_path.name} uses {fs} Hz, "
                    f"expected {expected_fs} Hz."
                )
            if X.shape[1:] != expected_x_shape or Y.shape[1:] != expected_y_shape:
                raise ValueError(
                    f"Shape mismatch in {mat_path.name}: X{X.shape[1:]}, Y{Y.shape[1:]}; "
                    f"expected X{expected_x_shape}, Y{expected_y_shape}."
                )

        all_X.append(X)
        all_Y.append(Y)
        print(f"Loaded {mat_path.name}: {len(X)} windows")

    Xs = np.concatenate(all_X, axis=0)
    Ys = np.concatenate(all_Y, axis=0)
    print(f"Loaded {len(mat_files)} files and {len(Xs)} total windows")

    return Xs, Ys, expected_fs


class EMGGloveDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
