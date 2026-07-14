# EMG-to-Glove Regression with a Temporal Convolutional Network

This project trains a Temporal Convolutional Network (TCN) to predict glove sensor values from EMG signals stored in MATLAB `.mat` files.

The model performs sequence-to-sequence regression:

- **Input:** EMG windows with shape `(batch, emg_channels, time_steps)`
- **Output:** Glove predictions with shape `(batch, time_steps, glove_channels)`

## Project structure

```text
TCN/
├── datas/
│   ├── subject_01.mat
│   ├── subject_02.mat
│   └── ...
├── data_utils.py
├── tcn_model.py
├── train_emg.py
├── .gitignore
└── README.md
```

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy torch matplotlib scikit-learn
```

The code has been used with Python 3.13 and PyTorch.

A CUDA-compatible GPU is optional. If CUDA is available, PyTorch will use it automatically. Otherwise, training runs on the CPU.

## Dataset format

Place all `.mat` files inside the `datas` folder.

Each `.mat` file must contain:

```text
emg
glove_calibrated
frequency
```

Expected meanings:

- `emg`: EMG signal array
- `glove_calibrated`: calibrated glove sensor array
- `frequency`: sampling frequency in Hz

The loader converts each signal into time-major format when necessary.

Typical signal shapes before windowing are:

```text
emg:               (samples, emg_channels)
glove_calibrated:  (samples, glove_channels)
```

All files must have compatible:

- sampling frequencies
- EMG channel counts
- glove channel counts
- window lengths

## Ignoring dataset files in Git

The dataset can be excluded from Git using:

```gitignore
datas/*.mat
```

This prevents `.mat` files from being committed while keeping the `datas` directory available locally.

To ignore the entire directory instead:

```gitignore
datas/
```

## Loading all `.mat` files

The training script loads every `.mat` file in `datas`:

```python
Xs, Ys, fs = load_all_emg_glove_windows(
    "datas",
    win_sec=1.0,
    step_sec=0.5
)
```

The loader:

1. Finds all `.mat` files in the directory.
2. Loads EMG, glove, and sampling-frequency data.
3. Splits each recording into overlapping windows.
4. Concatenates windows from all files.

With a 2,000 Hz sampling rate and a 1-second window:

```text
time_steps = 2000
```

A 0.5-second step gives 50% overlap between consecutive windows.

## Window shapes

After loading:

```text
Xs: (number_of_windows, emg_channels, time_steps)
Ys: (number_of_windows, time_steps, glove_channels)
```

For example:

```text
Xs: (N, C_emg, 2000)
Ys: (N, 2000, 18)
```

PyTorch `Conv1d` expects the input layout:

```text
(batch, channels, time)
```

The target remains:

```text
(batch, time, glove_channels)
```

## Preprocessing

Each EMG window is standardised independently across time:

```python
mu = Xs.mean(axis=2, keepdims=True)
sd = Xs.std(axis=2, keepdims=True)
Xs = (Xs - mu) / (sd + 1e-8)
```

The glove targets are normalised using statistics calculated from the training set:

```python
y_mu = y_train.mean(axis=0, keepdims=True)
y_sd = y_train.std(axis=0, keepdims=True)
```

The same training statistics are then applied to the validation and test sets.

## Train, validation, and test split

The data is divided approximately as follows:

```text
Training:   70%
Validation: 15%
Test:       15%
```

A small gap is placed between consecutive splits to reduce overlap leakage caused by neighbouring windows.

## Model output

The TCN produces a sequence of latent representations:

```text
tokens: (batch, time_steps, embedding_dimension)
```

For example:

```text
tokens: (64, 2000, 128)
```

The regression head is applied to every time step:

```python
pred_seq = model.reg_head(tokens)
```

Expected prediction shape:

```text
pred_seq: (64, 2000, 18)
```

This must match the target shape:

```text
yb: (64, 2000, 18)
```

The output size must therefore be set using the glove-channel dimension:

```python
output_size = y_train.shape[2]
```

Do not use:

```python
output_size = y_train.shape[1]
```

because `shape[1]` is the sequence length rather than the number of glove channels.

## Training

Run the project from its root directory:

```bash
python train_emg.py
```

During startup, the script prints the loaded file count and checks the model and target shapes.

Expected output resembles:

```text
Loaded subject_01.mat: 120 windows
Loaded subject_02.mat: 135 windows
Loaded 2 files and 255 total windows
tokens torch.Size([64, 2000, 128])
pred_seq torch.Size([64, 2000, 18])
yb torch.Size([64, 2000, 18])
```

Training then reports the mean squared error for each epoch:

```text
Epoch 01 | Train MSE ... | Val MSE ...
Epoch 02 | Train MSE ... | Val MSE ...
```

At the end, the script reports test MSE and plots:

- training and validation loss
- predicted and true values for one glove channel

## Loss function and optimiser

The model uses mean squared error:

```python
loss_fn = torch.nn.MSELoss()
```

The optimiser is AdamW:

```python
torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)
```

A `ReduceLROnPlateau` scheduler lowers the learning rate when validation loss stops improving.

Gradient clipping is also used:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=0.5
)
```

## Common errors

### `OSError: Invalid argument: 'datas/*.mat'`

`scipy.io.loadmat()` cannot directly open a wildcard pattern.

Incorrect:

```python
loadmat("datas/*.mat")
```

Use the directory loader instead:

```python
load_all_emg_glove_windows("datas")
```

### Prediction and target shapes do not match

For sequence-to-sequence training, both tensors must have shape:

```text
(batch, time_steps, glove_channels)
```

Set:

```python
output_size = y_train.shape[2]
```

and apply the regression head to all tokens:

```python
pred_seq = model.reg_head(tokens)
```

### No `.mat` files found

Check that the project contains:

```text
TCN/datas/
```

and that the `.mat` files are stored directly inside it.

Also run the script from the project root:

```bash
cd C:\Users\Jeevan\TCN
python train_emg.py
```

### Shape mismatch between files

All `.mat` files must use the same sampling rate and compatible EMG and glove channel dimensions.

The loader raises an error showing which file has an unexpected shape.

### `weight_norm` deprecation warning

PyTorch may display:

```text
FutureWarning: torch.nn.utils.weight_norm is deprecated
```

This is a warning rather than a training failure. The code can still run.

A future update can replace:

```python
from torch.nn.utils import weight_norm
```

with the parametrisation-based API:

```python
from torch.nn.utils.parametrizations import weight_norm
```

## Notes

Because neighbouring windows overlap, randomly shuffling all windows before splitting may create data leakage. A stronger evaluation approach is to split by recording or participant before creating training and test windows.

For large datasets, converting all NumPy arrays into tensors at once may consume substantial memory. A file-based or lazy-loading dataset can be introduced if memory becomes a limitation.
