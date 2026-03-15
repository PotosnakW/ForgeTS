# TSFM &nbsp;·&nbsp; Time Series Foundation Model Pipeline

> A PyTorch training pipeline for multivariate time series forecasting, built around **Forking Sequence (FCD) training** and designed to scale from a single GPU to distributed multi-node training.

<br>

---

<br>

## Table of Contents

| | Section |
|---|---|
| 1  | [Core Concepts](#core-concepts)                   |
| 2  | [Quick Start](#quick-start)                       |
| 3  | [Dataloaders](#dataloaders)                       |
| 4  | [Forking Sequences](#forking-sequences)           |
| 5  | [Training](#training)                             |
| 6  | [Loss Functions](#loss-functions)                 |
| 7  | [Validation Strategies](#validation-strategies)   |
| 8  | [Distributed Training](#distributed-training)     |
| 9  | [Data Sharding](#data-sharding)                   |
| 10 | [Inference & Predictions](#inference--predictions)|
| 11 | [License](#license)                               |

<br>

---

<br>

## Core Concepts

<br>

### Forking Sequence Training (FCD)

Rather than training on fixed windows, each training step samples a random anchor point per series and unfolds `fcd_samples` overlapping context-forecast pairs from it. This gives the model exposure to many temporal positions per step without loading more data.

```
anchor →  [───── context (L) ─────][── horizon (H) ──]   FCD 0
            step →  [───── context ─────][── horizon ──]   FCD 1
                step →  [───── context ─────][── horizon ──]   FCD 2
```

| Mode | `fcd_samples` | Behaviour |
|---|---|---|
| Training | `> 1` | `heterogeneous_sampler` picks a random valid anchor per series |
| Val / Test | `-1` | Full series — all valid FCD windows exhausted |

<br>

### Heterogeneous Batching

Multiple datasets with different numbers of channels, series lengths, and missing data patterns can be combined in a single batch. The collation layer:

- **Left-pads** time to the longest series in the batch
- **Right-pads** channels to `C_max` with zeros
- **`available_mask [B, C, T]`** encodes both padding and mid-series gaps — zero means "don't use this element"

<br>

### Foundation Model Design

- Single model trained across many datasets simultaneously
- Weighted mixing controls dataset influence during training
- `outsample_mask` propagates through `fork_sequences` so loss is only computed over real, observed timesteps — padded channels and missing values are automatically excluded

<br>

---

<br>

## Quick Start

```python
import torch
from dataloaders.ts_dataloader import DataLoaderFactory
from common.train import train, eval_test
from common._base_model import BaseModel
from types import SimpleNamespace

# 1. Configs
mcfg = SimpleNamespace(
    context_length=512, fcd_samples=4, batch_size=32,
    max_steps=10000, val_check_interval=500,
    loss="mse", mixing_strategy="concat",
    val_strategy="stratified",
    normalize=False, num_workers=4,
    checkpoint_dir="checkpoints/", checkpoint_step=1000,
    learning_rate=1e-3, gradient_clip_val=1.0,
    early_stopping_patience=10, drop_last=False,
    valid_batch_size=32,
)

dcfg = SimpleNamespace(
    train=[SimpleNamespace(
        path="data/simglucose.csv", name="simglucose",
        horizon=6, val_size=2592, test_size=2592,
        weight=1.0, hist_exog_cols=["CHO", "insulin"],
        futr_exog_cols=[], stat_exog_cols=[],
        per_series_split=False, sharded_dir=None,
    )],
)
dcfg.validation = dcfg.train
dcfg.test       = dcfg.train

# 2. Data
factory      = DataLoaderFactory(mcfg, dcfg)
train_loader = factory.train_dataloader()
val_loaders  = factory.val_dataloaders()

# 3. Model
model = MyModel(mcfg)

# 4. Train
metrics = train(model, mcfg, train_loader, val_loaders, device=torch.device("cuda"))

# 5. Predict
results = eval_test(model, factory, device=torch.device("cuda"))
```

<br>

---

<br>

## Dataloaders

<br>

### Dataset Config

Dataset configs define paths, train/val/test splits, exogenous features, and per-dataset sampling weights. Multiple datasets can be listed as separate entries under `train`, `validation`, and `test`.

```yaml
train:
  - path: "../datasets/simglucose_90_days.csv"
    name: "simglucose"
    horizon: 6
    val_size: 2592        # timesteps reserved for validation
    test_size: 2592       # timesteps reserved for test
    weight: 1.0           # relative sampling weight during training
    hist_exog_cols: [CHO, insulin]
    per_series_split: False
```

<br>

**Splitting modes:**

`per_series_split: False` — global time split (default):
```
[──────────── train ────────────][──── val ────][──── test ────]
            T - val - test              val             test
```

`per_series_split: True` — each `unique_id` gets its own split boundary, for panel data where series have independent timelines.

<br>

### DataLoaderFactory

Central object that owns all dataset construction and dataloader creation.

```python
factory      = DataLoaderFactory(mcfg, dcfg)
train_loader = factory.train_dataloader()
val_loaders  = factory.val_dataloaders()   # {"val": DataLoader}
test_loaders = factory.test_dataloaders()  # {"test": DataLoader}
```

<br>

### FullSeriesDataset

Each dataset is a single item (`__len__ == 1`) — the entire series delivered to the model in one shot. `fork_sequences` handles all windowing inside `_prepare_batch`. This means:

- No fixed window size baked into the dataset
- `context_length` can be changed at inference time without reloading data
- Heterogeneous series lengths are handled by left-padding at collation

<br>

### HorizonBatchSampler

Groups datasets by horizon so all items in a batch share the same `H`. Required for autoregressive rollouts — each batch must have a consistent forecast length.

| Strategy | Behaviour |
|---|---|
| `concat` | All datasets pooled together, sampled by weight |
| `round_robin` | Horizons interleaved — one batch per horizon per round |

> Dataset `weight` controls relative sampling frequency. A dataset with `weight: 3.0` gets 3× more batches than one with `weight: 1.0`.

<br>

---

<br>

## Forking Sequences

```python
from dataloaders._forking_sequences import fork_sequences, n_valid_fcds

# Training — sample fcd_samples windows per series
out = fork_sequences(batch, context_length=512, fcd_samples=4, horizon=6)

# Val / Test — all valid windows, no sampling
out = fork_sequences(batch, context_length=512, fcd_samples=-1, horizon=6)
```

<br>

### Outputs

| Key | Shape | Description |
|---|---|---|
| `insample_y` | `[B, enc_size, C, 1+Vh]` | Encoder input — context window + historical exogenous |
| `outsample_y` | `[B, n_fcds, H, C]` | Forecast targets |
| `outsample_mask` | `[B, n_fcds, H, C]` | `1` = real, `0` = missing / padded |
| `available_mask` | `[B, C, enc_size]` | Encoder availability mask |

`outsample_mask` is derived directly from `available_mask` — any timestep or channel missing in the encoder context is also masked in the targets, flowing into loss functions automatically.

<br>

### FCD Count

In both training and eval modes the number of complete windows produced is:

```
n_fcds = floor((enc_block_len - L - H) / step) + 1
```

Incomplete trailing windows are always dropped — the last FCD horizon never extends beyond available data.

<br>

### heterogeneous_sampler

Called during training (`fcd_samples != -1`) to pick one `window_start` per series. A timestep is only valid if **all channels** have real data there — this naturally skips left-padding and mid-series gaps. Sampling is via `torch.multinomial` so each series gets an independent draw.

<br>

---

<br>

## Training

<br>

### Model Config

```yaml
# ── Architecture ──────────────────────────────
context_length: 512
input_size: 512

# ── Training loop ─────────────────────────────
max_steps: 10000
val_check_interval: 500
early_stopping_patience: 10
fcd_samples: 4             # 1 = single window, >1 = forking sequences

# ── Loss ──────────────────────────────────────
loss: "quantile"           # mse | mae | quantile
quantiles: [0.1, 0.5, 0.9]

# ── Optimiser ─────────────────────────────────
learning_rate: 1e-3
gradient_clip_val: 1.0

# ── Batching ──────────────────────────────────
batch_size: 32
valid_batch_size: 32
mixing_strategy: "concat"  # concat | round_robin
drop_last: False

# ── Validation ────────────────────────────────
val_strategy: "exhaustive"       # exhaustive | random_datasets | stratified
val_max_datasets: 4              # used by random_datasets only

# ── Misc ──────────────────────────────────────
normalize: False
num_workers: 4
checkpoint_dir: "checkpoints/"
checkpoint_step: 1000
```

<br>

### Single GPU

```python
from common.train import train, eval_test

metrics = train(model, mcfg, train_loader, val_loaders, device=torch.device("cuda"))
results = eval_test(model, factory, device=torch.device("cuda"))
```

<br>

### Resuming from Checkpoint

```python
metrics = train(model, mcfg, train_loader, val_loaders, resume="checkpoints/final.pt")
```

<br>

### BaseModel Subclassing

Only `forward` is required. `compute_loss`, `train_step`, `val_step`, and `predict_step` all have sensible defaults and can be overridden selectively.

```python
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = MLPDecoder(config)

    def forward(self, batch):
        # batch keys: insample_y, outsample_y, outsample_mask, available_mask
        x = batch["insample_y"]               # [B, enc_size, C, 1+Vh]
        return self.decoder(self.encoder(x))  # [B, n_fcds, H, C]

model = MyModel(config)
model.setup_training(mcfg, train_loader, val_loaders)
model.fit()
```

<br>

---

<br>

## Loss Functions

> Adapted from [datasetsforecast](https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py)

All losses share a common masked-reduction contract — padded channels and
missing timesteps are excluded from both the numerator and denominator.
`outsample_mask` from `fork_sequences` flows directly into every loss call
without any additional preprocessing.

<br>

---

<br>

### Training Losses (PyTorch)

Select the loss via `mcfg.loss`:

| `mcfg.loss` | Function | Extra config |
|---|---|---|
| `"mae"` | `mae_loss` | — |
| `"mse"` | `mse_loss` | — |
| `"huber"` | `huber_loss` | `delta: 1.0` |
| `"quantile"` | `quantile_loss` | `quantiles: [0.1, 0.5, 0.9]` |

All point-forecast losses share the same signature:
```python
loss_fn(
    preds:   torch.Tensor,         # [B, H, C]
    targets: torch.Tensor,         # [B, H, C]
    mask:    torch.Tensor | None,  # [B, H, C]  1=real, 0=padded/missing
) -> torch.Tensor                  # scalar
```

Masked reduction applied by every loss:
```python
(raw_loss * mask).sum() / mask.sum().clamp(min=1)
```

When `mask is None` a plain `.mean()` is used — equivalent to a mask of all ones.

<br>

#### MAE

Mean absolute error. Constant-magnitude gradients make it robust to large
outliers but the non-differentiability at zero can slow convergence near
the optimum.
```
L = mean( |y − ŷ| )
```
```yaml
loss: "mae"
```

<br>

#### MSE

Mean squared error. Quadratic penalty makes large errors dominate the
gradient signal — useful when big misses should be prioritised, but
sensitive to outliers.
```
L = mean( (y − ŷ)² )
```
```yaml
loss: "mse"
```

<br>

#### Huber

Quadratic (MSE-like) when the absolute error is below `delta`, linear
(MAE-like) above it. Smooth at zero, robust in the tails.
```
         ½ (y − ŷ)²                   if |y − ŷ| < δ
L_δ  =
         δ · (|y − ŷ| − ½δ)           otherwise
```

`delta` controls the crossover point — set it to the typical scale of your
residuals. Smaller values increase robustness; larger values approach MSE.
```yaml
loss: "huber"
delta: 1.0      # default
```

<br>

#### Quantile (Pinball)

Trains the model to predict Q quantile levels simultaneously. The model
output expands to `[B, H, C × Q]`; the loss function handles the reshape
internally.
```
L_q = mean over q of  max( q·(y − ŷ_q),  (q−1)·(y − ŷ_q) )
```

The median quantile `q = 0.5` is equivalent to MAE up to a constant factor.
Interval width is calibrated by the gap between symmetric quantile pairs
(e.g. `0.1` / `0.9`).
```yaml
loss: "quantile"
quantiles: [0.1, 0.5, 0.9]
```
```python
# Quantile loss signature differs — preds carries C × Q channels
quantile_loss(
    preds:     torch.Tensor,   # [B, H, C × Q]
    targets:   torch.Tensor,   # [B, H, C]
    quantiles: list[float],
    mask:      torch.Tensor | None,  # [B, H, C]
) -> torch.Tensor
```

<br>

---

<br>

### Evaluation Losses (NumPy)

Used in `eval_test` and any offline metric computation. Inputs are plain
NumPy arrays; the mask is an optional boolean or `{0,1}` integer array with
the same shape as `preds`.
```python
from common.losses_np import mae, mse, rmse, mape, smape
```

| Function | Formula |
|---|---|
| `mae` | `mean( \|y − ŷ\| )` |
| `mse` | `mean( (y − ŷ)² )` |
| `rmse` | `sqrt( mse )` |
| `mape` | `mean( \|y − ŷ\| / \|y\| ) × 100` |
| `smape` | `mean( 2\|y − ŷ\| / (\|y\| + \|ŷ\|) ) × 100` |

All functions share the same array contract:
```python
metric_fn(
    preds:   np.ndarray,            # [..., H, C]
    targets: np.ndarray,            # [..., H, C]
    mask:    np.ndarray | None,     # [..., H, C]  1=real, 0=missing
) -> float
```

<br>

**Typical usage with `eval_test` results:**
```python
from common.losses_np import mae, rmse

results = eval_test(model, factory, device=torch.device("cuda"))

preds   = results["simglucose"]["preds"].numpy()           # [n_fcds, H, C]
targets = results["simglucose"]["targets"].numpy()         # [n_fcds, H, C]
mask    = results["simglucose"]["outsample_mask"].numpy()  # [n_fcds, H, C]

print("MAE: ", mae(preds, targets, mask))
print("RMSE:", rmse(preds, targets, mask))
```

Example use cases:
```python
preds   = results["simglucose"]["preds"].numpy()           # [n_fcds, H, C]
targets = results["simglucose"]["targets"].numpy()         # [n_fcds, H, C]
mask    = results["simglucose"]["outsample_mask"].numpy()  # [n_fcds, H, C]
print("MAE: ", mae(preds, targets, mask))
print("RMSE:", rmse(preds, targets, mask))
```

> **Quantile models** — pass only the median slice to point-forecast metrics:
> ```python
> Q   = len(mcfg.quantiles)
> mid = mcfg.quantiles.index(0.5)          # index of q=0.5
> # preds shape is [n_fcds, H, C, Q] after reshaping
> p_median = preds.reshape(*preds.shape[:-1], -1, Q)[..., mid]
> print("Median MAE:", mae(p_median, targets, mask))
> ```

<br>

---

## Validation Strategies

Controlled by `mcfg.val_strategy`:

| Strategy | Behaviour | Best for |
|---|---|---|
| `exhaustive` | Full pass over all val datasets every check | Final evaluation, small dataset counts |
| `random_datasets` | K random datasets per check (`val_max_datasets`) | Many datasets, preventing val overfitting |
| `stratified` | One random dataset per horizon group | Fast convergence signal across all horizons |

> Test always uses `exhaustive` evaluation regardless of `val_strategy`.

<br>

---

<br>

## Distributed Training

Uses PyTorch DDP via `torchrun` or `mp.spawn`. Sharded datasets are the recommended backend — each rank loads a non-overlapping subset of shard files.

<br>

### Launch

**torchrun** (recommended):
```bash
torchrun --nproc_per_node=4 train_script.py
```

**mp.spawn** (programmatic, single-machine):
```python
from common.train import train_distributed

train_distributed(model, mcfg, factory, use_spawn=True, world_size=4)
results = eval_test(model, factory)
```

<br>

### Data Strategy

| Phase | Behaviour |
|---|---|
| **Training** | Each rank receives a non-overlapping contiguous slice of every horizon group's pool. Padding ensures identical batch counts across ranks — required by DDP's barrier synchronisation. |
| **Validation** | Every rank evaluates the full val set independently (same data, different model weights). Losses are all-reduced and averaged for the global metric. Only rank 0 logs and saves checkpoints. |
| **Test** | Always single GPU, always after `dist.destroy_process_group()`. Call `eval_test()` outside of any distributed context. |

<br>

### Shard Assignment

```python
files[rank::world_size]  # strided for balanced time coverage across ranks
```

<br>

---

<br>

## Data Sharding

For datasets too large to fit in RAM, `write_sharded_dataset` partitions data into time-blocked parquet files with configurable shard size.

<br>

### Writing Shards

```python
from dataloaders.ts_sharding import write_sharded_dataset

write_sharded_dataset(
    df             = full_df,           # long-format DataFrame — must have 'available_mask'
    out_dir        = "data/sharded/simglucose",
    val_size       = 2592,              # matches dcfg entry val_size
    test_size      = 2592,              # matches dcfg entry test_size
    context_length = 512,               # L — baked into shard overlap
    shard_size     = 5_000,             # train timesteps per file
    hist_exog_cols = ["CHO", "insulin"],
)
```

<br>

### Disk Layout

```
out_dir/
    shard_000000.parquet   # t = [0,          shard_size + L)
    shard_000001.parquet   # t = [shard_size,  2·shard_size + L)   ← L-row overlap
    ...
    val.parquet            # t = [train_end - L,  val_end)
    test.parquet           # t = [val_end - L,    T_total)
    static.parquet         # per-channel static features
    metadata.json          # split boundaries, schema, shard index
```

The **L-row right-edge overlap** on each train shard means windows near shard boundaries are self-contained — no cross-file reads needed.

<br>

### Using Sharded Data

Add `sharded_dir` to your dcfg entry:

```python
entry = make_entry(
    path        = "data/simglucose.csv",
    name        = "simglucose",
    sharded_dir = "data/sharded/simglucose",
)
```

The factory automatically uses `ShardedTrainDataset`, `ShardedValDataset`, and `ShardedTestDataset` when `sharded_dir` is present.

<br>

---

<br>

## Inference & Predictions

```python
results = eval_test(model, factory, device=torch.device("cuda"))
```

Returns a nested dict keyed by dataset name:

```python
{
    "simglucose": {
        "channel_ids":    ["patient_001", "patient_002", ...],
        "preds":          Tensor,   # [n_fcds, H, C]
        "targets":        Tensor,   # [n_fcds, H, C]
        "outsample_mask": Tensor,   # [n_fcds, H, C]   1 = real, 0 = missing
    }
}
```

<br>

Use `outsample_mask` when computing metrics to exclude missing ground truth:

```python
preds   = results["simglucose"]["preds"]
targets = results["simglucose"]["targets"]
mask    = results["simglucose"]["outsample_mask"]

mae = (torch.abs(preds - targets) * mask).sum() / mask.sum()
```

Per-channel access:

```python
for i, uid in enumerate(results["simglucose"]["channel_ids"]):
    preds_i = results["simglucose"]["preds"][:, :, i]          # [n_fcds, H]
    mask_i  = results["simglucose"]["outsample_mask"][:, :, i]
```

<br>

---

<br>


## License

MIT License

Copyright (c) 2024 Auton Lab, Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/mononitogoswami/labelerrors/blob/main/LICENSE) for details.