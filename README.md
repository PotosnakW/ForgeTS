# TSFM — Time Series Foundation Model Pipeline

A PyTorch training pipeline for multivariate time series forecasting, built around **Forking Sequence (FCD) training** and designed to scale from a single GPU to distributed multi-node training.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Project Layout](#project-layout)
- [Data Config](#data-config)
- [Model Config](#model-config)
- [Dataloading](#dataloading)
- [Forking Sequences](#forking-sequences)
- [Sharded Datasets](#sharded-datasets)
- [Training](#training)
- [Distributed Training](#distributed-training)
- [Validation Strategies](#validation-strategies)
- [Loss Functions](#loss-functions)
- [Inference & Predictions](#inference--predictions)
- [Quick Start](#quick-start)

---

## Core Concepts

### Forking Sequence Training (FCD)

Rather than training on fixed windows, each training step samples a random anchor point per series and unfolds `fcd_samples` overlapping context-forecast pairs from it. This gives the model exposure to many temporal positions per step without loading more data.

```
anchor →  [───── context (L) ─────][── horizon (H) ──]   FCD 0
            step →  [───── context ─────][── horizon ──]   FCD 1
                step →  [───── context ─────][── horizon ──]   FCD 2
```

- **Training**: `fcd_samples > 1` — `heterogeneous_sampler` picks a random valid anchor per series
- **Val / Test**: `fcd_samples = -1` — full series, all valid FCD windows exhausted

### Heterogeneous Batching

Multiple datasets with different numbers of channels, series lengths, and missing data patterns can be combined in a single batch. The collation layer:
- **Left-pads** time to the longest series in the batch
- **Right-pads** channels to `C_max` with zeros
- **`available_mask [B, C, T]`** encodes both padding and mid-series gaps — zero means "don't use this element"

### Foundation Model Design

- Single model trained across many datasets simultaneously
- Weighted mixing controls dataset influence during training
- `outsample_mask` propagates through `fork_sequences` so loss is only computed over real, observed timesteps — padded channels and missing values are automatically excluded

---

## Project Layout

```
src/
├── dataloaders/
│   ├── _forking_sequences.py   # fork_sequences, heterogeneous_sampler, n_valid_fcds
│   ├── ts_dataloader.py        # FullSeriesDataset, DataLoaderFactory, HorizonBatchSampler
│   └── ts_sharding.py          # write_sharded_dataset, ShardedTrain/Val/TestDataset
└── common/
    ├── _base_model.py          # BaseModel — training loop, val loop, predict
    ├── train.py                # train(), train_distributed(), eval_test()
    ├── losses.py               # mse_loss, mae_loss, quantile_loss (all mask-aware)
    └── utils.py                # EarlyStopper
```

---

## Data Config

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

Multiple datasets are listed as separate entries under `train`, `validation`, and `test`. Each entry can have a different horizon, weight, and exogenous feature set.

### Splitting

`per_series_split: False` — global time split (default for most time series):
```
[──────── train ────────][── val ──][── test ──]
         T - val - test     val       test
```

`per_series_split: True` — each unique_id gets its own split boundary (for panel data where series have independent timelines).

---

## Model Config

```yaml
# Architecture
context_length: 512
input_size: 512

# Training loop
max_steps: 10000
val_check_interval: 500
early_stopping_patience: 10
fcd_samples: 4             # 1 = single window, >1 = forking sequences

# Loss
loss: "quantile"           # mse | mae | quantile
quantiles: [0.1, 0.5, 0.9]

# Optimiser
learning_rate: 1e-3
gradient_clip_val: 1.0

# Batching
batch_size: 32
valid_batch_size: 32
mixing_strategy: "concat"  # concat | round_robin
drop_last: False

# Validation strategy
val_strategy: "exhaustive"       # exhaustive | random_datasets | stratified
val_max_datasets: 4              # used by random_datasets only

# Misc
normalize: False
num_workers: 4
checkpoint_dir: "checkpoints/"
checkpoint_step: 1000
```

---

## Dataloading

### `DataLoaderFactory`

Central object that owns all dataset construction and dataloader creation.

```python
factory      = DataLoaderFactory(mcfg, dcfg)
train_loader = factory.train_dataloader()
val_loaders  = factory.val_dataloaders()   # {"val": DataLoader}
test_loaders = factory.test_dataloaders()  # {"test": DataLoader}
```

### `HorizonBatchSampler`

Groups datasets by horizon so all items in a batch share the same `H`. This is required for models that do autoregressive rollouts — each batch must have a consistent forecast length.

**Mixing strategies:**

| Strategy | Behaviour |
|---|---|
| `concat` | All datasets pooled together, sampled by weight |
| `round_robin` | Horizons interleaved — one batch per horizon per round |

**Weights** control how often each dataset appears relative to others. A dataset with `weight: 3.0` gets 3× more batches than one with `weight: 1.0`.

### `FullSeriesDataset`

Each dataset is a single item (`__len__ == 1`) — the entire series delivered to the model in one shot. `fork_sequences` handles all windowing inside the model's `_prepare_batch`. This means:

- No fixed window size baked into the dataset
- `context_length` can be changed at inference time without reloading data
- Heterogeneous series lengths are handled by left-padding at collation

---

## Forking Sequences

```python
from dataloaders._forking_sequences import fork_sequences, n_valid_fcds

# Training: sample fcd_samples windows per series
out = fork_sequences(batch, context_length=512, fcd_samples=4, horizon=6)

# Val/Test: all valid windows
out = fork_sequences(batch, context_length=512, fcd_samples=-1, horizon=6)
```

**Outputs:**

| Key | Shape | Description |
|---|---|---|
| `insample_y` | `[B, enc_size, C, 1+Vh]` | Encoder input (context + hist exog) |
| `outsample_y` | `[B, n_fcds, H, C]` | Forecast targets |
| `outsample_mask` | `[B, n_fcds, H, C]` | 1 = real, 0 = missing/padded |
| `available_mask` | `[B, C, enc_size]` | Encoder availability mask |

`outsample_mask` is derived directly from `available_mask` — any timestep or channel that is missing in the encoder context will also be masked in the targets. This flows into the loss functions automatically.

---

## Sharded Datasets

For datasets too large to fit in RAM, `write_sharded_dataset` partitions data into time-blocked parquet files with configurable shard size.

```python
from dataloaders.ts_sharding import write_sharded_dataset

write_sharded_dataset(
    df             = full_df,          # long-format DataFrame, must have 'available_mask'
    out_dir        = "data/sharded/simglucose",
    val_size       = 2592,             # matches dcfg entry val_size
    test_size      = 2592,             # matches dcfg entry test_size
    context_length = 512,              # L — baked into shard overlap
    shard_size     = 5_000,            # train timesteps per file
    hist_exog_cols = ["CHO", "insulin"],
)
```

**Disk layout:**
```
out_dir/
    shard_000000.parquet   # t=[0, shard_size+L)
    shard_000001.parquet   # t=[shard_size, 2*shard_size+L)   ← L-row overlap
    ...
    val.parquet            # t=[train_end-L, val_end)          ← context head from train
    test.parquet           # t=[val_end-L, T_total)            ← context head from val
    static.parquet         # per-channel static features
    metadata.json          # split boundaries, schema, shard index
```

The **L-row right-edge overlap** on each train shard means windows near shard boundaries are self-contained — no cross-file reads needed.

To use sharded data, add `sharded_dir` to the dcfg entry:

```python
entry = make_entry(
    path        = "data/simglucose.csv",   # still used for test fallback
    name        = "simglucose",
    sharded_dir = "data/sharded/simglucose",
)
```

---

## Training

### Single GPU

```python
from common.train import train, eval_test

factory      = DataLoaderFactory(mcfg, dcfg)
train_loader = factory.train_dataloader()
val_loaders  = factory.val_dataloaders()

metrics = train(model, mcfg, train_loader, val_loaders, device=torch.device("cuda"))

# Test inference
results = eval_test(model, factory, device=torch.device("cuda"))
```

### Resume from Checkpoint

```python
metrics = train(model, mcfg, train_loader, val_loaders,
                resume="checkpoints/final.pt")
```

### BaseModel Subclassing

```python
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = MLPDecoder(config)

    def forward(self, batch):
        # batch keys: insample_y, outsample_y, outsample_mask, available_mask
        x = batch["insample_y"]           # [B, enc_size, C, 1+Vh]
        return self.decoder(self.encoder(x))  # [B, n_fcds, H, C]

model = MyModel(config)
model.setup_training(mcfg, train_loader, val_loaders)
model.fit()
```

Only `forward` is required. `compute_loss`, `train_step`, `val_step`, and `predict_step` all have sensible defaults and can be overridden selectively.

---

## Distributed Training

Uses PyTorch DDP via `mp.spawn`. Sharded datasets are the recommended backend — each rank loads a non-overlapping subset of shard files.

```python
from common.train import train_distributed, eval_test

train_distributed(
    model      = model,
    mcfg       = mcfg,
    factory    = factory,       # built before spawn, rebuilt per rank inside
    backend    = "nccl",        # "gloo" for CPU
    world_size = 4,
    seed       = 42,
)

# Test always runs single-GPU after distributed context is destroyed
results = eval_test(model, factory, device=torch.device("cuda:0"))
```

**Shard assignment:** `files[rank::world_size]` — strided for balanced time coverage across ranks.

**Validation in distributed mode:** Every rank evaluates the full val set independently (same data, different model weights). Losses are all-reduced and averaged for the global metric. Only rank 0 logs and saves checkpoints.

---

## Validation Strategies

Controlled by `mcfg.val_strategy`:

| Strategy | Behaviour | Best for |
|---|---|---|
| `exhaustive` | Full pass over all val datasets every check | Final evaluation, small dataset counts |
| `random_datasets` | K random datasets per val check (`val_max_datasets`) | Many datasets, preventing overfitting |
| `stratified` | One random dataset per horizon group | Fast convergence signal across all horizons |

```yaml
val_strategy: "stratified"    # fast training feedback
# val_strategy: "random_datasets"
# val_max_datasets: 4
```

Test always uses exhaustive evaluation regardless of `val_strategy`.

---


## Inference & Predictions

```python
results = eval_test(model, factory, device=torch.device("cuda"))
```

Returns a nested dict keyed by dataset name, ready to pickle:

```python
{
    "simglucose": {
        "channel_ids":    ["patient_001", "patient_002", ...],  # unique_ids
        "preds":          Tensor,   # [n_fcds, H, C]
        "targets":        Tensor,   # [n_fcds, H, C]
        "outsample_mask": Tensor,   # [n_fcds, H, C]  1=real, 0=missing
    }
}

# Save
import pickle
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# Access per channel
for i, uid in enumerate(results["simglucose"]["channel_ids"]):
    preds_i  = results["simglucose"]["preds"][:, :, i]    # [n_fcds, H]
    mask_i   = results["simglucose"]["outsample_mask"][:, :, i]
```

Use `outsample_mask` when computing metrics to exclude missing ground truth:

```python
preds  = results["simglucose"]["preds"]           # [n_fcds, H, C]
targets = results["simglucose"]["targets"]
mask   = results["simglucose"]["outsample_mask"]

mae = (torch.abs(preds - targets) * mask).sum() / mask.sum()
```

---

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
metrics = train(model, mcfg, train_loader, val_loaders,
                device=torch.device("cuda"))

# 5. Predict
results = eval_test(model, factory, device=torch.device("cuda"))
```
