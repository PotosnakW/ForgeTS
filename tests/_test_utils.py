import pandas as pd
import numpy as np
from types import SimpleNamespace
import os, shutil


def make_df(n_series=3, n_steps=500, n_hist=1, seed=42):
    rng   = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=n_steps, freq="5min")
    rows  = []
    for s in range(n_series):
        y    = rng.standard_normal(n_steps).astype(np.float32)
        hist = {f"hist_{i}": rng.standard_normal(n_steps).astype(np.float32)
                for i in range(n_hist)}
        rows.append(pd.DataFrame({
            "unique_id": f"series_{s}", "ds": times, "y": y,
            "available_mask": np.ones(n_steps, dtype=np.float32), **hist,
        }))
    return pd.concat(rows, ignore_index=True)
 
 
def make_mcfg(
        model_name='linear', 
        encoder="google/t5-efficient-tiny",
        h=6, 
        context_length=64,
        batch_size=2, 
        max_steps=10, 
        fcd_samples=4, 
        mixing_strategy="concat", 
        checkpoint_dir=None
    ):

    if model_name == 'linear':
        return SimpleNamespace(
            h                       = h,
            context_length          = context_length,
            n_channels              = 3,
            fcd_samples             = fcd_samples,
            batch_size              = batch_size,
            valid_batch_size        = batch_size,
            max_steps               = max_steps,
            val_check_interval      = 10,
            learning_rate           = 1e-3,
            gradient_clip_val       = 1.0,
            early_stopping_patience = 9999,
            monitor_metric          = "loss",
            monitor_mode            = "min",
            mixing_strategy         = mixing_strategy,
            val_strategy            = "exhaustive",
            drop_last               = False,
            batch_mode              = "full_series",
            checkpoint_dir          = checkpoint_dir,
            checkpoint_step         = 99999,
            num_workers             = 0,
            loss                    = "mae",
            seed                    = 42,
            stat_exog_cols          = [],
        )
    elif model_name == 'mica':
        return SimpleNamespace(
                context_length         = context_length,
                patch_len              = 8,
                stride                 = 1,
                h                      = h,
                pe_type                = 'sincos',
                learn_pe               = False,
                dropout                = 0.0,
                encoder                = encoder,
                output_layer           = "linear_proj",
                infini_mixer_type      = "none",
                infini_channel_exclusion = False,
                layerwise_beta         = False,
                channelwise_beta       = False,
                mlpmixer_hidden_size   = 128,
                mlpmixer_n_layers      = 3,
                mlpmixer_dropout       = 0.0,
                multivariate_head      = False,
                head_dropout           = 0.0,
                hidden_size            = 256,
                linear_hidden_size     = 1024,
                n_heads                = 4,
                n_layers               = 4,
                learning_rate          = 1e-3,
                fcd_samples            = fcd_samples,
                batch_size             = batch_size,
                valid_batch_size       = batch_size,
                max_steps              = max_steps,
                val_check_interval     = 10,
                gradient_clip_val      = 1.0,
                early_stopping_patience= 9999,  # disable for tests
                monitor_metric         = "loss",
                monitor_mode           = "min",
                mixing_strategy        = mixing_strategy,
                drop_last              = False,
                batch_mode             = "full_series",
                checkpoint_dir         = checkpoint_dir,
                checkpoint_step        = 99999,  # suppress mid-run saves
                num_workers            = 0,
                revin                  = True,
                revin_affine           = False,
                revin_subtract_last    = False,
                res_attention          = True,
                d_k                    = 32,
                d_v                    = 32,
                qkv_bias               = False,
                attn_dropout           = False,
                proj_dropout           = 0.0,
                norm                   = "BatchNorm",
                activation             = 'gelu',
                pre_norm               = False,
                store_attn             = False,
                loss                   = "mae",
            )

 
def make_entry(path, name="ds", horizon=6, val_size=50, test_size=50,
               weight=1.0, hist_exog_cols=None, futr_exog_cols=None,
               stat_exog_cols=None, per_series_split=False,
               use_context_head=True, sharded_dir=None,
               multivariate=False):          # <-- added
    return SimpleNamespace(
        path             = path,
        name             = name,
        horizon          = horizon,
        val_size         = val_size,
        test_size        = test_size,
        weight           = weight,
        hist_exog_cols   = hist_exog_cols or [],
        futr_exog_cols   = futr_exog_cols or [],
        stat_exog_cols   = stat_exog_cols or [],
        per_series_split = per_series_split,
        use_context_head = use_context_head,
        sharded_dir      = sharded_dir,
        multivariate     = multivariate,     # <-- added
    )
 
 
def make_dcfg(train_entries, val_entries=None, test_entries=None):
    return SimpleNamespace(
        train      = train_entries,
        validation = val_entries  or train_entries,
        test       = test_entries or train_entries,
    )

def setup_test_data(TEST_DATA_DIR):
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

def remove_test_data(TEST_DATA_DIR):
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)