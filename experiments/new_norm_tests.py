"""
Test suite: Causal Norm with ForkingSequences + Scaler

Tests:
1. Index alignment — FCD indices point to correct positions in original series
2. No temporal leakage — stat at position t uses ONLY data from 0..t, never t+1..S
3. Heterogeneous vs homogeneous sampling — per-batch offsets handled correctly
4. Scaler norm path — causal_stats applied correctly to insample_y
5. Scaler denorm path — causal_fcd_stats applied correctly to preds
6. Scaler norm_targets path — causal_fcd_stats applied correctly to outsample_y
7. Round-trip — norm then denorm recovers original scale
8. Stride > 1 — indices still correct with non-unit stride
"""

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════
# Minimal reimplementations (self-contained, no imports from codebase)
# ═══════════════════════════════════════════════════════════════════

def _unfold_windows(src, size, step):
    unfolded = src.unfold(dimension=1, size=size, step=step)
    if unfolded.ndim == 3:
        return unfolded.contiguous()
    ndim = unfolded.ndim
    order = [0, 1, ndim - 1] + list(range(2, ndim - 1))
    return unfolded.permute(*order).contiguous()


def _gather_block(src, window_start, block_len, T):
    B = src.shape[0]
    extra = src.shape[2:]
    offsets = torch.arange(block_len, device=src.device)
    grid = (window_start.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, T - 1)
    return src.gather(1, grid.unsqueeze(-1).unsqueeze(-1).expand(B, block_len, *extra))


def _gather_mask(mask, window_start, block_len, T):
    B, _, C = mask.shape
    grid = (window_start.unsqueeze(1) + torch.arange(block_len, device=mask.device).unsqueeze(0)).clamp(0, T - 1)
    grid = grid.unsqueeze(-1).expand(B, block_len, C)
    return mask.gather(1, grid)


def forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start):
    """Simplified ForkingSequences.__call__ with explicit window_start."""
    x_enc_full = batch["x_enc"]
    available_mask = batch["available_mask"]
    loss_mask = batch["loss_mask"]
    B, S, C, X1 = x_enc_full.shape

    window_size = context_len + horizon
    block_len = context_len + (fcd_samples - 1) * stride + horizon

    enc_block = _gather_block(x_enc_full, window_start, block_len, S)
    mask_block = _gather_mask(available_mask, window_start, block_len, S)
    loss_mask_block = _gather_mask(loss_mask, window_start, block_len, S)

    enc_windows = _unfold_windows(enc_block, size=window_size, step=stride)
    loss_mask_windows = _unfold_windows(loss_mask_block, size=window_size, step=stride)

    eff_L = window_size - horizon
    valid_fcds = fcd_samples
    outsample_mask = loss_mask_windows[:, :, eff_L:, :]
    enc_size = enc_block.shape[1] - horizon

    out = dict(
        insample_y=enc_block[:, :enc_size],
        outsample_y=enc_windows[:, :, eff_L:, :, 0],
        outsample_mask=outsample_mask,
        available_mask=mask_block[:, :enc_size],
        channel_mask=batch['channel_mask'],
        fcd_samples=valid_fcds,
        horizon=horizon,
    )

    # Causal stats on FULL original series
    mask_full = available_mask.unsqueeze(-1).expand_as(x_enc_full)
    counts = torch.cumsum(mask_full, dim=1).clamp(min=1)
    mean = torch.cumsum(x_enc_full * mask_full, dim=1) / counts
    mean_sq = torch.cumsum((x_enc_full ** 2) * mask_full, dim=1) / counts
    stdev = torch.sqrt((mean_sq - mean ** 2).clamp(min=0) + 1e-5)

    # Per-timestep stats for norm
    ts_offsets = torch.arange(enc_size, device=x_enc_full.device)
    if window_start is not None:
        ts_indices = window_start.unsqueeze(1) + ts_offsets.unsqueeze(0)
    else:
        ts_indices = ts_offsets.unsqueeze(0).expand(B, -1)
    ts_idx = ts_indices.unsqueeze(-1).unsqueeze(-1).expand(B, enc_size, C, X1)
    out["causal_stats"] = {
        'mean': mean.gather(1, ts_idx),
        'stdev': stdev.gather(1, ts_idx),
    }

    # Per-FCD stats for denorm/norm_targets
    fcd_offsets = torch.arange(valid_fcds, device=x_enc_full.device) * stride + eff_L - 1
    if window_start is not None:
        fcd_indices = window_start.unsqueeze(1) + fcd_offsets.unsqueeze(0)
    else:
        fcd_indices = fcd_offsets.unsqueeze(0).expand(B, -1)
    fcd_idx = fcd_indices.unsqueeze(-1).unsqueeze(-1).expand(B, valid_fcds, C, X1)
    out["causal_fcd_stats"] = {
        'mean': mean.gather(1, fcd_idx),
        'stdev': stdev.gather(1, fcd_idx),
    }

    # Debug info
    out["_full_mean"] = mean
    out["_full_stdev"] = stdev
    out["_ts_indices"] = ts_indices
    out["_fcd_indices"] = fcd_indices

    return out


def _standardize(x, stats, stride, norm_type, **kwargs):
    mean = stats['mean']
    stdev = stats['stdev']
    affine_weight = kwargs['affine_weight']
    affine_bias = kwargs['affine_bias']
    eps = kwargs['eps']
    causal_fcd_stats = kwargs.get('causal_fcd_stats', None)

    if norm_type == 'norm':
        x = (x - mean) / stdev
        if affine_weight is not None:
            x = x * affine_weight + affine_bias
        return x

    elif norm_type == 'denorm':
        if causal_fcd_stats is not None:
            fcd_mean = causal_fcd_stats['mean'][:, :, :, 0:1].unsqueeze(2)
            fcd_stdev = causal_fcd_stats['stdev'][:, :, :, 0:1].unsqueeze(2)
        else:
            T = x.shape[1]
            fcd_mean = mean[:, -T*stride::stride, :, 0:1].unsqueeze(2)
            fcd_stdev = stdev[:, -T*stride::stride, :, 0:1].unsqueeze(2)
        if affine_weight is not None:
            x = (x - affine_bias) / (affine_weight + eps ** 2)
        return x * fcd_stdev + fcd_mean

    elif norm_type == 'norm_targets':
        if causal_fcd_stats is not None:
            fcd_mean = causal_fcd_stats['mean'][:, :, :, 0].unsqueeze(2).expand_as(x)
            fcd_stdev = causal_fcd_stats['stdev'][:, :, :, 0].unsqueeze(2).expand_as(x)
        else:
            T = x.shape[1]
            fcd_mean = mean[:, -T*stride::stride, :, 0].unsqueeze(2).expand_as(x)
            fcd_stdev = stdev[:, -T*stride::stride, :, 0].unsqueeze(2).expand_as(x)
        x = (x - fcd_mean) / fcd_stdev
        if affine_weight is not None:
            x = x * affine_weight + affine_bias
        return x


class Scaler(nn.Module):
    def __init__(self, scaler_type, stride, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scaler_type = scaler_type
        self.stride = stride
        if scaler_type == 'standard':
            self.affine_weight = None
            self.affine_bias = None
            self.scaler = _standardize

    def forward(self, batch, norm_type):
        causal_stats = batch.get("causal_stats", None)
        causal_fcd_stats = batch.get("causal_fcd_stats", None)

        if norm_type == 'norm':
            x = batch["insample_y"].clone()
            if causal_stats is not None:
                self.stats = causal_stats
            else:
                raise RuntimeError("This test expects causal_stats in batch")

        elif norm_type == 'denorm':
            x = batch["preds"].clone()

        elif norm_type == 'norm_targets':
            x = batch["outsample_y"].clone()

        x_scaled = self.scaler(
            x=x,
            stats=self.stats,
            stride=self.stride,
            norm_type=norm_type,
            affine_weight=self.affine_weight,
            affine_bias=self.affine_bias,
            eps=self.eps,
            causal_fcd_stats=causal_fcd_stats if norm_type != 'norm' else None,
        )

        if norm_type == 'norm':
            batch["insample_y"] = x_scaled
        elif norm_type == 'denorm':
            batch["preds"] = x_scaled
        elif norm_type == 'norm_targets':
            batch["outsample_y"] = x_scaled

        return batch


# ═══════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════

def make_batch(B, S, C, X1=1):
    """Deterministic series: value at position t = t + b*1000 for batch b."""
    x_enc = torch.arange(S, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    x_enc = x_enc.expand(B, S, C, X1).clone()
    for b in range(B):
        x_enc[b] += b * 1000
    return {
        "x_enc": x_enc,
        "available_mask": torch.ones(B, S, C),
        "loss_mask": torch.ones(B, S, C),
        "channel_mask": torch.ones(B, C),
    }

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════
# TEST 1: No temporal leakage
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 1: No temporal leakage")
print("=" * 70)

B, S, C = 2, 50, 1
context_len, horizon, stride, fcd_samples = 8, 4, 1, 5
batch = make_batch(B, S, C)
window_start = torch.tensor([10, 20])

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)

# For each FCD, verify the stat uses ONLY data from 0..fcd_idx, not beyond
x_full = batch["x_enc"]
fcd_indices = out["_fcd_indices"]  # [B, n_fcds]

for b in range(B):
    for i in range(fcd_samples):
        fcd_idx = fcd_indices[b, i].item()
        # Manual causal mean: average of series[0..fcd_idx]
        manual_mean = x_full[b, :fcd_idx + 1, 0, 0].mean().item()
        gathered_mean = out["causal_fcd_stats"]["mean"][b, i, 0, 0].item()
        check(
            f"batch={b}, fcd={i}: stat at idx={fcd_idx} uses only data [0..{fcd_idx}]",
            abs(manual_mean - gathered_mean) < 1e-4
        )

        # Verify stat would be DIFFERENT if it included future data
        if fcd_idx + 1 < S:
            mean_with_future = x_full[b, :fcd_idx + 2, 0, 0].mean().item()
            check(
                f"batch={b}, fcd={i}: stat DIFFERS from [0..{fcd_idx+1}] (no future leak)",
                abs(gathered_mean - mean_with_future) > 1e-6
            )

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Heterogeneous sampling — per-batch offsets
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 2: Heterogeneous sampling — different window_start per batch")
print("=" * 70)

B, S, C = 3, 60, 1
context_len, horizon, stride, fcd_samples = 10, 4, 1, 3
batch = make_batch(B, S, C)
window_start = torch.tensor([5, 15, 25])  # different per batch

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)
eff_L = context_len

fcd_indices = out["_fcd_indices"]
ts_indices = out["_ts_indices"]

for b in range(B):
    ws = window_start[b].item()

    # Check first timestep of insample_y maps to window_start[b] in original
    check(
        f"batch={b}: ts_indices[0] = {ts_indices[b, 0].item()} == window_start={ws}",
        ts_indices[b, 0].item() == ws
    )

    # Check FCD 0 maps to window_start[b] + eff_L - 1
    expected_fcd0 = ws + eff_L - 1
    check(
        f"batch={b}: fcd_indices[0] = {fcd_indices[b, 0].item()} == {expected_fcd0}",
        fcd_indices[b, 0].item() == expected_fcd0
    )

    # Check FCD i maps to window_start[b] + i*stride + eff_L - 1
    for i in range(fcd_samples):
        expected = ws + i * stride + eff_L - 1
        check(
            f"batch={b}, fcd={i}: fcd_indices={fcd_indices[b, i].item()} == {expected}",
            fcd_indices[b, i].item() == expected
        )

    # Verify different batches have different FCD indices
    if b > 0:
        check(
            f"batch {b} FCDs differ from batch {b-1}",
            not torch.equal(fcd_indices[b], fcd_indices[b-1])
        )

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Stride > 1
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 3: Stride > 1")
print("=" * 70)

B, S, C = 2, 80, 1
context_len, horizon, stride, fcd_samples = 12, 4, 3, 4
batch = make_batch(B, S, C)
window_start = torch.tensor([10, 30])

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)
eff_L = context_len

fcd_indices = out["_fcd_indices"]
x_full = batch["x_enc"]

for b in range(B):
    ws = window_start[b].item()
    for i in range(fcd_samples):
        expected = ws + i * stride + eff_L - 1
        actual = fcd_indices[b, i].item()
        check(
            f"batch={b}, fcd={i}: idx={actual} == {expected} (stride={stride})",
            actual == expected
        )

        # Verify causal mean
        manual_mean = x_full[b, :actual + 1, 0, 0].mean().item()
        gathered_mean = out["causal_fcd_stats"]["mean"][b, i, 0, 0].item()
        check(
            f"batch={b}, fcd={i}: causal mean correct at idx={actual}",
            abs(manual_mean - gathered_mean) < 1e-4
        )

    # Verify FCDs are spaced by stride
    for i in range(1, fcd_samples):
        gap = fcd_indices[b, i].item() - fcd_indices[b, i-1].item()
        check(
            f"batch={b}: FCD {i-1}→{i} gap = {gap} == stride={stride}",
            gap == stride
        )

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Scaler norm path
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 4: Scaler norm path — causal_stats applied to insample_y")
print("=" * 70)

B, S, C = 2, 50, 1
context_len, horizon, stride, fcd_samples = 8, 4, 1, 5
batch = make_batch(B, S, C)
window_start = torch.tensor([10, 20])

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)

scaler = Scaler(scaler_type='standard', stride=stride)
original_insample = out["insample_y"].clone()

out = scaler(out, norm_type='norm')
normalized = out["insample_y"]

check(
    "norm output shape matches input",
    normalized.shape == original_insample.shape
)

# Verify normalization: (x - mean) / stdev
causal_stats = out["causal_stats"]
expected_norm = (original_insample - causal_stats['mean']) / causal_stats['stdev']
check(
    "norm values match manual (x - mean) / stdev",
    torch.allclose(normalized, expected_norm, atol=1e-5)
)

# Verify each timestep uses its own causal stat, not a global one
# If all timesteps used the same stat, normalized values would have a pattern
# With per-timestep stats, early timesteps normalize differently than late ones
norm_t0 = normalized[:, 0, 0, 0]
norm_t5 = normalized[:, 5, 0, 0]
check(
    "different timesteps produce different normalized values (per-timestep stats)",
    not torch.allclose(norm_t0, norm_t5)
)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Scaler denorm path
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 5: Scaler denorm path — causal_fcd_stats applied to preds")
print("=" * 70)

B, S, C = 2, 50, 1
Q = 3
context_len, horizon, stride, fcd_samples = 8, 4, 1, 5
batch = make_batch(B, S, C)
window_start = torch.tensor([10, 20])

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)
scaler = Scaler(scaler_type='standard', stride=stride)
out = scaler(out, norm_type='norm')

# Create fake normalized preds
fake_preds = torch.randn(B, fcd_samples, horizon, C, Q)
out["preds"] = fake_preds.clone()

out = scaler(out, norm_type='denorm')
denormed = out["preds"]

check(
    "denorm output shape matches input",
    denormed.shape == fake_preds.shape
)

# Verify: denorm = preds * fcd_stdev + fcd_mean
causal_fcd_stats = out["causal_fcd_stats"]
fcd_mean = causal_fcd_stats['mean'][:, :, :, 0:1].unsqueeze(2)
fcd_stdev = causal_fcd_stats['stdev'][:, :, :, 0:1].unsqueeze(2)
expected_denorm = fake_preds * fcd_stdev + fcd_mean
check(
    "denorm values match manual preds * stdev + mean",
    torch.allclose(denormed, expected_denorm, atol=1e-5)
)

# Verify each FCD uses its own stat
denorm_fcd0 = denormed[:, 0, 0, 0, 0]
denorm_fcd4 = denormed[:, 4, 0, 0, 0]
# Same raw pred value, different denorm result (different FCD stats)
same_pred = torch.ones(B, fcd_samples, horizon, C, Q)
out["preds"] = same_pred.clone()
out_same = scaler(out, norm_type='denorm')
check(
    "same pred at different FCDs produces different denorm values",
    not torch.allclose(out_same["preds"][:, 0], out_same["preds"][:, 4])
)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Scaler norm_targets path
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 6: Scaler norm_targets path — causal_fcd_stats applied to outsample_y")
print("=" * 70)

B, S, C = 2, 50, 1
context_len, horizon, stride, fcd_samples = 8, 4, 1, 5
batch = make_batch(B, S, C)
window_start = torch.tensor([10, 20])

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)
scaler = Scaler(scaler_type='standard', stride=stride)
out = scaler(out, norm_type='norm')

original_outsample = out["outsample_y"].clone()
out = scaler(out, norm_type='norm_targets')
normed_targets = out["outsample_y"]

check(
    "norm_targets output shape matches input",
    normed_targets.shape == original_outsample.shape
)

# Verify: norm_targets = (outsample_y - fcd_mean) / fcd_stdev
causal_fcd_stats = out["causal_fcd_stats"]
fcd_mean = causal_fcd_stats['mean'][:, :, :, 0].unsqueeze(2).expand_as(original_outsample)
fcd_stdev = causal_fcd_stats['stdev'][:, :, :, 0].unsqueeze(2).expand_as(original_outsample)
expected_norm_targets = (original_outsample - fcd_mean) / fcd_stdev
check(
    "norm_targets values match manual (y - mean) / stdev",
    torch.allclose(normed_targets, expected_norm_targets, atol=1e-5)
)

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Round-trip — norm then denorm recovers original scale
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 7: Round-trip — norm_targets then denorm recovers original")
print("=" * 70)

B, S, C = 2, 50, 1
Q = 1  # single quantile for clean round-trip
context_len, horizon, stride, fcd_samples = 8, 4, 1, 5
batch = make_batch(B, S, C)
window_start = torch.tensor([10, 20])

out = forking_sequences_call(batch, context_len, horizon, stride, fcd_samples, window_start)
scaler = Scaler(scaler_type='standard', stride=stride)
out = scaler(out, norm_type='norm')

original_outsample = out["outsample_y"].clone()  # [B, n_fcds, H, C]

# norm_targets
out = scaler(out, norm_type='norm_targets')
normed = out["outsample_y"]

# Pretend model predicts exactly the normalized targets (perfect model)
# preds shape needs to be [B, n_fcds, H, C, Q]
out["preds"] = normed.unsqueeze(-1).clone()

# denorm
out = scaler(out, norm_type='denorm')
recovered = out["preds"].squeeze(-1)  # [B, n_fcds, H, C]

check(
    "round-trip: norm_targets → denorm recovers original outsample_y",
    torch.allclose(recovered, original_outsample, atol=1e-4)
)

# Check per-FCD recovery
for b in range(B):
    for i in range(fcd_samples):
        check(
            f"batch={b}, fcd={i}: round-trip error < 1e-4",
            (recovered[b, i] - original_outsample[b, i]).abs().max().item() < 1e-4
        )

print()


# ═══════════════════════════════════════════════════════════════════
# TEST 8: Cross-batch isolation
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("TEST 8: Cross-batch isolation — batch 0 stats independent of batch 1")
print("=" * 70)

B, S, C = 2, 50, 1
context_len, horizon, stride, fcd_samples = 8, 4, 1, 5

# Run with 2-batch
batch_2 = make_batch(2, S, C)
window_start_2 = torch.tensor([10, 20])
out_2 = forking_sequences_call(batch_2, context_len, horizon, stride, fcd_samples, window_start_2)

# Run batch 0 alone
batch_solo = make_batch(1, S, C)
window_start_solo = torch.tensor([10])
out_solo = forking_sequences_call(batch_solo, context_len, horizon, stride, fcd_samples, window_start_solo)

# Batch 0's stats should be identical whether run alone or in a 2-batch
check(
    "batch 0 causal_stats identical when run alone vs in 2-batch",
    torch.allclose(out_2["causal_stats"]["mean"][0], out_solo["causal_stats"]["mean"][0], atol=1e-6)
)
check(
    "batch 0 causal_fcd_stats identical when run alone vs in 2-batch",
    torch.allclose(out_2["causal_fcd_stats"]["mean"][0], out_solo["causal_fcd_stats"]["mean"][0], atol=1e-6)
)

print()


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed}/{total} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 70)