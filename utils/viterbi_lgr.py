"""
viterbi_lgr.py — Temporally coherent latent granular resynthesis via Viterbi path selection.

Extends the baseline greedy/softmax matching from Tokui & Baker (ISMIR 2025) with a
global sequence constraint that penalises abrupt jumps between codebook grains.

Algorithm
---------
Given:
  - emission matrix E[t, i]  = cosine_similarity(target_grain_t, pool_grain_i)   (N_t × N_p)
  - transition matrix Tr[i,j] = cosine_distance(pool_grain_i, pool_grain_j)      (N_p × N_p)

Viterbi score:
  V[t, j] = max_i ( V[t-1, i]  −  smoothness * Tr[i, j] )  +  E[t, j]

The `smoothness` hyperparameter (λ) interpolates between:
  λ = 0  → identical to deterministic argmax (original paper, tau→0)
  λ → ∞  → fully path-constrained, ignores timbral match, maximises trajectory smoothness

Warning: For large pools (N_p > 500) the (N_p, N_p) transition matrix can be large.
No chunking is implemented — keep pool sizes manageable.
"""

import torch
import torch.nn.functional as F
from utils.m2l_utils import build_granular_codebook


def viterbi_granular_resynthesis(
    target_latents,       # torch.Tensor (T_target, D)
    pool_grains,          # torch.Tensor (N_pool, D)  — mean-pooled
    pool_grain_frames,    # torch.Tensor (N_pool, grain_size, D)
    grain_size,           # int
    smoothness=1.0,       # float  λ — transition penalty weight
):
    """
    Viterbi-optimal grain path resynthesis.

    Parameters
    ----------
    target_latents : torch.Tensor
        Target latent tensor of shape (T_target, D).
    pool_grains : torch.Tensor
        Mean-pooled pool codebook grains of shape (N_pool, D).
    pool_grain_frames : torch.Tensor
        Original frame sequences of pool grains, shape (N_pool, grain_size, D).
    grain_size : int
        Number of consecutive latent frames per grain.
    smoothness : float
        Transition penalty weight λ. 0 = greedy argmax, higher = smoother path.

    Returns
    -------
    hybrid_latents : torch.Tensor
        Hybrid latent tensor of shape (T_target, D).
    path : torch.Tensor
        Chosen pool index per grain, shape (N_target_grains,).
    match_similarities : torch.Tensor
        Emission cosine sim of chosen grain, shape (N_target_grains,).
    """
    T_target = target_latents.shape[0]
    D = target_latents.shape[1]
    device = target_latents.device
    pool_grains = pool_grains.to(device)
    pool_grain_frames = pool_grain_frames.to(device)

    # Zero-pad target to next multiple of grain_size
    remainder = T_target % grain_size
    if remainder != 0:
        pad_frames = grain_size - remainder
        padding = torch.zeros(pad_frames, D, device=device)
        padded_target = torch.cat([target_latents, padding], dim=0)
    else:
        padded_target = target_latents

    # Build target grains (non-overlapping)
    target_grains, _ = build_granular_codebook(padded_target, grain_size, grain_size)

    N_t = target_grains.shape[0]
    N_p = pool_grains.shape[0]

    # L2-normalise for cosine similarity
    t_norm = F.normalize(target_grains, dim=-1)
    p_norm = F.normalize(pool_grains, dim=-1)

    # Emission matrix: (N_t, N_p) cosine similarities
    emission = torch.mm(t_norm, p_norm.T)

    # Transition cost matrix: (N_p, N_p) cosine distances
    Tr = 1.0 - torch.mm(p_norm, p_norm.T)

    # Normalise transition costs relative to emission scale so that λ is
    # interpretable: λ=1 means the average transition penalty equals the
    # average emission score.  Without this, raw cosine distances (~0.5–1.0)
    # dominate the much smaller emission scores (~0.2–0.4), causing the path
    # to collapse onto a single grain even at moderate λ.
    emission_mean = emission.mean()
    tr_mean = Tr.mean()
    if tr_mean > 1e-8:
        Tr = Tr * (emission_mean / tr_mean)

    # Forward Viterbi pass
    V = torch.zeros(N_t, N_p, device=device)
    ptr = torch.zeros(N_t, N_p, dtype=torch.long, device=device)

    V[0] = emission[0]

    for t in range(1, N_t):
        scores = V[t - 1].unsqueeze(1) - smoothness * Tr  # (N_p, N_p)
        best_prev, ptr[t] = scores.max(dim=0)
        V[t] = best_prev + emission[t]

    # Backtrack
    path = torch.zeros(N_t, dtype=torch.long, device=device)
    path[-1] = V[-1].argmax()
    for t in range(N_t - 2, -1, -1):
        path[t] = ptr[t + 1, path[t + 1]]

    # Extract match similarities
    match_similarities = emission[torch.arange(N_t, device=device), path]

    # Reconstruct hybrid latents
    matched_grain_frames = pool_grain_frames[path]  # (N_t, grain_size, D)
    hybrid_latents = matched_grain_frames.reshape(-1, D)
    hybrid_latents = hybrid_latents[:T_target]

    print(f"  Viterbi: {path.unique().numel()} unique grains / {N_p} pool grains  (λ={smoothness})")

    return hybrid_latents, path, match_similarities


def batch_viterbi_sweep(
    target_latents,
    pool_grains,
    pool_grain_frames,
    grain_size,
    smoothness_values,
):
    """
    Run Viterbi resynthesis for multiple smoothness values.

    Parameters
    ----------
    target_latents : torch.Tensor
        Target latent tensor of shape (T_target, D).
    pool_grains : torch.Tensor
        Mean-pooled pool codebook grains of shape (N_pool, D).
    pool_grain_frames : torch.Tensor
        Original frame sequences of pool grains, shape (N_pool, grain_size, D).
    grain_size : int
        Number of consecutive latent frames per grain.
    smoothness_values : list[float]
        List of smoothness λ values to sweep.

    Returns
    -------
    list[dict]
        One dict per smoothness value with keys:
        "smoothness", "hybrid_latents", "path", "match_similarities".
    """
    results = []
    for lam in smoothness_values:
        hybrid_latents, path, match_similarities = viterbi_granular_resynthesis(
            target_latents, pool_grains, pool_grain_frames,
            grain_size=grain_size, smoothness=lam,
        )
        results.append({
            "smoothness": lam,
            "hybrid_latents": hybrid_latents,
            "path": path,
            "match_similarities": match_similarities,
        })
    return results
