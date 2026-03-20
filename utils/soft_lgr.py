"""
soft_lgr.py — Soft latent interpolation for latent granular resynthesis.

Instead of selecting a single codebook grain (hard selection) or a globally
optimal path (Viterbi), this method computes a weighted blend of the top-k
closest pool grains in latent space and passes the blended vector to the decoder.

Research question: is Music2Latent's latent space linear enough that convex
combinations of grains decode to perceptually meaningful audio?
"""

import torch
import torch.nn.functional as F
from utils.m2l_utils import build_granular_codebook


def soft_granular_resynthesis(
    target_latents,       # torch.Tensor (T_target, D)
    pool_grains,          # torch.Tensor (N_pool, D)
    pool_grain_frames,    # torch.Tensor (N_pool, grain_size, D)
    grain_size,           # int
    top_k=5,              # int — number of grains to blend
    temperature=1.0,      # float — softmax temperature over top-k similarities
):
    """
    Soft latent blending resynthesis using top-k weighted grain interpolation.

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
    top_k : int
        Number of closest grains to blend.
    temperature : float
        Softmax temperature over top-k similarities.

    Returns
    -------
    hybrid_latents : torch.Tensor
        Hybrid latent tensor of shape (T_target, D).
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

    # Cosine similarity matrix: (N_t, N_p)
    t_norm = F.normalize(target_grains, dim=-1)
    p_norm = F.normalize(pool_grains, dim=-1)
    sim = torch.mm(t_norm, p_norm.T)

    # Top-k selection
    topk_sim, topk_idx = sim.topk(top_k, dim=-1)  # (N_t, k), (N_t, k)

    # Softmax weights over top-k similarities
    weights = F.softmax(topk_sim / temperature, dim=-1)  # (N_t, k)

    # Index frames for top-k grains
    topk_frames = pool_grain_frames[topk_idx]  # (N_t, k, grain_size, D)

    # Weighted blend
    blended = (weights[:, :, None, None] * topk_frames).sum(dim=1)  # (N_t, grain_size, D)

    # Reshape and trim
    hybrid_latents = blended.reshape(-1, D)
    hybrid_latents = hybrid_latents[:T_target]

    print(f"  Soft blend: top_k={top_k}, tau={temperature}")

    return hybrid_latents
