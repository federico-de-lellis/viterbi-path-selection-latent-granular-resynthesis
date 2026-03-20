"""
evaluation.py — Quantitative evaluation for latent granular resynthesis.

Metrics
-------
structural_preservation(target, hybrid)
    RMS envelope Pearson correlation. Measures how well the output follows
    the target's temporal energy contour.
    Higher = better structural preservation.

evaluate_all(target_np, hybrid_np, sr, label)
    Convenience wrapper returning a dict with the metric plus the label.

compare_methods(results_list)
    Pretty-print a comparison table given a list of evaluate_all dicts.
"""

import numpy as np
import librosa


def structural_preservation(target_np, hybrid_np, sr=44100, frame_length=2048, hop_length=512):
    """
    RMS envelope Pearson correlation between target and hybrid audio.

    Parameters
    ----------
    target_np : np.ndarray
        Target audio waveform, shape (T,), float32.
    hybrid_np : np.ndarray
        Hybrid audio waveform, shape (T,), float32.
    sr : int
        Sample rate.
    frame_length : int
        RMS frame length.
    hop_length : int
        RMS hop length.

    Returns
    -------
    float
        Pearson r in [-1, 1]; higher = better structure.
    """
    env_t = librosa.feature.rms(y=target_np, frame_length=frame_length, hop_length=hop_length)[0].squeeze()
    env_h = librosa.feature.rms(y=hybrid_np, frame_length=frame_length, hop_length=hop_length)[0].squeeze()
    min_len = min(len(env_t), len(env_h))
    return float(np.corrcoef(env_t[:min_len], env_h[:min_len])[0, 1])


def evaluate_all(target_np, hybrid_np, sr=44100, label=""):
    """
    Compute all evaluation metrics for a single resynthesis result.

    Parameters
    ----------
    target_np : np.ndarray
        Target audio waveform.
    hybrid_np : np.ndarray
        Hybrid audio waveform.
    sr : int
        Sample rate.
    label : str
        Label for this result (e.g. "Viterbi λ=1.0").

    Returns
    -------
    dict
        Keys: "label", "structural_preservation".
    """
    sp = structural_preservation(target_np, hybrid_np, sr=sr)
    print(f"  [{label}]  structure={sp:.4f}")
    return {
        "label": label,
        "structural_preservation": sp,
    }


def compare_methods(results_list):
    """
    Pretty-print a comparison table of evaluation results.

    Parameters
    ----------
    results_list : list[dict]
        List of dicts from evaluate_all.
    """
    best_struct = max(r["structural_preservation"] for r in results_list)

    print(f"{'Method':<30s} {'Structure corr ↑':>16s}")
    print("-" * 48)
    for r in results_list:
        sp = r["structural_preservation"]
        suffix = "   ← best" if sp == best_struct else ""
        print(f"{r['label']:<30s} {sp:>16.4f}{suffix}")