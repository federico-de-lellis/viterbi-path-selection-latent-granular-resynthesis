"""
Utility functions for Latent Granular Resynthesis with Music2Latent.

"""

import numpy as np
import torch
import torchaudio
import soundfile as sf


def load_wav_mono(path, target_sr=44100):
    """
    Load a mono WAV file as a float32 numpy array.

    Parameters
    ----------
    path : str
        Path to WAV file.
    target_sr : int
        Expected sample rate (default 44100).

    Returns
    -------
    np.ndarray
        Audio waveform of shape (T,), dtype float32.

    Raises
    ------
    ValueError
        If the file's sample rate doesn't match target_sr.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"Expected {target_sr} Hz but got {sr}. Resample externally.")
    return audio


@torch.no_grad()
def encode_continuous(wv_np, encdec):
    """
    Encode audio to continuous latent representations via Music2Latent.

    Parameters
    ----------
    wv_np : np.ndarray
        Mono audio waveform of shape (T,), float32, at 44100 Hz.
    encdec : music2latent.EncoderDecoder
        Music2Latent encoder-decoder instance.

    Returns
    -------
    torch.Tensor
        Latent tensor of shape (T_frames, 64), dtype float32.
    """
    latent = encdec.encode(wv_np)  # (1, 64, T_frames) — may be numpy or torch
    latent = torch.as_tensor(np.asarray(latent, dtype=np.float32))
    return latent.squeeze(0).T.contiguous()  # (T_frames, 64)


def build_granular_codebook(latents, grain_size, stride):
    """
    Segment a latent frame sequence into grains.

    Returns mean-pooled grain vectors (for similarity matching) and the
    original multi-frame grain sequences (for reconstruction).

    Parameters
    ----------
    latents : torch.Tensor
        Latent tensor of shape (T, D).
    grain_size : int
        Number of consecutive latent frames per grain.
    stride : int
        Step between successive grain start positions.

    Returns
    -------
    grains : torch.Tensor
        Mean-pooled grain codebook of shape (N_grains, D).
    grain_frames : torch.Tensor
        Original frame sequences of shape (N_grains, grain_size, D).
    """
    T, D = latents.shape
    starts = range(0, T - grain_size + 1, stride)
    grain_frames = torch.stack([latents[s:s + grain_size] for s in starts])  # (N, grain_size, D)
    grains = grain_frames.mean(dim=1)  # (N, D)
    print(f"  build_granular_codebook: {T} frames -> {grains.shape[0]} grains "
          f"(grain_size={grain_size}, stride={stride})")
    return grains, grain_frames


def augment_and_build_codebook(wv_np, encdec, grain_size, stride,
                               semitone_shifts=(-2, -1, 1, 2),
                               device="cpu"):
    """
    Build an augmented granular codebook with pitch-shifted source variants.

    Parameters
    ----------
    wv_np : np.ndarray
        Mono audio waveform of shape (T,), float32, at 44100 Hz.
    encdec : music2latent.EncoderDecoder
        Music2Latent encoder-decoder instance.
    grain_size : int
        Number of consecutive latent frames per grain.
    stride : int
        Step between successive grains.
    semitone_shifts : tuple of int
        Semitone shifts for data augmentation.
    device : str, optional
        Torch device string for the returned tensors, e.g. "cpu" or "cuda".
        Defaults to "cpu". Pass torch.device or a device string. The encoder
        always runs on its own internal device; this only controls where the
        pooled output tensors are placed.

    Returns
    -------
    pool : torch.Tensor
        Mean-pooled augmented codebook of shape (N_total_grains, D).
    pool_frames : torch.Tensor
        Original frame sequences of shape (N_total_grains, grain_size, D).
    """
    sample_rate = 44100

    # Original (unshifted)
    latents_orig = encode_continuous(wv_np, encdec)
    grains_orig, frames_orig = build_granular_codebook(latents_orig, grain_size, stride)
    all_grains = [grains_orig]
    all_frames = [frames_orig]
    n_before = grains_orig.shape[0]
    print(f"  Original: {n_before} grains")

    # Pitch-shifted variants
    for shift in semitone_shifts:
        audio_tensor = torch.from_numpy(wv_np).unsqueeze(0)  # (1, T) — always CPU for PitchShift
        pitch_shifter = torchaudio.transforms.PitchShift(sample_rate, shift)
        shifted_tensor = pitch_shifter(audio_tensor)          # (1, T) on CPU
        # Move to target device before encoding
        shifted_np = shifted_tensor.squeeze(0).detach().to("cpu").numpy().astype(np.float32)
        latents_shifted = encode_continuous(shifted_np, encdec)
        grains_shifted, frames_shifted = build_granular_codebook(latents_shifted, grain_size, stride)
        all_grains.append(grains_shifted)
        all_frames.append(frames_shifted)
        print(f"  Shift {shift:+d} semitones: {grains_shifted.shape[0]} grains")

    pool = torch.cat(all_grains, dim=0).to(device)
    pool_frames = torch.cat(all_frames, dim=0).to(device)
    print(f"  Total pool: {n_before} grains (original) -> "
          f"{pool.shape[0]} grains (after augmentation)")
    return pool, pool_frames


@torch.no_grad()
def decode_latents(latents_TD, encdec):
    """
    Decode a latent frame sequence back to audio via Music2Latent.

    Parameters
    ----------
    latents_TD : torch.Tensor
        Latent tensor of shape (T_frames, 64).
    encdec : music2latent.EncoderDecoder
        Music2Latent encoder-decoder instance.

    Returns
    -------
    np.ndarray
        Audio waveform of shape (T_audio,), dtype float32.
    """
    # (T, 64) -> (1, 64, T) numpy for encdec.decode
    latent_1DT = latents_TD.T.unsqueeze(0).cpu().numpy().astype(np.float32)
    wv_rec = encdec.decode(latent_1DT)
    return np.array(wv_rec, dtype=np.float32).squeeze()
