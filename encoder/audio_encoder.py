"""
encoder/audio_encoder.py
=========================
Audio → Latent Vector Encoder

Audio is the most natural modality for a "frequency space" architecture —
the auditory cortex literally performs a spectral decomposition.

Pipeline
--------
  waveform (float32 array, mono)
     │
  STFT → magnitude spectrogram
     │
  Mel filterbank (perceptually scaled frequency bands)
     │
  Log compression  (mimics cochlear gain control)
     │
  Temporal statistics: mean + std per band → 1D feature vector
     │
  Random projection → dim-dimensional latent vector
     │
  L2-normalise → unit vector in shared frequency space

No audio library required — pure numpy STFT included.
scipy is used for faster computation if available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pure-numpy STFT
# ---------------------------------------------------------------------------

def _stft(
    signal: np.ndarray,
    frame_len: int = 512,
    hop: int = 256,
) -> np.ndarray:
    """Magnitude spectrogram via overlapping Hann windows + FFT."""
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(frame_len) / frame_len))
    n_frames = max(1, 1 + (len(signal) - frame_len) // hop)
    out = np.zeros((n_frames, frame_len // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        out[i] = np.abs(np.fft.rfft(frame * win)).astype(np.float32)
    return out


def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank matrix. shape (n_mels, n_fft)."""
    def hz2mel(h):  return 2595 * np.log10(1 + h / 700)
    def mel2hz(m):  return 700 * (10 ** (m / 2595) - 1)
    mels  = np.linspace(hz2mel(20), hz2mel(sr / 2), n_mels + 2)
    freqs = mel2hz(mels)
    bins  = np.floor((n_fft - 1) * freqs / (sr / 2)).astype(int).clip(0, n_fft - 1)
    fb    = np.zeros((n_mels, n_fft), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, ctr + 1):
            if ctr > lo: fb[m - 1, k] = (k - lo) / (ctr - lo + 1e-8)
        for k in range(ctr, hi + 1):
            if hi > ctr: fb[m - 1, k] = (hi - k) / (hi - ctr + 1e-8)
    return fb


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------

class AudioEncoder:
    """
    Encodes audio waveforms into unit latent vectors.

    Parameters
    ----------
    dim       : Output latent dimension.
    sr        : Expected sample rate (Hz).
    n_mels    : Number of Mel frequency bands.
    frame_len : STFT frame length (samples).
    hop       : STFT hop size (samples).
    seed      : RNG seed.
    """

    def __init__(
        self,
        dim: int = 128,
        sr: int = 22050,
        n_mels: int = 40,
        frame_len: int = 512,
        hop: int = 256,
        seed: int = 42,
    ) -> None:
        self.dim       = dim
        self.sr        = sr
        self.n_mels    = n_mels
        self.frame_len = frame_len
        self.hop       = hop

        self._fb = _mel_filterbank(n_mels, frame_len // 2 + 1, sr)

        feature_dim = n_mels * 2   # mean + std over time
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal(
            (feature_dim, dim)
        ).astype(np.float32)
        if feature_dim >= dim:
            Q, _ = np.linalg.qr(self._proj)
            self._proj = Q[:, :dim].astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """
        Encode a waveform into a unit latent vector.

        Parameters
        ----------
        waveform : np.ndarray  shape (N,)  float32, values ≈ [-1, 1]

        Returns
        -------
        np.ndarray  shape (dim,)  float32, L2-normalised
        """
        w = waveform.astype(np.float32)
        peak = np.max(np.abs(w))
        if peak > 1e-8:
            w = w / peak
        if len(w) < self.frame_len:
            w = np.pad(w, (0, self.frame_len - len(w)))

        spec = _stft(w, self.frame_len, self.hop)        # (T, F)
        mel  = spec @ self._fb.T                         # (T, n_mels)
        mel  = np.log1p(mel)                             # log compression

        feat = np.concatenate([mel.mean(axis=0),
                               mel.std(axis=0)]).astype(np.float32)
        vec  = feat @ self._proj
        return self._l2(vec)

    def encode_batch(self, waveforms: list) -> np.ndarray:
        """Encode a list of waveforms. Returns (N, dim)."""
        return np.stack([self.encode(w) for w in waveforms])

    def encode_sine(self, freq_hz: float = 440.0, duration: float = 0.5) -> np.ndarray:
        """Helper: encode a synthetic sine wave (for testing)."""
        t = np.linspace(0, duration, int(self.sr * duration))
        wave = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
        return self.encode(wave)

    @staticmethod
    def _l2(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    def __repr__(self) -> str:
        return (f"AudioEncoder(dim={self.dim}, "
                f"sr={self.sr}, n_mels={self.n_mels})")
