"""
encoder/image_encoder.py
=========================
Image → Latent Vector Encoder

Architecture
------------
Without a pretrained CNN we use a principled signal-processing pipeline:

  RGB image
     │
  Luminance conversion (Y = 0.299R + 0.587G + 0.114B)
     │
  Resize to 32×32 grid
     │
  2D Discrete Cosine Transform (DCT-II)
     │  The DCT is the basis of JPEG — it is a natural frequency decomposition
     │  of visual content.  Low-frequency coefficients = coarse structure;
     │  high-frequency = fine detail.
     │
  Zig-zag scan (low-freq first) → 1D feature vector
     │
  Histogram of oriented gradients (HOG) features at 3 scales
     │  Captures edge structure regardless of colour
     │
  Concatenate + L2 normalise
     │
  Learned random projection → dim-dimensional latent vector
     │
  L2-normalise → unit vector in frequency space

This gives us a deterministic, interpretable image representation
that can be aligned with text vectors in the shared latent space.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# DCT helpers (pure numpy)
# ---------------------------------------------------------------------------

def _dct2(x: np.ndarray) -> np.ndarray:
    """2D DCT-II via row-then-column 1D DCT."""
    def dct1d(v: np.ndarray) -> np.ndarray:
        N = len(v)
        # Rearrange input
        vv = np.concatenate([v[::2], v[-1::-2][: N // 2]])
        V  = np.fft.rfft(vv, n=N)
        k  = np.arange(N // 2 + 1, dtype=float)
        W  = 2 * np.exp(-1j * np.pi * k / (2 * N))
        return np.real(V * W)[:N].astype(np.float32)

    rows = np.array([dct1d(row) for row in x])
    return np.array([dct1d(col) for col in rows.T]).T.astype(np.float32)


def _zigzag(H: int, W: int, n: int) -> list:
    """Return first *n* (row, col) indices in zig-zag order."""
    idx = []
    for d in range(H + W - 1):
        if d % 2 == 0:
            r, c = min(d, H - 1), max(0, d - H + 1)
            while r >= 0 and c < W:
                idx.append((r, c)); r -= 1; c += 1
        else:
            r, c = max(0, d - W + 1), min(d, W - 1)
            while c >= 0 and r < H:
                idx.append((r, c)); r += 1; c -= 1
        if len(idx) >= n:
            break
    return idx[:n]


# ---------------------------------------------------------------------------
# HOG-like gradient features (pure numpy)
# ---------------------------------------------------------------------------

def _hog_features(luma: np.ndarray, cell_size: int = 8, n_bins: int = 9) -> np.ndarray:
    """Minimal HOG: gradient histogram over cells."""
    H, W = luma.shape
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = luma[:, 2:] - luma[:, :-2]
    gy[1:-1, :]  = luma[2:, :] - luma[:-2, :]
    mag  = np.sqrt(gx**2 + gy**2)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180

    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    hog = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            m = mag[cy*cell_size:(cy+1)*cell_size,
                    cx*cell_size:(cx+1)*cell_size]
            a = angle[cy*cell_size:(cy+1)*cell_size,
                      cx*cell_size:(cx+1)*cell_size]
            hist, _ = np.histogram(a, bins=n_bins, range=(0, 180),
                                   weights=m)
            hog.append(hist)
    feat = np.concatenate(hog).astype(np.float32)
    n = np.linalg.norm(feat)
    return feat / (n + 1e-8)


# ---------------------------------------------------------------------------
# ImageEncoder
# ---------------------------------------------------------------------------

class ImageEncoder:
    """
    Encodes images (file paths, numpy arrays, or PIL images) into
    unit latent vectors in the shared frequency space.

    Parameters
    ----------
    dim      : Output latent dimension.
    grid     : Resize grid before DCT (grid × grid).
    n_dct    : Number of DCT zig-zag coefficients to use.
    seed     : RNG seed.
    """

    def __init__(
        self,
        dim: int = 128,
        grid: int = 32,
        n_dct: int = 128,
        seed: int = 42,
    ) -> None:
        self.dim   = dim
        self.grid  = grid
        self.n_dct = min(n_dct, grid * grid)
        self._zz   = _zigzag(grid, grid, self.n_dct)

        # HOG features at 32×32 with cell_size=8 → (32//8)^2 * 9 = 144
        hog_dim = ((grid // 8) ** 2) * 9

        feature_dim = self.n_dct + hog_dim
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal(
            (feature_dim, dim)
        ).astype(np.float32)
        # Orthonormalise
        if feature_dim >= dim:
            Q, _ = np.linalg.qr(self._proj)
            self._proj = Q[:, :dim].astype(np.float32)

        # Try scipy DCT for speed
        try:
            from scipy.fft import dctn as _dctn
            self._dctn = _dctn
            self._use_scipy = True
        except Exception:
            self._use_scipy = False

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_luma(self, image) -> np.ndarray:
        """Convert any input to a (grid, grid) luminance float32 array."""
        if isinstance(image, (str, Path)):
            if not _PIL_AVAILABLE:
                raise ImportError("PIL required to load image files.")
            img = PILImage.open(image).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
        elif _PIL_AVAILABLE and isinstance(image, PILImage.Image):
            arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # To luminance
        if arr.ndim == 3 and arr.shape[2] >= 3:
            luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        elif arr.ndim == 2:
            luma = arr
        else:
            luma = arr.mean(axis=-1)

        # Resize to grid × grid (bilinear-like)
        H, W = luma.shape
        G    = self.grid
        row_idx = np.clip(np.floor(np.linspace(0, H - 1, G)).astype(int), 0, H - 1)
        col_idx = np.clip(np.floor(np.linspace(0, W - 1, G)).astype(int), 0, W - 1)
        return luma[np.ix_(row_idx, col_idx)].astype(np.float32)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, luma: np.ndarray) -> np.ndarray:
        """Extract combined DCT + HOG feature vector."""
        # DCT
        if self._use_scipy:
            dct = self._dctn(luma, norm="ortho").astype(np.float32)
        else:
            dct = _dct2(luma)

        dct_coefs = np.array(
            [dct[r, c] for r, c in self._zz], dtype=np.float32
        )
        # Log-compress
        dct_coefs = np.sign(dct_coefs) * np.log1p(np.abs(dct_coefs))

        # HOG
        hog = _hog_features(luma)

        return np.concatenate([dct_coefs, hog])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, image) -> np.ndarray:
        """
        Encode an image into a unit latent vector.

        Parameters
        ----------
        image : str | Path | np.ndarray | PIL.Image

        Returns
        -------
        np.ndarray  shape (dim,)  float32, L2-normalised
        """
        luma = self._load_luma(image)
        feat = self._extract_features(luma)
        n    = min(len(feat), self._proj.shape[0])
        vec  = feat[:n] @ self._proj[:n, :]
        return self._l2_norm(vec)

    def encode_batch(self, images: list) -> np.ndarray:
        """Encode a list of images. Returns (N, dim)."""
        return np.stack([self.encode(img) for img in images])

    def encode_random(self) -> np.ndarray:
        """Generate a random synthetic image and encode it (for testing)."""
        fake = np.random.rand(self.grid, self.grid).astype(np.float32)
        feat = self._extract_features(fake)
        n    = min(len(feat), self._proj.shape[0])
        vec  = feat[:n] @ self._proj[:n, :]
        return self._l2_norm(vec)

    @staticmethod
    def _l2_norm(vec: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(vec)
        return vec / (n + 1e-8)

    def __repr__(self) -> str:
        return (f"ImageEncoder(dim={self.dim}, "
                f"grid={self.grid}×{self.grid}, n_dct={self.n_dct})")
