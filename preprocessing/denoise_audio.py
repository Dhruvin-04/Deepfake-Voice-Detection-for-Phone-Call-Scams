"""Optional denoising experiment using spectral gating."""

from __future__ import annotations

import numpy as np


def denoise_audio(
    signal: np.ndarray,
    sr: int,
    stationary: bool = True,
    prop_decrease: float = 0.9,
) -> np.ndarray:
    """Reduce background noise. Used only for robustness experiments."""
    try:
        import noisereduce as nr
    except ImportError as exc:
        raise ImportError(
            "Optional denoising requires 'noisereduce'. "
            "Install it with: pip install noisereduce"
        ) from exc

    return nr.reduce_noise(
        y=np.asarray(signal, dtype=np.float32),
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
    ).astype(np.float32)
