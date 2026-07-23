from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.ndimage import uniform_filter


@dataclass(frozen=True)
class PolarimetricConfig:
    """Configuration for explicit H/V polarimetric RD features."""

    velocity_window: int = 5
    range_window: int = 3
    eps: float = 1e-8
    zdr_clip_db: float = 20.0

    def __post_init__(self) -> None:
        if self.velocity_window <= 0 or self.range_window <= 0:
            raise ValueError("Neighborhood sizes must be positive")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.zdr_clip_db <= 0:
            raise ValueError("zdr_clip_db must be positive")


def _ensure_iq_pair(
    h: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(h)
    v = np.asarray(v)
    if h.shape != v.shape:
        raise ValueError(
            f"H/V shape mismatch: {h.shape} vs {v.shape}"
        )
    if h.ndim != 2:
        raise ValueError(
            f"H/V IQ must be 2-D [slow_time, range], got {h.shape}"
        )
    if not np.iscomplexobj(h) or not np.iscomplexobj(v):
        raise TypeError("H/V inputs must be complex IQ")
    if not np.isfinite(h).all() or not np.isfinite(v).all():
        raise ValueError("H/V IQ contains NaN or Inf")
    return h, v


def complex_rd(
    iq: np.ndarray,
) -> np.ndarray:
    """Slow-time FFT with a Hann window.

    Input shape: [pulses, range_gates]
    Output shape: [doppler_bins, range_gates]
    """
    iq = np.asarray(iq)
    if iq.ndim != 2:
        raise ValueError("IQ must be two-dimensional")
    window = np.hanning(iq.shape[0]).astype(np.float64)[:, None]
    rd = np.fft.fftshift(
        np.fft.fft(iq * window, axis=0),
        axes=0,
    )
    return rd.astype(np.complex64, copy=False)


def _local_mean(
    values: np.ndarray,
    velocity_window: int,
    range_window: int,
) -> np.ndarray:
    size = (int(velocity_window), int(range_window))
    values = np.asarray(values)
    if np.iscomplexobj(values):
        real = uniform_filter(
            values.real.astype(np.float64),
            size=size,
            mode="nearest",
        )
        imag = uniform_filter(
            values.imag.astype(np.float64),
            size=size,
            mode="nearest",
        )
        return real + 1j * imag
    return uniform_filter(
        values.astype(np.float64),
        size=size,
        mode="nearest",
    )


def robust_power_channel(
    power: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert power to a robust [0,1] dB channel."""
    db = 10.0 * np.log10(
        np.asarray(power, dtype=np.float64) + eps
    )
    low = float(np.percentile(db, low_percentile))
    high = float(np.percentile(db, high_percentile))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Invalid robust normalization percentiles")
    if high <= low:
        return np.zeros_like(db, dtype=np.float32)
    return np.clip(
        (db - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def robust_signed_channel(
    values: np.ndarray,
    *,
    percentile: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Scale a signed real-valued map approximately to [-1,1]."""
    values = np.asarray(values, dtype=np.float64)
    scale = float(np.percentile(np.abs(values), percentile))
    if not np.isfinite(scale) or scale <= eps:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(values / scale, -1.0, 1.0).astype(np.float32)


def explicit_polarimetric_rd(
    h: np.ndarray,
    v: np.ndarray,
    config: PolarimetricConfig | None = None,
) -> Mapping[str, np.ndarray]:
    """Compute explicit H/V polarimetric features in the RD plane.

    Notes
    -----
    ``zdr_like_db`` and ``phi_dp_like_rad`` are relative features. Without
    amplitude/phase channel calibration they must not be reported as absolute
    meteorological ZDR or PhiDP.

    ``rho_hv_local`` is estimated from a local neighborhood. A pointwise
    |H*conj(V)| / (|H||V|) would be almost identically one and is therefore
    not used as a correlation coefficient.
    """
    config = config or PolarimetricConfig()
    h, v = _ensure_iq_pair(h, v)

    rd_h = complex_rd(h)
    rd_v = complex_rd(v)

    power_h = np.abs(rd_h).astype(np.float64) ** 2
    power_v = np.abs(rd_v).astype(np.float64) ** 2
    cross = rd_h.astype(np.complex128) * np.conj(
        rd_v.astype(np.complex128)
    )

    local_power_h = _local_mean(
        power_h,
        config.velocity_window,
        config.range_window,
    )
    local_power_v = _local_mean(
        power_v,
        config.velocity_window,
        config.range_window,
    )
    local_cross = _local_mean(
        cross,
        config.velocity_window,
        config.range_window,
    )

    denominator = np.sqrt(
        np.maximum(local_power_h, 0.0)
        * np.maximum(local_power_v, 0.0)
    ) + config.eps

    rho_hv_local = np.clip(
        np.abs(local_cross) / denominator,
        0.0,
        1.0,
    )
    phi_dp_like_rad = np.angle(local_cross)
    zdr_like_db = 10.0 * np.log10(
        (local_power_h + config.eps)
        / (local_power_v + config.eps)
    )

    total_power = local_power_h + local_power_v + config.eps
    stokes_s1 = (
        local_power_h - local_power_v
    ) / total_power
    stokes_s2 = (
        2.0 * np.real(local_cross)
    ) / total_power
    stokes_s3 = (
        -2.0 * np.imag(local_cross)
    ) / total_power

    return {
        "rd_h": rd_h.astype(np.complex64, copy=False),
        "rd_v": rd_v.astype(np.complex64, copy=False),
        "power_h": power_h.astype(np.float32),
        "power_v": power_v.astype(np.float32),
        "local_power_h": local_power_h.astype(np.float32),
        "local_power_v": local_power_v.astype(np.float32),
        "cross_local": local_cross.astype(
            np.complex64,
            copy=False,
        ),
        "zdr_like_db": zdr_like_db.astype(np.float32),
        "rho_hv_local": rho_hv_local.astype(np.float32),
        "phi_dp_like_rad": phi_dp_like_rad.astype(np.float32),
        "phi_cos": np.cos(phi_dp_like_rad).astype(np.float32),
        "phi_sin": np.sin(phi_dp_like_rad).astype(np.float32),
        "stokes_s1": np.clip(stokes_s1, -1.0, 1.0).astype(
            np.float32
        ),
        "stokes_s2": np.clip(stokes_s2, -1.0, 1.0).astype(
            np.float32
        ),
        "stokes_s3": np.clip(stokes_s3, -1.0, 1.0).astype(
            np.float32
        ),
    }


def make_power2(
    features: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Current H/V normalized power-RD input."""
    return np.stack(
        (
            robust_power_channel(features["power_h"]),
            robust_power_channel(features["power_v"]),
        ),
        axis=0,
    ).astype(np.float32)


def make_polar6(
    features: Mapping[str, np.ndarray],
    *,
    zdr_clip_db: float = 20.0,
) -> np.ndarray:
    """Six explicit polarimetric RD channels.

    Channels:
      0 H normalized power
      1 V normalized power
      2 clipped relative ZDR-like value scaled to [-1,1]
      3 local |rho_HV| in [0,1]
      4 cos(relative differential phase)
      5 sin(relative differential phase)
    """
    zdr = np.clip(
        features["zdr_like_db"] / float(zdr_clip_db),
        -1.0,
        1.0,
    )
    return np.stack(
        (
            robust_power_channel(features["power_h"]),
            robust_power_channel(features["power_v"]),
            zdr.astype(np.float32),
            features["rho_hv_local"].astype(np.float32),
            features["phi_cos"].astype(np.float32),
            features["phi_sin"].astype(np.float32),
        ),
        axis=0,
    ).astype(np.float32)


def make_ri4(
    features: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Robustly normalized real/imaginary RD channels."""
    rd_h = features["rd_h"]
    rd_v = features["rd_v"]
    return np.stack(
        (
            robust_signed_channel(rd_h.real),
            robust_signed_channel(rd_h.imag),
            robust_signed_channel(rd_v.real),
            robust_signed_channel(rd_v.imag),
        ),
        axis=0,
    ).astype(np.float32)


def make_ri8(
    features: Mapping[str, np.ndarray],
    *,
    zdr_clip_db: float = 20.0,
) -> np.ndarray:
    """RI4 plus four explicit polarimetric channels."""
    ri4 = make_ri4(features)
    zdr = np.clip(
        features["zdr_like_db"] / float(zdr_clip_db),
        -1.0,
        1.0,
    )
    extra = np.stack(
        (
            zdr.astype(np.float32),
            features["rho_hv_local"].astype(np.float32),
            features["phi_cos"].astype(np.float32),
            features["phi_sin"].astype(np.float32),
        ),
        axis=0,
    )
    return np.concatenate((ri4, extra), axis=0).astype(
        np.float32
    )
