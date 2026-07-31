from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import stft

from features.polarimetric_rd import PolarimetricConfig, explicit_polarimetric_rd


FEATURE_DOMAINS: Mapping[str, str] = {
    "quality": "input and anchor quality",
    "time": "anchor-range slow-time statistics",
    "rd": "range-Doppler energy and clutter morphology",
    "polar": "relative H/V polarimetric statistics",
    "tf": "normalized-frequency time-frequency descriptors",
}


@dataclass(frozen=True)
class MultiDomainFeatureConfig:
    polar_velocity_window: int = 3
    polar_range_window: int = 3
    roi_velocity_radius: int = 2
    roi_range_radius: int = 2
    zero_doppler_radius: int = 1
    edge_doppler_width: int = 4
    main_band_radius: int = 2
    stft_nperseg: int = 128
    stft_overlap_fraction: float = 0.75
    minimum_long_window_pulses: int = 512
    eps: float = 1e-12

    def __post_init__(self) -> None:
        integer_positive = (
            self.polar_velocity_window,
            self.polar_range_window,
            self.edge_doppler_width,
            self.stft_nperseg,
            self.minimum_long_window_pulses,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("positive feature dimensions must be greater than zero")
        if min(
            self.roi_velocity_radius,
            self.roi_range_radius,
            self.zero_doppler_radius,
            self.main_band_radius,
        ) < 0:
            raise ValueError("feature radii must be nonnegative")
        if not 0.0 <= self.stft_overlap_fraction < 1.0:
            raise ValueError("stft_overlap_fraction must be in [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def _validate_iq(h: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(h)
    v = np.asarray(v)
    if h.shape != v.shape or h.ndim != 2:
        raise ValueError("H/V IQ must be aligned [slow_time, range] arrays")
    if not np.iscomplexobj(h) or not np.iscomplexobj(v):
        raise TypeError("H/V IQ must be complex")
    if h.shape[0] < 8 or h.shape[1] < 1:
        raise ValueError("H/V IQ does not contain enough samples")
    if not np.isfinite(h).all() or not np.isfinite(v).all():
        raise ValueError("H/V IQ contains NaN or Inf")
    return h.astype(np.complex64, copy=False), v.astype(np.complex64, copy=False)


def _normalized_distribution(values: np.ndarray, eps: float) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= eps:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def _entropy(probability: np.ndarray, eps: float) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.size <= 1:
        return 0.0
    return float(
        -np.sum(probability * np.log(probability + eps)) / np.log(probability.size)
    )


def _coefficient_of_variation(values: np.ndarray, eps: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std() / (abs(values.mean()) + eps))


def _excess_kurtosis(values: np.ndarray, eps: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    variance = float(np.mean(centered**2))
    if variance <= eps:
        return 0.0
    return float(np.mean(centered**4) / (variance**2) - 3.0)


def _normalized_complex_coherence(
    left: np.ndarray,
    right: np.ndarray,
    eps: float,
) -> float:
    numerator = abs(np.sum(left * np.conj(right)))
    denominator = np.sqrt(
        float(np.sum(np.abs(left) ** 2)) * float(np.sum(np.abs(right) ** 2))
    )
    return float(np.clip(numerator / (denominator + eps), 0.0, 1.0))


def _phase_resultant(cross: np.ndarray, eps: float) -> float:
    cross = np.asarray(cross)
    unit = cross / (np.abs(cross) + eps)
    return float(np.clip(abs(np.mean(unit)), 0.0, 1.0))


def _lag_one_coherence(values: np.ndarray, eps: float) -> float:
    if len(values) < 2:
        return 0.0
    return _normalized_complex_coherence(values[1:], values[:-1], eps)


def _bounded_roi(
    values: np.ndarray,
    velocity_index: int,
    range_index: int,
    velocity_radius: int,
    range_radius: int,
) -> np.ndarray:
    v0 = max(0, velocity_index - velocity_radius)
    v1 = min(values.shape[0], velocity_index + velocity_radius + 1)
    r0 = max(0, range_index - range_radius)
    r1 = min(values.shape[1], range_index + range_radius + 1)
    return values[v0:v1, r0:r1]


def _spectral_features(
    spectrum: np.ndarray,
    peak_index: int,
    config: MultiDomainFeatureConfig,
) -> dict[str, float]:
    probability = _normalized_distribution(spectrum, config.eps)
    bins = np.arange(probability.size, dtype=np.float64)
    center = probability.size // 2
    centroid = float(np.sum(bins * probability))
    variance = float(np.sum(((bins - centroid) ** 2) * probability))
    main_start = max(0, peak_index - config.main_band_radius)
    main_end = min(probability.size, peak_index + config.main_band_radius + 1)
    zero_start = max(0, center - config.zero_doppler_radius)
    zero_end = min(probability.size, center + config.zero_doppler_radius + 1)
    main_fraction = float(probability[main_start:main_end].sum())
    geometric_mean = float(np.exp(np.mean(np.log(spectrum + config.eps))))
    arithmetic_mean = float(np.mean(spectrum) + config.eps)
    return {
        "rd_anchor_centroid_bin": centroid,
        "rd_anchor_centroid_offset": centroid - center,
        "rd_anchor_spectral_width_bins": float(np.sqrt(max(variance, 0.0))),
        "rd_anchor_entropy": _entropy(probability, config.eps),
        "rd_anchor_flatness": geometric_mean / arithmetic_mean,
        "rd_anchor_peak_fraction": float(probability[peak_index]),
        "rd_anchor_main_band_fraction": main_fraction,
        "rd_anchor_sideband_fraction": 1.0 - main_fraction,
        "rd_anchor_zero_doppler_fraction": float(
            probability[zero_start:zero_end].sum()
        ),
    }


def _time_frequency_features(
    h_series: np.ndarray,
    v_series: np.ndarray,
    config: MultiDomainFeatureConfig,
) -> dict[str, float | int]:
    nperseg = min(config.stft_nperseg, len(h_series))
    noverlap = min(
        nperseg - 1,
        int(round(nperseg * config.stft_overlap_fraction)),
    )
    _, _, h_tf = stft(
        h_series,
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    _, _, v_tf = stft(
        v_series,
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    power = np.fft.fftshift(np.abs(h_tf) ** 2 + np.abs(v_tf) ** 2, axes=0)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0))
    frame_energy = power.sum(axis=0)
    probabilities = power / (frame_energy[None, :] + config.eps)
    centroid = np.sum(frequencies[:, None] * probabilities, axis=0)
    bandwidth = np.sqrt(
        np.maximum(
            np.sum(
                ((frequencies[:, None] - centroid[None, :]) ** 2) * probabilities,
                axis=0,
            ),
            0.0,
        )
    )
    entropy = -np.sum(
        probabilities * np.log(probabilities + config.eps), axis=0
    ) / np.log(nperseg)
    dominant = frequencies[np.argmax(power, axis=0)]
    return {
        "tf_frame_count": int(power.shape[1]),
        "tf_nperseg": int(nperseg),
        "tf_centroid_mean_normalized": float(np.mean(centroid)),
        "tf_centroid_std_normalized": float(np.std(centroid)),
        "tf_bandwidth_mean_normalized": float(np.mean(bandwidth)),
        "tf_bandwidth_std_normalized": float(np.std(bandwidth)),
        "tf_entropy_mean": float(np.mean(entropy)),
        "tf_entropy_std": float(np.std(entropy)),
        "tf_dominant_std_normalized": float(np.std(dominant)),
        "tf_dominant_span_normalized": float(np.max(dominant) - np.min(dominant)),
        "tf_energy_cv": _coefficient_of_variation(frame_energy, config.eps),
        "tf_long_window_available": int(
            len(h_series) >= config.minimum_long_window_pulses
            and power.shape[1] >= 8
        ),
    }


def extract_multidomain_features(
    h: np.ndarray,
    v: np.ndarray,
    config: MultiDomainFeatureConfig | None = None,
    *,
    anchor_velocity_index: int | None = None,
    anchor_range_index: int | None = None,
) -> dict[str, float | int]:
    """Extract sample-independent H/V features without using class labels.

    If an external detector candidate is supplied, local time, polarimetric and
    time-frequency features use that candidate. Otherwise the strongest raw
    combined-power cell is used. Truth coordinates are never accepted. Global
    scene features always describe the strongest raw cell. Time-frequency
    values use normalized frequency because PRF is not available for every
    current dataset.
    """
    config = config or MultiDomainFeatureConfig()
    h, v = _validate_iq(h, v)
    polar_config = PolarimetricConfig(
        velocity_window=config.polar_velocity_window,
        range_window=config.polar_range_window,
        eps=max(config.eps, 1e-8),
    )
    polar = explicit_polarimetric_rd(h, v, polar_config)
    combined = polar["power_h"].astype(np.float64) + polar["power_v"].astype(
        np.float64
    )
    peak_velocity, peak_range = np.unravel_index(int(np.argmax(combined)), combined.shape)
    supplied = (anchor_velocity_index is not None, anchor_range_index is not None)
    if supplied[0] != supplied[1]:
        raise ValueError("external anchor coordinates must be supplied together")
    if supplied[0]:
        anchor_velocity = int(anchor_velocity_index)
        anchor_range = int(anchor_range_index)
        if not (
            0 <= anchor_velocity < combined.shape[0]
            and 0 <= anchor_range < combined.shape[1]
        ):
            raise ValueError("external anchor lies outside the RD map")
    else:
        anchor_velocity, anchor_range = int(peak_velocity), int(peak_range)
    center = combined.shape[0] // 2
    total_energy = float(combined.sum()) + config.eps
    zero_start = max(0, center - config.zero_doppler_radius)
    zero_end = min(combined.shape[0], center + config.zero_doppler_radius + 1)
    edge = min(config.edge_doppler_width, combined.shape[0] // 2)
    edge_energy = float(combined[:edge].sum() + combined[-edge:].sum())
    anchor_spectrum = combined[:, anchor_range]
    h_series = h[:, anchor_range]
    v_series = v[:, anchor_range]
    magnitude_h = np.abs(h_series).astype(np.float64)
    magnitude_v = np.abs(v_series).astype(np.float64)
    roi = {
        name: _bounded_roi(
            np.asarray(values),
            anchor_velocity,
            anchor_range,
            config.roi_velocity_radius,
            config.roi_range_radius,
        )
        for name, values in polar.items()
        if np.asarray(values).ndim == 2
    }

    output: dict[str, float | int] = {
        "quality_slow_time_samples": int(h.shape[0]),
        "quality_range_gates": int(h.shape[1]),
        "quality_finite_fraction": 1.0,
        "time_h_magnitude_mean": float(magnitude_h.mean()),
        "time_h_magnitude_cv": _coefficient_of_variation(magnitude_h, config.eps),
        "time_h_magnitude_kurtosis": _excess_kurtosis(magnitude_h, config.eps),
        "time_v_magnitude_mean": float(magnitude_v.mean()),
        "time_v_magnitude_cv": _coefficient_of_variation(magnitude_v, config.eps),
        "time_v_magnitude_kurtosis": _excess_kurtosis(magnitude_v, config.eps),
        "time_h_lag1_coherence": _lag_one_coherence(h_series, config.eps),
        "time_v_lag1_coherence": _lag_one_coherence(v_series, config.eps),
        "time_hv_coherence": _normalized_complex_coherence(
            h_series, v_series, config.eps
        ),
        "time_hv_phase_resultant": _phase_resultant(
            h_series * np.conj(v_series), config.eps
        ),
        "time_hv_power_ratio_db": float(
            10.0
            * np.log10(
                (float(np.sum(np.abs(h_series) ** 2)) + config.eps)
                / (float(np.sum(np.abs(v_series) ** 2)) + config.eps)
            )
        ),
        "rd_peak_velocity_index": int(peak_velocity),
        "rd_peak_range_index": int(peak_range),
        "rd_peak_velocity_offset": int(peak_velocity - center),
        "rd_peak_at_zero_band": int(
            abs(peak_velocity - center) <= config.zero_doppler_radius
        ),
        "rd_peak_at_edge_band": int(
            peak_velocity < edge or peak_velocity >= combined.shape[0] - edge
        ),
        "rd_anchor_velocity_index": anchor_velocity,
        "rd_anchor_range_index": anchor_range,
        "rd_anchor_is_external_candidate": int(supplied[0]),
        "rd_anchor_matches_scene_peak": int(
            anchor_velocity == peak_velocity and anchor_range == peak_range
        ),
        "rd_total_energy_db": float(10.0 * np.log10(total_energy)),
        "rd_global_peak_fraction": float(combined[peak_velocity, peak_range] / total_energy),
        "rd_zero_doppler_energy_fraction": float(
            combined[zero_start:zero_end].sum() / total_energy
        ),
        "rd_edge_doppler_energy_fraction": edge_energy / total_energy,
        "polar_roi_zdr_median_db": float(np.median(roi["zdr_like_db"])),
        "polar_roi_zdr_iqr_db": float(
            np.quantile(roi["zdr_like_db"], 0.75)
            - np.quantile(roi["zdr_like_db"], 0.25)
        ),
        "polar_roi_rho_mean": float(np.mean(roi["rho_hv_local"])),
        "polar_roi_rho_p10": float(np.quantile(roi["rho_hv_local"], 0.10)),
        "polar_roi_phase_resultant": _phase_resultant(
            roi["cross_local"], config.eps
        ),
        "polar_roi_stokes_s1_mean": float(np.mean(roi["stokes_s1"])),
        "polar_roi_stokes_s2_mean": float(np.mean(roi["stokes_s2"])),
        "polar_roi_stokes_s3_mean": float(np.mean(roi["stokes_s3"])),
    }
    output.update(_spectral_features(anchor_spectrum, anchor_velocity, config))
    output.update(_time_frequency_features(h_series, v_series, config))
    values = np.asarray(list(output.values()), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("multi-domain feature vector contains NaN or Inf")
    return output
