import logging

import numpy as np
import scipy as sp

from .pt import check_waveform_polarity
from .signal import angle_dependant_filtering
from .sobi import sobi

logger = logging.getLogger(__name__)


def _get_param(params: dict, key: str, default):
    return params[key] if key in params else default


def _normalize_channels(sig: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sig = sig - np.mean(sig, axis=0, keepdims=True)
    scale = np.std(sig, axis=0, keepdims=True)
    return sig / np.maximum(scale, eps)


def _sos_filter(sig: np.ndarray, f_samp: float, btype: str, freqs, order: int = 4) -> np.ndarray:
    nyquist = f_samp / 2
    if btype in ('highpass', 'lowpass'):
        wn = freqs / nyquist
    else:
        wn = [freqs[0] / nyquist, freqs[1] / nyquist]
    if np.any(np.asarray(wn) <= 0) or np.any(np.asarray(wn) >= 1):
        logger.warning(f"Skipping {btype} filter with invalid cutoff {freqs} Hz for sampling rate {f_samp:.3f} Hz.")
        return sig
    sos = sp.signal.butter(order, wn, btype=btype, output='sos')
    padlen = min(sig.shape[0] - 1, 3 * (2 * sos.shape[0] + 1))
    return sp.signal.sosfiltfilt(sos, sig, axis=0, padlen=padlen)


def _bandpower_ratio(latent_vectors: np.ndarray, f_samp: float, signal_band: tuple[float, float],
                     total_band: tuple[float, float] = (0.05, 5.0), eps: float = 1e-12) -> np.ndarray:
    freqs = np.fft.rfftfreq(latent_vectors.shape[0], d=1 / f_samp)
    spectrum = np.abs(np.fft.rfft(latent_vectors, axis=0)) ** 2
    signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    total_mask = (freqs >= total_band[0]) & (freqs <= total_band[1])
    signal_power = np.sum(spectrum[signal_mask, :], axis=0)
    total_power = np.sum(spectrum[total_mask, :], axis=0)
    return signal_power / np.maximum(total_power, eps)


def _extract_pca_sources(sig: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    _, _, vt = np.linalg.svd(sig, full_matrices=False)
    n_components = min(n_components, vt.shape[0])
    sources = vt[:n_components, :]
    mixing = sig.T @ sources.T
    return sources, mixing


def _edge_triggers(time_pt: np.ndarray, beat_waveform: np.ndarray, edge_waveform: np.ndarray,
                   interp_factor: int = 4, skip_time: float = 0.5, max_hr: float = 160,
                   prominence: float = 0.4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    dt = time_pt[1] - time_pt[0]
    time_fine = np.arange(time_pt[0], time_pt[-1] + dt / interp_factor, dt / interp_factor)
    beat_fine = np.interp(time_fine, time_pt, beat_waveform)
    edge_fine = np.interp(time_fine, time_pt, edge_waveform)
    edge_derivative_fine = np.hstack((0, np.diff(edge_fine) / (time_fine[1] - time_fine[0])))

    min_distance = int(np.ceil((60 / max_hr) / (time_fine[1] - time_fine[0])))
    beat_locs, _ = sp.signal.find_peaks(beat_fine[time_fine > skip_time], prominence=prominence, distance=min_distance)
    beat_locs += np.sum(time_fine <= skip_time)

    if beat_locs.size == 0:
        return (
            np.zeros(time_pt.shape, dtype=np.uint32),
            np.zeros(time_pt.shape, dtype=np.uint32),
            np.zeros(time_pt.shape),
            np.asarray([], dtype=int),
            'none',
        )

    beat_orig_locs = np.searchsorted(time_pt, time_fine[beat_locs])
    beat_orig_locs = np.clip(beat_orig_locs, 0, time_pt.shape[0] - 1)
    beat_orig_locs = np.unique(beat_orig_locs)
    beat_triggers = np.zeros(time_pt.shape, dtype=np.uint32)
    beat_triggers[beat_orig_locs] = 1

    half_window = max(1, min_distance // 2)
    rising_locs = []
    falling_locs = []
    rising_strength = []
    falling_strength = []
    for loc in beat_locs:
        lo = max(0, loc - half_window)
        hi = min(edge_derivative_fine.shape[0], loc + half_window + 1)
        window = edge_derivative_fine[lo:hi]
        if window.size == 0:
            continue
        rise = lo + int(np.argmax(window))
        fall = lo + int(np.argmin(window))
        rising_locs.append(rise)
        falling_locs.append(fall)
        rising_strength.append(edge_derivative_fine[rise])
        falling_strength.append(abs(edge_derivative_fine[fall]))

    if len(rising_locs) == 0:
        return (
            beat_triggers,
            np.zeros(time_pt.shape, dtype=np.uint32),
            np.interp(time_pt, time_fine, edge_derivative_fine),
            np.asarray([], dtype=int),
            'none',
        )

    if np.mean(rising_strength) >= np.mean(falling_strength):
        fine_locs = np.asarray(rising_locs)
        edge = 'rising'
    else:
        fine_locs = np.asarray(falling_locs)
        edge = 'falling'

    orig_locs = np.searchsorted(time_pt, time_fine[fine_locs])
    orig_locs = np.clip(orig_locs, 0, time_pt.shape[0] - 1)
    orig_locs = np.unique(orig_locs)

    triggers = np.zeros(time_pt.shape, dtype=np.uint32)
    triggers[orig_locs] = 1
    edge_derivative = np.interp(time_pt, time_fine, edge_derivative_fine)
    edge_derivative = edge_derivative - np.percentile(edge_derivative, 10)
    scale = np.percentile(edge_derivative, 98)
    if abs(scale) > 1e-12:
        edge_derivative = edge_derivative / scale

    return beat_triggers, triggers, edge_derivative, orig_locs, edge


def extract_selfnav_navs(ksp_measured: np.ndarray, center_sample_idx: int, n_unique_angles: int,
                         f_samp: float, params: dict) -> dict:
    '''Extract cardiac and respiratory self-navigation waveforms from spiral k-space center samples.'''
    observables = _get_param(params, 'observables', ('magnitude', 'phase'))
    drift_highpass_hz = _get_param(params, 'drift_highpass_hz', 0.05)
    pre_bss_lowpass_hz = _get_param(params, 'pre_bss_lowpass_hz', 5.0)
    separation_method = _get_param(params, 'separation_method', 'sobi').lower()
    pca_components = _get_param(params, 'pca_components', 5)
    num_lags = _get_param(params, 'num_lags', 375)
    cardiac_score_band = tuple(_get_param(params, 'cardiac_score_band_hz', (0.5, 3.5)))
    cardiac_beat_band = tuple(_get_param(params, 'cardiac_beat_band_hz', (0.5, 2.0)))
    cardiac_edge_band = tuple(_get_param(params, 'cardiac_edge_band_hz', (0.5, 5.0)))
    respiratory_band = tuple(_get_param(params, 'respiratory_band_hz', (0.1, 0.6)))
    interp_factor = int(_get_param(params, 'interp_factor', 4))
    skip_time = _get_param(params, 'skip_time', 0.5)
    max_hr = _get_param(params, 'max_hr', 160)
    prominence = _get_param(params, 'prominence', 0.4)
    apply_angle_filter = _get_param(params, 'trajectory_filter', True)

    k0 = ksp_measured[center_sample_idx, :, :]
    candidates = []
    candidate_labels = []
    if 'magnitude' in observables:
        candidates.append(np.abs(k0))
        candidate_labels.extend([f'magnitude_ch{ii}' for ii in range(k0.shape[1])])
    if 'phase' in observables:
        candidates.append(np.unwrap(np.angle(k0), axis=0))
        candidate_labels.extend([f'phase_ch{ii}' for ii in range(k0.shape[1])])
    if 'real' in observables:
        candidates.append(np.real(k0))
        candidate_labels.extend([f'real_ch{ii}' for ii in range(k0.shape[1])])
    if 'imag' in observables:
        candidates.append(np.imag(k0))
        candidate_labels.extend([f'imag_ch{ii}' for ii in range(k0.shape[1])])

    if len(candidates) == 0:
        raise ValueError("At least one self-navigation observable must be requested.")

    raw_signal = np.concatenate(candidates, axis=1)
    keep = np.std(raw_signal, axis=0) > 1e-12
    raw_signal = raw_signal[:, keep]
    candidate_labels = np.asarray(candidate_labels)[keep]

    processed_signal = raw_signal - np.mean(raw_signal, axis=0, keepdims=True)
    if drift_highpass_hz is not None and drift_highpass_hz > 0:
        processed_signal = _sos_filter(processed_signal, f_samp, 'highpass', drift_highpass_hz, order=2)
    if apply_angle_filter:
        processed_signal = angle_dependant_filtering(processed_signal, n_unique_angles)
    if pre_bss_lowpass_hz is not None and pre_bss_lowpass_hz > 0:
        processed_signal = _sos_filter(processed_signal, f_samp, 'lowpass', pre_bss_lowpass_hz, order=4)
    processed_signal = _normalize_channels(processed_signal)

    if separation_method == 'sobi':
        sobi_lags = min(num_lags, max(1, processed_signal.shape[0] // 2))
        if sobi_lags != num_lags:
            logger.warning(f"Using {sobi_lags} SOBI lags instead of requested {num_lags} for {processed_signal.shape[0]} samples.")
        sources, mixing, unmixing = sobi(processed_signal.T, num_lags=sobi_lags)
    elif separation_method == 'pca':
        sources, mixing = _extract_pca_sources(processed_signal, pca_components)
        unmixing = None
    else:
        raise ValueError(f"Unsupported self-navigation separation method: {separation_method}")

    source_matrix = sources.T
    cardiac_scores = _bandpower_ratio(source_matrix, f_samp, cardiac_score_band)
    respiratory_scores = _bandpower_ratio(source_matrix, f_samp, respiratory_band, total_band=(0.05, 2.0))
    cardiac_idx = int(np.argmax(cardiac_scores))
    respiratory_idx = int(np.argmax(respiratory_scores))

    cardiac_source = source_matrix[:, cardiac_idx]
    respiratory_source = source_matrix[:, respiratory_idx]
    cardiac = _sos_filter(cardiac_source, f_samp, 'bandpass', cardiac_beat_band, order=6)
    sign_check = cardiac[40:] if cardiac.shape[0] > 80 else cardiac
    if np.ptp(sign_check) > 1e-12:
        cardiac = check_waveform_polarity(sign_check, method='width') * cardiac
    edge_waveform = _sos_filter(cardiac_source, f_samp, 'bandpass', cardiac_edge_band, order=6)
    respiratory = _sos_filter(respiratory_source, f_samp, 'bandpass', respiratory_band, order=4)

    cardiac = cardiac - np.percentile(cardiac, 10)
    cardiac_scale = np.percentile(cardiac, 95)
    if abs(cardiac_scale) > 1e-12:
        cardiac = cardiac / cardiac_scale
    respiratory = respiratory - np.mean(respiratory)
    respiratory_scale = np.max(np.abs(respiratory))
    if respiratory_scale > 1e-12:
        respiratory = respiratory / respiratory_scale

    time_pt = np.arange(cardiac.shape[0]) / f_samp
    beat_triggers, edge_triggers, cardiac_derivative, trigger_locs, trigger_edge = _edge_triggers(
        time_pt, cardiac, edge_waveform, interp_factor=interp_factor,
        skip_time=skip_time, max_hr=max_hr, prominence=prominence
    )

    return {
        'raw_signal': raw_signal,
        'processed_signal': processed_signal,
        'candidate_labels': candidate_labels,
        'sources': sources,
        'mixing': mixing,
        'unmixing': unmixing,
        'cardiac_scores': cardiac_scores,
        'respiratory_scores': respiratory_scores,
        'cardiac_source_index': cardiac_idx,
        'respiratory_source_index': respiratory_idx,
        'respiratory': respiratory,
        'cardiac': cardiac,
        'cardiac_derivative': cardiac_derivative,
        'cardiac_triggers': beat_triggers,
        'edge_triggers': edge_triggers,
        'cardiac_trigger_locs': trigger_locs,
        'trigger_edge': trigger_edge,
    }
