from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


def beat_rejection(pktimes, padloc, pkamps=None):
    '''Rejects the "bad beats", that are 3*std away from the mean
    heart rate, and optionally peaks that have amplitude at least 3*std away 
    from the mean amplitude.
    '''
    hr_perpeak = 60./np.diff(pktimes)
    hr_variation = np.std(hr_perpeak)
    hr_mean = np.mean(hr_perpeak)
    hr_diffs = hr_perpeak - hr_mean
    if padloc == "pre":
        hr_accept_list = np.hstack((1, abs(hr_diffs)<(3*hr_variation)))
    elif padloc == "post":
        hr_accept_list = np.hstack((abs(hr_diffs)<(3*hr_variation), 1))
    
    if pkamps is not None:
        mean_ptpk = np.mean(pkamps(hr_accept_list))
        ptpk_variation = np.std(pkamps(hr_accept_list))
        hr_accept_list = hr_accept_list & (np.abs(pkamps-mean_ptpk) < 3*ptpk_variation)
    
    return hr_accept_list


def prepeak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs):
    # Somewhat working immediately preceding peak matching algo.

    t_first_ecg_pk = time_ecg[ecg_peak_locs[0]]
    idx_first_pt_after_ecg = np.nonzero(time_pt[pt_cardiac_peak_locs] > t_first_ecg_pk)[0][0]
    pt_peaks_selected = pt_cardiac_peak_locs[idx_first_pt_after_ecg:]
    ecg_peaks_selected = ecg_peak_locs[:len(pt_peaks_selected)]

    # # Reject bad beats
    # ptPeaksSelected = ptPeaksSelected(hrAcceptList(idx_first_pt_after_ecg:end))
    # ecgPeaksSelected = ecgPeaksSelected(hrAcceptList(idx_first_pt_after_ecg:end))

    peak_diff = time_pt[pt_peaks_selected] - time_ecg[ecg_peaks_selected]
    return peak_diff, pt_peaks_selected

def interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs):

    # Iterate over ECG peaks, and check if there is a PT peak between current and next ECG peak.
    pt_trig_wf = np.zeros(time_pt.shape, dtype=int)
    pt_trig_wf[pt_cardiac_peak_locs] = 1
    peak_diff = []
    extra_pk_idx = []
    miss_pk_idx = []

    n_ecg_pk = ecg_peak_locs.shape[0]
    for pk_i in range(n_ecg_pk-1):
        curr_pk_t = time_ecg[ecg_peak_locs[pk_i]]
        next_pk_t = time_ecg[ecg_peak_locs[pk_i+1]]

        masked_pt_trig = pt_trig_wf & ((time_pt > curr_pk_t) & (time_pt < next_pk_t))

        n_trig_per_trig = np.sum(masked_pt_trig)
        if n_trig_per_trig == 0:
            # Missed beat, nothing to do, move on.
            miss_pk_idx.append(pk_i)
            continue
        elif n_trig_per_trig > 1:
            # Extraneous trigger, not cool. Mark, but still accept the first one.
            extra_pk_idx.append(pk_i)
        
        pt_peak_idx = np.nonzero(masked_pt_trig)[0][0]
        peak_diff.append(time_pt[pt_peak_idx] - curr_pk_t)

    return np.asarray(peak_diff), np.asarray(miss_pk_idx), np.asarray(extra_pk_idx)

def extract_triggers(time_pt, cardiac_waveform, skip_time=0.6, prominence=0.4, max_hr=120):
    ''' Extract triggers from the cardiac waveform.
        Parameters:
            time_pt: np.array
                Time points for the cardiac waveform.
            cardiac_waveform: np.array
                Cardiac waveform.
            skip_time: float
                Time to skip at the beginning of the waveform.
        Returns:
            pt_cardiac_trigs: np.array
                Trigger waveform.
    '''
    dt_pt = (time_pt[1] - time_pt[0])
    Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
    pt_cardiac_peak_locs,_ = find_peaks(cardiac_waveform[time_pt > skip_time], prominence=prominence, distance=Dmin)
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)
    pt_peaks_selected = pt_cardiac_peak_locs
    n_acq = time_pt.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1

    return pt_cardiac_trigs

def pt_ecg_jitter(time_pt, pt_cardiac, pt_cardiac_derivative, time_ecg, ecg_waveform, pt_cardiac_trigs=None, pt_derivative_trigs=None, ecg_trigs=None, skip_time=0.6, max_hr=120, show_outputs=True, debug_output_dir=None): 
    """ 
    This function calculates the jitter between the pilot tone and the ECG triggers.
    The ECG triggers are assumed to be correct. The PT triggers are assumed to be correct.
    The peak locations of the PT and ECG are found and matched. The jitter is calculated from the differences between the matched peaks.

    Parameters
    ----------
    time_pt : numpy array
        Time axis of the PT waveform in seconds.
    pt_cardiac : numpy array
        PT waveform.
    pt_cardiac_derivative : numpy array
        Derivative of the PT waveform.
    time_ecg : numpy array
        Time axis of the ECG waveform in seconds.
    ecg_waveform : numpy array
        ECG waveform.
    pt_cardiac_trigs : numpy array, optional
        PT triggers. If None, they are calculated.
    pt_derivative_trigs : numpy array, optional
        Derivative PT triggers. If None, they are calculated.
    ecg_trigs : numpy array, optional
        ECG triggers. If None, they are calculated.
    skip_time : float, optional
        Time to skip at the beginning of the waveforms in seconds. The default is 0.6.
    max_hr : int, optional
        Maximum heart rate in bpm. The default is 120.
    show_outputs : bool, optional
        Whether to show the outputs. The default is True.

    Returns
    -------
    peak_diff : numpy array
        Differences between the matched peaks.
    derivative_peak_diff : numpy array
        Differences between the matched derivative peaks.

    """

    # ECG Triggers
    if ecg_trigs is None:
        ecg_peak_locs,_ = find_peaks(ecg_waveform[time_ecg > skip_time], prominence=0.7)
    else:
        ecg_peak_locs = np.nonzero(ecg_trigs[time_ecg > skip_time])[0]
    ecg_peak_locs += np.sum(time_ecg <= skip_time)

    # PT Triggers
    dt_pt = (time_pt[1] - time_pt[0])
    
    if pt_cardiac_trigs is None:
        Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
        pt_cardiac_peak_locs,_ = find_peaks(pt_cardiac[time_pt > skip_time], prominence=0.4, distance=Dmin)
    else:
        pt_cardiac_peak_locs = np.nonzero(pt_cardiac_trigs[time_pt > skip_time])[0]
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)

    # PT Derivative Triggers
    if pt_derivative_trigs is None:
        pt_cardiac_derivative_peak_locs,_ = find_peaks(pt_cardiac_derivative[time_pt > skip_time], prominence=0.6, distance=Dmin)
    else:
        pt_cardiac_derivative_peak_locs = np.nonzero(pt_derivative_trigs[time_pt > skip_time])[0]
    pt_cardiac_derivative_peak_locs += np.sum(time_pt <= skip_time)

    # "Arryhtmia detection" by heart rate variation
    hr_accept_list = beat_rejection(pt_cardiac_peak_locs*dt_pt, "post")
    hr_accept_list_derivative = beat_rejection(pt_cardiac_derivative_peak_locs*dt_pt, "pre")
    # TODO: Is pre post even correct? Why does it change? Need to investigate.

    # peak_diff, pt_peaks_selected = prepeak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    # derivative_peak_diff, pt_derivative_peaks_selected = prepeak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)

    peak_diff, miss_pks, extra_pks = interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    pt_peaks_selected = pt_cardiac_peak_locs

    derivative_peak_diff, derivative_miss_pks, derivative_extra_pks = interval_peak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)
    pt_derivative_peaks_selected = pt_cardiac_derivative_peak_locs

    # Create trigger waveforms from peak locations.
    n_acq = pt_cardiac.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_derivative_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1
    pt_derivative_trigs[pt_derivative_peaks_selected] = 1

    emit_outputs = show_outputs or debug_output_dir is not None
    if emit_outputs:
        # Print some useful info

        print(f'Rejection ratio for pt peaks is {100*(len(hr_accept_list) - np.sum(hr_accept_list))/len(hr_accept_list):.2f} percent.\n')
        print(f'Rejection ratio for derivative pt peaks is {100*(len(hr_accept_list_derivative) - np.sum(hr_accept_list_derivative))/len(hr_accept_list_derivative):.2f} percent.\n')

        print(f'Peak difference {np.mean(peak_diff*1e3):.1f} \u00B1 {np.std(peak_diff*1e3):.1f}')
        print(f'Derivative peak difference {np.mean(derivative_peak_diff*1e3):.1f} \u00B1 {np.std(derivative_peak_diff*1e3):.1f}')

        print(f'Number of ECG triggers: {ecg_peak_locs.shape[0]}.')
        print(f'Number of PT triggers: {pt_cardiac_peak_locs.shape[0]}.')
        print(f'Number of missed PT triggers: {miss_pks.shape[0]}.')
        print(f'Number of extraneous PT triggers: {extra_pks.shape[0]}.')
        print(f'Number of derivative PT triggers: {pt_cardiac_derivative_peak_locs.shape[0]}.')
        print(f'Number of missed derivative PT triggers: {derivative_miss_pks.shape[0]}.')
        print(f'Number of extraneous derivative PT triggers: {derivative_extra_pks.shape[0]}.')

        import matplotlib.pyplot as plt
        # Plots
        fig_overview = plt.figure()
        plt.plot(time_ecg, ecg_waveform)
        plt.plot(time_ecg[ecg_trigs==1], ecg_waveform[ecg_trigs==1], '*')
        plt.plot(time_pt, pt_cardiac_trigs, 'x', label='PT Triggers')

        fig_jitter, axs = plt.subplots(2,2, sharex='col')
        axs[0,0].plot(time_pt, pt_cardiac, '-gD', markevery=pt_cardiac_peak_locs, label='Pilot Tone')
        axs[0,0].plot(time_ecg, ecg_waveform, '-bs', markevery=ecg_peak_locs, label='ECG')
        axs[0,0].set_xlabel('Time [s]')
        axs[0,0].legend()
        axs[0,0].set_title('ECG and Pilot Tone. Markers show triggers.')

        axs[0,1].hist((peak_diff - np.mean(peak_diff))*1e3)
        axs[0,1].set_xlabel('Time diff [ms]')
        axs[0,1].set_ylabel('Number of peaks')

        axs[1,0].plot(time_pt, pt_cardiac_derivative, '-gD', markevery=pt_cardiac_derivative_peak_locs, label='Pilot Tone')
        axs[1,0].plot(time_ecg, ecg_waveform, '-bs', markevery=ecg_peak_locs, label='ECG')
        axs[1,0].set_xlabel('Time [s]')
        axs[1,0].legend()
        axs[1,0].set_title('ECG and Inverse Derivative Pilot Tone. Markers show triggers.')

        axs[1,1].hist((derivative_peak_diff - np.mean(derivative_peak_diff))*1e3)
        axs[1,1].set_xlabel('Time diff [ms]')
        axs[1,1].set_ylabel('Number of peaks')

        if debug_output_dir is not None:
            debug_path = Path(debug_output_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            for fig, name in (
                (fig_overview, 'ecg_pt_trigger_overview'),
                (fig_jitter, 'ecg_pt_jitter'),
            ):
                fig.savefig(debug_path / f'{name}.png', dpi=200, bbox_inches='tight')
                fig.savefig(debug_path / f'{name}.svg', bbox_inches='tight')

        if show_outputs:
            plt.show()
    
    return peak_diff, derivative_peak_diff

def calculate_jitter(time_pt, pt_cardiac, time_ecg, ecg_waveform, pt_cardiac_trigs=None, ecg_trigs=None, skip_time=0.6, peak_prominence=0.4, max_hr=120): 
    """ 
    This function calculates the jitter between the pilot tone and the ECG triggers.
    The ECG triggers are assumed to be correct. The PT triggers are assumed to be correct.
    The peak locations of the PT and ECG are found and matched. The jitter is calculated from the differences between the matched peaks.

    Parameters
    ----------
    time_pt : numpy array
        Time axis of the PT waveform in seconds.
    pt_cardiac : numpy array
        PT waveform.
    time_ecg : numpy array
        Time axis of the ECG waveform in seconds.
    ecg_waveform : numpy array
        ECG waveform.
    pt_cardiac_trigs : numpy array, optional
        PT triggers. If None, they are calculated.
    ecg_trigs : numpy array, optional
        ECG triggers. If None, they are calculated.
    skip_time : float, optional
        Time to skip at the beginning of the waveforms in seconds. The default is 0.6.
    peak_prominence : float, optional
        Prominence of the pilot tone peaks. The default is 0.4.
    max_hr : int, optional
        Maximum heart rate in bpm. The default is 120.

    Returns
    -------
    peak_diff : numpy array
        Differences between the matched peaks.
    miss_pks : numpy array
        False negative PT triggers.
    extra_pks : numpy array
        False positive PT triggers.
    """

    # ECG Triggers
    if ecg_trigs is None:
        ecg_peak_locs,_ = find_peaks(ecg_waveform[time_ecg > skip_time], prominence=0.7)
    else:
        ecg_peak_locs = np.nonzero(ecg_trigs[time_ecg > skip_time])[0]
    ecg_peak_locs += np.sum(time_ecg <= skip_time)

    # PT Triggers
    dt_pt = (time_pt[1] - time_pt[0])
    
    if pt_cardiac_trigs is None:
        Dmin = int(np.ceil((60/max_hr)/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
        pt_cardiac_peak_locs,_ = find_peaks(pt_cardiac[time_pt > skip_time], prominence=peak_prominence, distance=Dmin)
    else:
        pt_cardiac_peak_locs = np.nonzero(pt_cardiac_trigs[time_pt > skip_time])[0]
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)

    peak_diff, miss_pks, extra_pks = interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    pt_peaks_selected = pt_cardiac_peak_locs

    # Create trigger waveforms from peak locations.
    n_acq = pt_cardiac.shape[0]
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1
    
    return peak_diff, miss_pks, extra_pks


def repair_cardiac_triggers_rr(time_cardiac, cardiac_triggers, params=None):
    """Repair cardiac trigger gaps using local RR consistency only.

    Close triggers are removed using a maximum heart-rate threshold. Long gaps
    are then filled by estimating a local median RR and uniformly distributing
    the expected missing triggers inside each gap.

    Parameters
    ----------
    time_cardiac : array_like
        Cardiac trigger time axis in seconds.
    cardiac_triggers : array_like
        Binary or thresholdable cardiac trigger waveform on ``time_cardiac``.
    params : dict, optional
        Repair parameters. Recognized keys include ``max_hr_bpm``,
        ``min_separation_s``, ``local_rr_window``, ``missing_rr_factor``,
        ``missing_abs_floor_s``, ``max_missing_per_gap``, and
        ``trigger_threshold``.

    Returns
    -------
    repaired_triggers : ndarray
        Binary repaired cardiac trigger waveform on ``time_cardiac``.
    stats : dict
        Summary statistics and event details, including input/observed/final
        trigger counts, expected trigger count, inserted trigger times, removed
        close-trigger count, RR/HR summaries, and per-gap repair events.
    """
    params = {} if params is None else dict(params)
    time_cardiac = np.asarray(time_cardiac, dtype=float)
    cardiac_triggers = np.asarray(cardiac_triggers)

    if time_cardiac.ndim != 1 or cardiac_triggers.ndim != 1:
        raise ValueError("time_cardiac and cardiac_triggers must be one-dimensional.")
    if time_cardiac.shape[0] != cardiac_triggers.shape[0]:
        raise ValueError("cardiac_triggers must have the same length as time_cardiac.")
    if time_cardiac.size < 3:
        raise ValueError("At least three time points are required.")

    max_hr_bpm = float(params.get("max_hr_bpm", params.get("max_hr", 160.0)))
    min_separation_s = float(params.get("min_separation_s", 60.0 / max_hr_bpm))
    local_rr_window = int(params.get("local_rr_window", 11))
    missing_rr_factor = float(params.get("missing_rr_factor", 1.25))
    missing_abs_floor_s = float(params.get("missing_abs_floor_s", 0.18))
    max_missing_per_gap = int(params.get("max_missing_per_gap", 8))
    trigger_threshold = float(params.get("trigger_threshold", 0.5))

    trigger_times_raw = np.asarray(time_cardiac[cardiac_triggers > trigger_threshold], dtype=float)
    cleaned_trigger_times = []
    for trig_t in trigger_times_raw:
        if not cleaned_trigger_times or trig_t - cleaned_trigger_times[-1] >= min_separation_s:
            cleaned_trigger_times.append(float(trig_t))
    observed_trigger_times = np.asarray(cleaned_trigger_times, dtype=float)

    events = []
    inserted_times = []
    repaired_trigger_times = observed_trigger_times.copy()
    if observed_trigger_times.size >= 4:
        ecg_rr = np.diff(observed_trigger_times)
        half_window = max(1, local_rr_window // 2)
        for gap_i, gap_duration in enumerate(ecg_rr):
            prev_t = observed_trigger_times[gap_i]
            next_t = observed_trigger_times[gap_i + 1]

            rr_lo = max(0, gap_i - half_window)
            rr_hi = min(ecg_rr.size, gap_i + half_window + 1)
            rr_block = np.delete(ecg_rr[rr_lo:rr_hi], gap_i - rr_lo)
            if rr_block.size == 0:
                rr_block = ecg_rr
            expected_rr = float(np.median(rr_block))

            n_intervals_expected = int(np.round(gap_duration / expected_rr))
            n_missing = int(np.clip(n_intervals_expected - 1, 0, max_missing_per_gap))
            is_long_gap = (
                n_missing > 0
                and gap_duration > missing_rr_factor * expected_rr
                and (gap_duration - expected_rr) > missing_abs_floor_s
            )
            if not is_long_gap:
                continue

            gap_inserted_times = np.linspace(prev_t, next_t, n_missing + 2)[1:-1]
            gap_inserted_times = gap_inserted_times[
                (gap_inserted_times > prev_t + min_separation_s)
                & (gap_inserted_times < next_t - min_separation_s)
            ]
            inserted_times.extend([float(t) for t in gap_inserted_times])
            events.append({
                "kind": "uniformly_inserted_missing_cardiac_trigger",
                "gap_index": int(gap_i),
                "prev_trigger_s": float(prev_t),
                "next_trigger_s": float(next_t),
                "gap_duration_s": float(gap_duration),
                "expected_rr_s": float(expected_rr),
                "n_missing": int(n_missing),
                "inserted_trigger_s": [float(t) for t in gap_inserted_times],
            })

        if inserted_times:
            repaired_trigger_times = np.sort(
                np.concatenate([observed_trigger_times, np.asarray(inserted_times)])
            )

    repaired_triggers = np.zeros(time_cardiac.shape, dtype=np.uint32)
    for trig_t in repaired_trigger_times:
        if trig_t < time_cardiac[0] or trig_t > time_cardiac[-1]:
            continue
        repaired_triggers[int(np.argmin(np.abs(time_cardiac - trig_t)))] = 1

    rr_observed = np.diff(observed_trigger_times)
    rr_repaired = np.diff(time_cardiac[repaired_triggers == 1])
    hr_observed_bpm = 60.0 / rr_observed if rr_observed.size else np.asarray([])
    hr_repaired_bpm = 60.0 / rr_repaired if rr_repaired.size else np.asarray([])
    stats = {
        "n_input_triggers": int(trigger_times_raw.size),
        "n_observed_triggers": int(observed_trigger_times.size),
        "n_close_triggers_removed": int(trigger_times_raw.size - observed_trigger_times.size),
        "n_total_expected_triggers": int(observed_trigger_times.size + sum(event["n_missing"] for event in events)),
        "n_repaired_triggers": len(inserted_times),
        "n_final_triggers": int(np.sum(repaired_triggers)),
        "min_separation_s": float(min_separation_s),
        "max_hr_bpm": float(max_hr_bpm),
        "rr_mean_observed_s": float(np.mean(rr_observed)) if rr_observed.size else np.nan,
        "rr_median_observed_s": float(np.median(rr_observed)) if rr_observed.size else np.nan,
        "rr_std_observed_s": float(np.std(rr_observed)) if rr_observed.size else np.nan,
        "rr_mean_repaired_s": float(np.mean(rr_repaired)) if rr_repaired.size else np.nan,
        "rr_median_repaired_s": float(np.median(rr_repaired)) if rr_repaired.size else np.nan,
        "rr_std_repaired_s": float(np.std(rr_repaired)) if rr_repaired.size else np.nan,
        "hr_mean_observed_bpm": float(np.mean(hr_observed_bpm)) if hr_observed_bpm.size else np.nan,
        "hr_median_observed_bpm": float(np.median(hr_observed_bpm)) if hr_observed_bpm.size else np.nan,
        "hr_mean_repaired_bpm": float(np.mean(hr_repaired_bpm)) if hr_repaired_bpm.size else np.nan,
        "hr_median_repaired_bpm": float(np.median(hr_repaired_bpm)) if hr_repaired_bpm.size else np.nan,
        "observed_trigger_times_s": [float(t) for t in observed_trigger_times],
        "inserted_trigger_times_s": [float(t) for t in inserted_times],
        "final_trigger_times_s": [float(t) for t in time_cardiac[repaired_triggers == 1]],
        "events": events,
    }

    return repaired_triggers, stats


def repair_ecg_triggers_with_pt(time_ecg, ecg_waveform, pt_waveform, params=None):
    """Repair missing or extra ECG triggers using pilot-tone cardiac peaks.

    The ECG waveform is interpreted as a trigger waveform. Missing beats are
    filled greedily from PT peaks using a local ECG-to-PT delay estimate, and
    unusually short ECG intervals can be removed when unsupported by PT.

    Parameters
    ----------
    time_ecg : array_like
        ECG trigger time axis in seconds.
    ecg_waveform : array_like
        Binary or thresholdable ECG trigger waveform on ``time_ecg``.
    pt_waveform : array_like
        Pilot-tone cardiac waveform.
    params : dict, optional
        Repair parameters. Use ``time_pt`` to pass the PT time axis when it
        differs from ``time_ecg``. Other recognized keys include
        ``skip_time_s``, ``max_hr_bpm``, ``pt_delay_search_s``,
        ``local_rr_window``, ``missing_rr_factor``, ``extra_rr_factor``,
        ``pt_peak_prominence``, ``min_rescue_prominence``,
        ``pt_search_half_width_s``, and local delay/RR guard settings.

    Returns
    -------
    repaired_triggers : ndarray
        Binary repaired ECG trigger waveform on ``time_ecg``.
    stats : dict
        Summary statistics and event details, including ECG beat count, total
        expected beats, PT-supported repaired beats, removed ECG beats,
        ECG/PT delay statistics, RR/HR summaries, trigger times, and per-event
        repair decisions.
    """
    params = {} if params is None else dict(params)
    time_ecg = np.asarray(time_ecg, dtype=float)
    ecg_waveform = np.asarray(ecg_waveform)
    pt_waveform = np.asarray(pt_waveform, dtype=float)
    time_pt = np.asarray(params.get("time_pt", time_ecg), dtype=float)

    if time_ecg.ndim != 1 or time_pt.ndim != 1:
        raise ValueError("time_ecg and time_pt must be one-dimensional.")
    if ecg_waveform.shape[0] != time_ecg.shape[0]:
        raise ValueError("ecg_waveform must have the same length as time_ecg.")
    if pt_waveform.shape[0] != time_pt.shape[0]:
        raise ValueError("pt_waveform must have the same length as time_pt.")
    if time_ecg.size < 3 or time_pt.size < 3:
        raise ValueError("At least three time points are required.")

    skip_time_s = float(params.get("skip_time_s", 0.1))
    max_hr_bpm = float(params.get("max_hr_bpm", params.get("max_hr", 160.0)))
    min_rr_s = 60.0 / max_hr_bpm
    pt_delay_search_s = tuple(params.get("pt_delay_search_s", (0.15, 0.45)))
    local_rr_window = int(params.get("local_rr_window", 11))
    missing_rr_factor = float(params.get("missing_rr_factor", 1.25))
    extra_rr_factor = float(params.get("extra_rr_factor", 0.80))
    missing_abs_floor_s = float(params.get("missing_abs_floor_s", 0.18))
    extra_abs_floor_s = float(params.get("extra_abs_floor_s", 0.12))
    inserted_beat_shortest_rr_factor = float(params.get("inserted_beat_shortest_rr_factor", 0.70))
    inserted_beat_longest_rr_factor = float(params.get("inserted_beat_longest_rr_factor", 1.35))
    max_missing_per_gap = int(params.get("max_missing_per_gap", 8))
    local_delay_window_s = float(params.get("local_delay_window_s", 12.0))
    min_local_delay_matches = int(params.get("min_local_delay_matches", 5))
    pt_peak_prominence = float(params.get("pt_peak_prominence", 0.35))
    min_rescue_prominence = params.get("min_rescue_prominence", None)
    ecg_trigger_threshold = float(params.get("ecg_trigger_threshold", 0.5))
    min_delay_matches = int(params.get("min_delay_matches", 5))

    dt_pt = float(np.median(np.diff(time_pt)))
    dt_ecg = float(np.median(np.diff(time_ecg)))
    pt_min_peak_distance = max(1, int(np.ceil(min_rr_s / dt_pt)))
    pt_peak_offset = int(np.sum(time_pt <= skip_time_s))
    pt_peak_locs_rel, pt_peak_props = find_peaks(
        pt_waveform[time_pt > skip_time_s],
        prominence=pt_peak_prominence,
        distance=pt_min_peak_distance,
    )
    pt_peak_locs = pt_peak_locs_rel + pt_peak_offset
    pt_peak_times = time_pt[pt_peak_locs]
    pt_peak_prom = pt_peak_props.get("prominences", np.ones_like(pt_peak_times, dtype=float))

    ecg_trigger_mask = np.asarray(ecg_waveform > ecg_trigger_threshold, dtype=bool)
    ecg_times_initial = np.asarray(time_ecg[ecg_trigger_mask], dtype=float)
    if ecg_times_initial.size < 3:
        raise ValueError("At least three ECG triggers are required.")
    if pt_peak_times.size < min_delay_matches:
        raise ValueError("Too few PT peaks to estimate ECG/PT delay.")

    matched_ecg_times = []
    matched_pt_times = []
    matched_pt_prom = []
    used_pt_peak_locs = set()
    target_delay = 0.5 * (pt_delay_search_s[0] + pt_delay_search_s[1])
    for ecg_t in ecg_times_initial:
        if ecg_t <= skip_time_s:
            continue
        candidate_mask = (
            (pt_peak_times >= ecg_t + pt_delay_search_s[0])
            & (pt_peak_times <= ecg_t + pt_delay_search_s[1])
        )
        candidate_indices = np.nonzero(candidate_mask)[0]
        if candidate_indices.size == 0:
            continue
        timing_penalty = np.abs((pt_peak_times[candidate_indices] - ecg_t) - target_delay)
        score = pt_peak_prom[candidate_indices] - 0.05 * timing_penalty / dt_pt
        best_i = candidate_indices[int(np.argmax(score))]
        if int(pt_peak_locs[best_i]) in used_pt_peak_locs:
            continue
        used_pt_peak_locs.add(int(pt_peak_locs[best_i]))
        matched_ecg_times.append(float(ecg_t))
        matched_pt_times.append(float(pt_peak_times[best_i]))
        matched_pt_prom.append(float(pt_peak_prom[best_i]))

    matched_ecg_times = np.asarray(matched_ecg_times)
    matched_pt_times = np.asarray(matched_pt_times)
    matched_pt_prom = np.asarray(matched_pt_prom)
    raw_delays = matched_pt_times - matched_ecg_times
    if raw_delays.size < min_delay_matches:
        raise ValueError("Too few ECG/PT matches to estimate a reliable PT delay.")

    delay_median_0 = float(np.median(raw_delays))
    delay_mad_0 = float(1.4826 * np.median(np.abs(raw_delays - delay_median_0)))
    delay_keep = np.abs(raw_delays - delay_median_0) <= max(0.075, 3.0 * delay_mad_0)
    if not np.any(delay_keep):
        delay_keep = np.ones(raw_delays.shape, dtype=bool)
    ecg_pt_delay_s = float(np.median(raw_delays[delay_keep]))
    ecg_pt_delay_mad_s = float(1.4826 * np.median(np.abs(raw_delays[delay_keep] - ecg_pt_delay_s)))
    ecg_pt_delay_mad_s = max(ecg_pt_delay_mad_s, dt_pt)
    pt_search_half_width_s = float(params.get(
        "pt_search_half_width_s",
        min(0.20, max(0.08, 3.0 * ecg_pt_delay_mad_s)),
    ))
    if min_rescue_prominence is None:
        min_rescue_prominence = max(
            0.20,
            0.5 * float(np.median(matched_pt_prom[delay_keep])) if matched_pt_prom.size else 0.20,
        )
    min_rescue_prominence = float(min_rescue_prominence)

    ecg_times_current = ecg_times_initial.copy()
    rr_initial = np.diff(ecg_times_current)
    half_window = max(1, local_rr_window // 2)
    local_rr = np.zeros_like(rr_initial)
    for rr_i in range(rr_initial.size):
        lo = max(0, rr_i - half_window)
        hi = min(rr_initial.size, rr_i + half_window + 1)
        local_rr[rr_i] = np.median(rr_initial[lo:hi])

    long_gap_mask = (rr_initial > missing_rr_factor * local_rr) & ((rr_initial - local_rr) > missing_abs_floor_s)
    short_gap_mask = (rr_initial < extra_rr_factor * local_rr) & ((local_rr - rr_initial) > extra_abs_floor_s)
    long_gap_indices = np.nonzero(long_gap_mask)[0]
    short_gap_indices = np.nonzero(short_gap_mask)[0]
    reliable_rr_mask = ~(long_gap_mask | short_gap_mask)
    reliable_rr_global = (
        float(np.median(rr_initial[reliable_rr_mask]))
        if np.any(reliable_rr_mask)
        else float(np.median(rr_initial))
    )

    events = []
    inserted_times = []
    delay_match_times = matched_ecg_times[delay_keep]
    delay_match_values = raw_delays[delay_keep]
    for gap_i in long_gap_indices:
        prev_t = float(ecg_times_current[gap_i])
        next_t = float(ecg_times_current[gap_i + 1])
        gap_duration = next_t - prev_t
        rr_lo = max(0, gap_i - local_rr_window)
        rr_hi = min(rr_initial.size, gap_i + local_rr_window + 1)
        nearby_reliable_rr = rr_initial[rr_lo:rr_hi][reliable_rr_mask[rr_lo:rr_hi]]
        expected_rr = float(np.median(nearby_reliable_rr)) if nearby_reliable_rr.size else reliable_rr_global
        n_missing_raw = int(np.round(gap_duration / expected_rr)) - 1
        n_missing = int(np.clip(n_missing_raw, 1, max_missing_per_gap))
        gap_rr = gap_duration / (n_missing + 1)

        gap_center_t = 0.5 * (prev_t + next_t)
        local_delay_mask = np.abs(delay_match_times - gap_center_t) <= local_delay_window_s
        if np.sum(local_delay_mask) >= min_local_delay_matches:
            local_delay_s = float(np.mean(delay_match_values[local_delay_mask]))
            local_delay_mad_s = float(1.4826 * np.median(np.abs(delay_match_values[local_delay_mask] - local_delay_s)))
            local_delay_source = "nearby_mean_matches"
        else:
            local_delay_s = ecg_pt_delay_s
            local_delay_mad_s = ecg_pt_delay_mad_s
            local_delay_source = "global_fallback"
        local_delay_mad_s = max(local_delay_mad_s, dt_pt)

        rr_lower_s = max(min_rr_s, inserted_beat_shortest_rr_factor * gap_rr)
        rr_upper_s = inserted_beat_longest_rr_factor * gap_rr
        insert_guard_s = min(extra_rr_factor * gap_rr, max(0.06, 3.0 * local_delay_mad_s))
        selected_inserted_times = []
        selected_expected_times = []
        selected_raw_inserted_times = []
        selected_candidate_indices = []
        beat_decisions = []
        used_candidate_indices = set()
        last_anchor_t = prev_t

        for missing_i in range(n_missing):
            expected_t = last_anchor_t + gap_rr
            if expected_t >= next_t - min_rr_s:
                beat_decisions.append({
                    "missing_i": int(missing_i),
                    "expected_ecg_s": float(expected_t),
                    "accepted": False,
                    "reject_reason": "expected_time_too_close_to_next_ecg",
                })
                break

            candidate_ecg_times_all = pt_peak_times - local_delay_s
            timing_error_all = np.abs(candidate_ecg_times_all - expected_t)
            valid_candidate_mask = (
                (timing_error_all <= pt_search_half_width_s)
                & (pt_peak_prom >= min_rescue_prominence)
                & (candidate_ecg_times_all > last_anchor_t + rr_lower_s)
                & (candidate_ecg_times_all < next_t - min_rr_s)
            )
            if used_candidate_indices:
                used_mask = np.zeros(pt_peak_times.shape, dtype=bool)
                used_mask[list(used_candidate_indices)] = True
                valid_candidate_mask &= ~used_mask

            valid_candidate_indices = np.nonzero(valid_candidate_mask)[0]
            if valid_candidate_indices.size == 0:
                local_peak_mask = np.abs((pt_peak_times - local_delay_s) - expected_t) <= pt_search_half_width_s
                local_prom_ok = bool(np.any(local_peak_mask & (pt_peak_prom >= min_rescue_prominence)))
                local_gap_ok = bool(np.any(
                    local_peak_mask
                    & (pt_peak_prom >= min_rescue_prominence)
                    & (candidate_ecg_times_all > last_anchor_t + rr_lower_s)
                    & (candidate_ecg_times_all < next_t - min_rr_s)
                ))
                if not np.any(local_peak_mask):
                    reject_reason = "no_pt_peak_in_expected_window"
                elif not local_prom_ok:
                    reject_reason = "pt_peak_below_prominence_threshold"
                elif not local_gap_ok:
                    reject_reason = "pt_peak_failed_rr_guard"
                else:
                    reject_reason = "pt_candidate_already_used"
                beat_decisions.append({
                    "missing_i": int(missing_i),
                    "expected_ecg_s": float(expected_t),
                    "accepted": False,
                    "reject_reason": reject_reason,
                })
                break

            prom_score = pt_peak_prom[valid_candidate_indices] / (np.median(pt_peak_prom) + 1e-8)
            score = prom_score - timing_error_all[valid_candidate_indices] / max(pt_search_half_width_s, 1e-8)
            best_idx = int(valid_candidate_indices[int(np.argmax(score))])
            raw_inserted_t = float(candidate_ecg_times_all[best_idx])
            inserted_t = float(np.clip(raw_inserted_t, expected_t - insert_guard_s, expected_t + insert_guard_s))
            rr_from_anchor_s = inserted_t - last_anchor_t
            rr_to_next_s = next_t - inserted_t

            if rr_from_anchor_s < rr_lower_s:
                reject_reason = "inserted_beat_too_close_to_previous_anchor"
                accepted = False
            elif rr_from_anchor_s > rr_upper_s:
                reject_reason = "inserted_beat_too_far_from_previous_anchor"
                accepted = False
            elif rr_to_next_s <= min_rr_s:
                reject_reason = "inserted_beat_too_close_to_next_ecg"
                accepted = False
            else:
                reject_reason = "accepted"
                accepted = True

            beat_decisions.append({
                "missing_i": int(missing_i),
                "expected_ecg_s": float(expected_t),
                "expected_pt_s": float(expected_t + local_delay_s),
                "pt_peak_s": float(pt_peak_times[best_idx]),
                "raw_delay_corrected_ecg_s": float(raw_inserted_t),
                "inserted_ecg_s": float(inserted_t),
                "pt_prominence": float(pt_peak_prom[best_idx]),
                "rr_from_anchor_s": float(rr_from_anchor_s),
                "rr_to_next_s": float(rr_to_next_s),
                "accepted": accepted,
                "reject_reason": reject_reason,
            })

            if not accepted:
                break

            used_candidate_indices.add(best_idx)
            selected_candidate_indices.append(best_idx)
            selected_expected_times.append(float(expected_t))
            selected_raw_inserted_times.append(raw_inserted_t)
            selected_inserted_times.append(inserted_t)
            last_anchor_t = inserted_t
            if next_t - last_anchor_t <= missing_rr_factor * gap_rr:
                break

        selected_inserted_times = np.asarray(selected_inserted_times, dtype=float)
        trial_times = (
            np.r_[prev_t, selected_inserted_times, next_t]
            if selected_inserted_times.size
            else np.asarray([prev_t, next_t])
        )
        trial_rr = np.diff(trial_times)
        accepted = bool(selected_inserted_times.size)
        event = {
            "kind": "missing_window",
            "gap_index": int(gap_i),
            "prev_ecg_s": prev_t,
            "next_ecg_s": next_t,
            "n_missing_expected": int(n_missing),
            "n_missing_raw": int(n_missing_raw),
            "expected_rr_s": float(expected_rr),
            "gap_rr_s": float(gap_rr),
            "local_delay_s": float(local_delay_s),
            "local_delay_mad_s": float(local_delay_mad_s),
            "local_delay_source": local_delay_source,
            "n_selected": int(selected_inserted_times.size),
            "accepted": accepted,
            "reject_reason": (
                "accepted_partial_sequence" if accepted and selected_inserted_times.size < n_missing
                else "accepted" if accepted
                else beat_decisions[-1]["reject_reason"] if beat_decisions
                else "no_repair_attempted"
            ),
            "new_cost_s": float(np.sum(np.abs(trial_rr - gap_rr))),
            "trial_rr_s": [float(t) for t in trial_rr],
            "beat_decisions": beat_decisions,
        }
        if accepted:
            inserted_times.extend([float(t) for t in selected_inserted_times])
            event.update({
                "inserted_ecg_s": [float(t) for t in selected_inserted_times],
                "raw_delay_corrected_ecg_s": [float(t) for t in selected_raw_inserted_times],
                "matched_expected_ecg_s": [float(t) for t in selected_expected_times],
                "pt_peak_s": [float(pt_peak_times[i]) for i in selected_candidate_indices],
                "pt_prominence": [float(pt_peak_prom[i]) for i in selected_candidate_indices],
            })
        events.append(event)

    if inserted_times:
        ecg_times_current = np.sort(np.concatenate([ecg_times_current, np.asarray(inserted_times)]))

    rr_after_insert = np.diff(ecg_times_current)
    local_rr_after_insert = np.zeros_like(rr_after_insert)
    for rr_i in range(rr_after_insert.size):
        lo = max(0, rr_i - half_window)
        hi = min(rr_after_insert.size, rr_i + half_window + 1)
        local_rr_after_insert[rr_i] = np.median(rr_after_insert[lo:hi])

    short_after_insert = np.nonzero(
        (rr_after_insert < extra_rr_factor * local_rr_after_insert)
        & ((local_rr_after_insert - rr_after_insert) > extra_abs_floor_s)
    )[0]
    removed_times = []
    for gap_i in short_after_insert:
        if gap_i <= 0 or gap_i + 2 >= ecg_times_current.size:
            continue
        best_remove_idx = None
        best_cost_gain = 0.0
        full_local = ecg_times_current[gap_i - 1:gap_i + 3]
        keep_cost = abs(np.diff(full_local)[0] - local_rr_after_insert[gap_i]) + abs(
            np.diff(full_local)[1] - local_rr_after_insert[gap_i]
        )
        for remove_idx in (gap_i, gap_i + 1):
            trial = np.delete(ecg_times_current[gap_i - 1:gap_i + 3], remove_idx - (gap_i - 1))
            trial_rr = np.diff(trial)
            trial_cost = np.sum(np.abs(trial_rr - local_rr_after_insert[gap_i]))
            cost_gain = keep_cost - trial_cost
            ecg_t = ecg_times_current[remove_idx]
            expected_pt_t = ecg_t + ecg_pt_delay_s
            pt_support = np.any(
                (pt_peak_times >= expected_pt_t - pt_search_half_width_s)
                & (pt_peak_times <= expected_pt_t + pt_search_half_width_s)
                & (pt_peak_prom >= min_rescue_prominence)
            )
            if (not pt_support) and cost_gain > best_cost_gain:
                best_cost_gain = float(cost_gain)
                best_remove_idx = int(remove_idx)
        event = {
            "kind": "extra_trigger_window",
            "gap_index": int(gap_i),
            "accepted": best_remove_idx is not None,
            "short_gap_s": float(rr_after_insert[gap_i]),
        }
        if best_remove_idx is not None:
            remove_t = float(ecg_times_current[best_remove_idx])
            event["removed_ecg_s"] = remove_t
            event["cost_gain_s"] = best_cost_gain
            removed_times.append(remove_t)
        events.append(event)

    if removed_times:
        remove_mask = np.ones(ecg_times_current.shape, dtype=bool)
        remove_tol_s = 0.5 * min(dt_ecg, dt_pt)
        for remove_t in removed_times:
            remove_mask &= np.abs(ecg_times_current - remove_t) > remove_tol_s
        ecg_times_current = ecg_times_current[remove_mask]

    repaired_triggers = np.zeros(time_ecg.shape, dtype=np.uint32)
    for trig_t in ecg_times_current:
        if trig_t < time_ecg[0] or trig_t > time_ecg[-1]:
            continue
        repaired_triggers[int(np.argmin(np.abs(time_ecg - trig_t)))] = 1

    rr_final = np.diff(ecg_times_current)
    hr_before_bpm = 60.0 / rr_initial if rr_initial.size else np.asarray([])
    hr_after_bpm = 60.0 / rr_final if rr_final.size else np.asarray([])
    total_expected_beats = ecg_times_initial.size + sum(
        int(event["n_missing_expected"]) for event in events if event["kind"] == "missing_window"
    )
    stats = {
        "n_ecg_beats": int(ecg_times_initial.size),
        "n_total_expected_beats": int(total_expected_beats),
        "n_repaired_beats": len(inserted_times),
        "n_final_beats": int(ecg_times_current.size),
        "n_removed_ecg_beats": len(removed_times),
        "n_pt_peaks": int(pt_peak_times.size),
        "n_ecg_pt_matches": int(raw_delays.size),
        "n_ecg_pt_delay_inliers": int(np.sum(delay_keep)),
        "ecg_pt_delay_s": float(ecg_pt_delay_s),
        "ecg_pt_delay_mad_s": float(ecg_pt_delay_mad_s),
        "pt_search_half_width_s": float(pt_search_half_width_s),
        "min_rescue_prominence": float(min_rescue_prominence),
        "n_long_gaps": int(long_gap_indices.size),
        "n_short_gaps_initial": int(short_gap_indices.size),
        "n_short_gaps_after_insert": int(short_after_insert.size),
        "rr_std_before_s": float(np.std(rr_initial)) if rr_initial.size else np.nan,
        "rr_std_after_s": float(np.std(rr_final)) if rr_final.size else np.nan,
        "hr_mean_before_bpm": float(np.mean(hr_before_bpm)) if hr_before_bpm.size else np.nan,
        "hr_median_before_bpm": float(np.median(hr_before_bpm)) if hr_before_bpm.size else np.nan,
        "hr_mean_after_bpm": float(np.mean(hr_after_bpm)) if hr_after_bpm.size else np.nan,
        "hr_median_after_bpm": float(np.median(hr_after_bpm)) if hr_after_bpm.size else np.nan,
        "initial_trigger_times_s": [float(t) for t in ecg_times_initial],
        "final_trigger_times_s": [float(t) for t in ecg_times_current],
        "inserted_trigger_times_s": [float(t) for t in inserted_times],
        "removed_trigger_times_s": [float(t) for t in removed_times],
        "events": events,
    }

    return repaired_triggers, stats
