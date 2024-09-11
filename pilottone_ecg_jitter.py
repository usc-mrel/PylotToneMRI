import datetime
import os
import rtoml
import numpy as np

from pilottone import beat_rejection, interval_peak_matching, get_volt_from_protoname
import mrdhelper
from scipy.signal import find_peaks

def pt_ecg_jitter(time_pt, pt_cardiac, pt_cardiac_derivative, time_ecg, ecg_waveform, pt_cardiac_trigs=None, pt_derivative_trigs=None, ecg_trigs=None, skip_time=0.6, show_outputs=True): 
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
        Dmin = int(np.ceil(0.65/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
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

    if show_outputs:
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
        plt.figure()
        plt.plot(time_ecg, ecg_waveform)
        plt.plot(time_ecg[ecg_trigs==1], ecg_waveform[ecg_trigs==1], '*')
        plt.plot(time_pt, pt_cardiac_trigs, 'x', label='PT Triggers')

        f, axs = plt.subplots(2,2, sharex='col')
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

        plt.show()
    
    return peak_diff, derivative_peak_diff

if __name__ == '__main__':

    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    # Read config
    with open('config.toml', 'r') as cf:
        cfg = rtoml.load(cf)

    DATA_ROOT = cfg['DATA_ROOT']
    DATA_DIR = cfg['data_folder']
    # raw_file = cfg['raw_file']
    raw_files = ['282', '283', '284', '285', '286', '287', '289', '290', '291', '292']
    pk_diffs = {}
    derivative_pk_diffs = {}
    for raw_file in raw_files:
        ismrmrd_data_fullpath, _ = mrdhelper.siemens_mrd_finder(DATA_ROOT, DATA_DIR, raw_file)
        ptvolt = get_volt_from_protoname(ismrmrd_data_fullpath.split('/')[-1])
        # Read the data in
        wf_list, _ = mrdhelper.read_waveforms(ismrmrd_data_fullpath)

        ecg_, pt_ = mrdhelper.waveforms_asarray(wf_list)
        ecg_waveform = ecg_['ecg_waveform']
        ecg_trigs = ecg_['ecg_trigs']
        time_ecg = ecg_['time_ecg']

        pt_cardiac_trigs = pt_['pt_cardiac_trigs']
        pt_derivative_trigs = pt_['pt_derivative_trigs']
        pt_cardiac = pt_['pt_cardiac']
        pt_cardiac_derivative = pt_['pt_cardiac_derivative']
        time_pt = pt_['time_pt']

        # Shift the time axes for our mortal brains
        t_ref = min(time_ecg[0], time_pt[0])
        time_ecg -= t_ref
        time_pt -= t_ref

        pk_diffs[raw_file], derivative_pk_diffs[raw_file] = pt_ecg_jitter(time_pt, pt_cardiac, pt_cardiac_derivative, time_ecg, ecg_waveform, pt_cardiac_trigs, pt_derivative_trigs, ecg_trigs, show_outputs=False)

        pk_diffs[raw_file] = (ptvolt, pk_diffs[raw_file])
        derivative_pk_diffs[raw_file] = (ptvolt, derivative_pk_diffs[raw_file])

    jitter = np.array([np.std(pk_diffs[raw_file][1]) for raw_file in raw_files])
    mean_delay = np.array([np.mean(pk_diffs[raw_file][1]) for raw_file in raw_files])
    ptvolts = np.array([pk_diffs[raw_file][0] for raw_file in raw_files])

    plt.figure()
    plt.plot(ptvolts, jitter*1e3, 'o')
    plt.xlabel('Pilot Tone Voltage [V]')
    plt.ylabel('Jitter [ms]')
    plt.title('Jitter vs PT Voltage')

    plt.figure()
    plt.plot(ptvolts, mean_delay*1e3, 'o')
    plt.xlabel('Pilot Tone Voltage [V]')
    plt.ylabel('Mean Delay [ms]')
    plt.title('Mean Delay vs PT Voltage')
    plt.show()

    np.savez_compressed(os.path.join('output_recons', DATA_DIR, f'jitter_{datetime.datetime.now()}.npz'), 
                        ptvolts=ptvolts, pk_diffs=pk_diffs, derivative_pk_diffs=derivative_pk_diffs)