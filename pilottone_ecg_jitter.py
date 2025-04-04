import datetime
import os
import rtoml
import numpy as np

from pilottone import beat_rejection, interval_peak_matching, get_volt_from_protoname
import mrdhelper
from scipy.signal import find_peaks
from ui.selectionui import get_multiple_filepaths

def pt_ecg_jitter(time_pt, pt_cardiac, pt_cardiac_derivative, time_ecg, ecg_waveform, pt_cardiac_trigs=None, pt_derivative_trigs=None, ecg_trigs=None, skip_time=0.6, max_hr=120, show_outputs=True): 
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

def print_pt_confusion_table(pt_stats, raw_files, title='Pilot Tone Confusion Table'):
    print('| {:^70} |'.format(title))
    print('| {:^10} | {:>15} | {:>15} | {:>15} |'.format('PT Voltage', 'Number of ECG triggers', 'False Negative', 'False Positive'))
    for raw_file in raw_files:
        row = [pt_stats[raw_file][0], np.sum(pt_stats[raw_file][-1]), len(pt_stats[raw_file][2]), len(pt_stats[raw_file][3])]
        print('| {:^10} | {:>20} | {:>15} | {:>15} |'.format(*row))

if __name__ == '__main__':

    import matplotlib
    matplotlib.use('QtAgg')
    import matplotlib.pyplot as plt
    # Read config
    with open('config.toml', 'r') as cf:
        cfg = rtoml.load(cf)

    filepaths = get_multiple_filepaths(dir=cfg['DATA_ROOT'])
    dataset_dir = os.path.join('/', *(os.path.dirname(filepaths[0]).split('/')[:-2]))
    DATA_ROOT = os.path.dirname(dataset_dir)
    DATA_DIR = os.path.basename(dataset_dir)
    raw_files = [os.path.basename(filepath) for filepath in filepaths]

    pt_stats = {}
    dpt_stats = {}

    for raw_file in raw_files:
        ismrmrd_data_fullpath, _ = mrdhelper.siemens_mrd_finder(DATA_ROOT, DATA_DIR, raw_file)
        ptvolt = get_volt_from_protoname(ismrmrd_data_fullpath.split('/')[-1])
        # Read the data in
        wf_list, _ = mrdhelper.read_waveforms(ismrmrd_data_fullpath)

        ecg_, pt_ = mrdhelper.waveforms_asarray(wf_list)
        ecg_waveform = ecg_['ecg_waveform']
        ecg_waveform /= np.percentile(ecg_waveform, 99.9)
        ecg_trigs = ecg_['ecg_trigs']
        time_ecg = ecg_['time_ecg']

        pt_cardiac_trigs = pt_['pt_cardiac_trigs']
        pt_derivative_trigs = pt_['pt_derivative_trigs']
        pt_cardiac = pt_['pt_cardiac']
        pt_cardiac_derivative = pt_['pt_cardiac_derivative']
        pt_cardiac[:20] = pt_cardiac[20]
        pt_cardiac[-20:] = pt_cardiac[-20]
        pt_cardiac -= np.percentile(pt_cardiac, 10)
        pt_cardiac /= np.percentile(pt_cardiac, 95)
        pt_cardiac_derivative[:20] = pt_cardiac_derivative[20]
        pt_cardiac_derivative[-20:] = pt_cardiac_derivative[-20]
        pt_cardiac_derivative -= np.percentile(pt_cardiac_derivative, 10)
        pt_cardiac_derivative /= np.percentile(pt_cardiac_derivative, 98)

        time_pt = pt_['time_pt']

        # Shift the time axes for our mortal brains
        t_ref = min(time_ecg[0], time_pt[0])
        time_ecg -= t_ref
        time_pt -= t_ref

        # plt.figure()
        # plt.plot(time_ecg, ecg_waveform)
        # plt.plot(time_ecg[ecg_trigs==1], ecg_waveform[ecg_trigs==1], '*')
        # plt.plot(time_pt, pt_cardiac)
        # plt.plot(time_pt[pt_cardiac_trigs==1], pt_cardiac[pt_cardiac_trigs==1], 'x', label='PT Triggers')
        # plt.show()
        skip_time = 1.5
        print(f'ECG trigs: {np.sum(ecg_trigs[time_ecg > skip_time])}')

        pt_stats_ = calculate_jitter(time_pt, pt_cardiac, time_ecg, ecg_waveform, ecg_trigs=ecg_trigs, 
                                    #  pt_cardiac_trigs=pt_cardiac_trigs, 
                                     skip_time=skip_time, peak_prominence=0.4, max_hr=160)
        dpt_stats_ = calculate_jitter(time_pt, pt_cardiac_derivative, time_ecg, ecg_waveform, ecg_trigs=ecg_trigs, 
                                    #   pt_cardiac_trigs=pt_derivative_trigs, 
                                      skip_time=skip_time, peak_prominence=0.5, max_hr=160)

        pt_stats[raw_file] = (ptvolt,*pt_stats_, ecg_trigs[time_ecg > skip_time])
        dpt_stats[raw_file] = (ptvolt,*dpt_stats_, ecg_trigs[time_ecg > skip_time])

    jitter = np.array([np.std(pt_stats[raw_file][1]) for raw_file in raw_files])
    mean_delay = np.array([np.mean(pt_stats[raw_file][1]) for raw_file in raw_files])

    dpt_jitter = np.array([np.std(dpt_stats[raw_file][1]) for raw_file in raw_files])
    dpt_mean_delay = np.array([np.mean(dpt_stats[raw_file][1]) for raw_file in raw_files])
    ptvolts = np.array([dpt_stats[raw_file][0] for raw_file in raw_files])
    pt_sampling_time_ms = pt_['pt_sampling_time']*1e3

    # Print table of the false negatives and false positives
    print_pt_confusion_table(pt_stats, raw_files)
    print_pt_confusion_table(dpt_stats, raw_files, title='Derivative Pilot Tone Confusion Table')

    # Plot the jitter and mean delays
    f, axs = plt.subplots(2,2)
    axs[0,0].plot(ptvolts, jitter*1e3, 'o', label='Jitter')
    axs[0,0].set_xlim([0, 1.6])
    axs[0,0].set_xlabel('Pilot Tone Voltage [V]')
    axs[0,0].set_ylabel('Jitter [ms]')
    axs[0,0].set_title('Jitter vs PT Voltage')
    axs[0,0].set_ylim([0, 25])
    # axs[0,0].axhline(y=pt_sampling_time_ms*2, xmin=0, xmax=1, color='red', linestyle='--', label='2x PT Sampling Period', alpha=0.7)

    axs[1,0].plot(ptvolts, mean_delay*1e3, 'o')
    axs[1,0].set_xlabel('Pilot Tone Voltage [V]')
    axs[1,0].set_ylabel('Mean Delay [ms]')
    axs[1,0].set_title('Mean Delay vs PT Voltage')

    axs[0,1].plot(ptvolts, dpt_jitter*1e3, 'o')
    axs[0,1].set_xlabel('Pilot Tone Voltage [V]')
    axs[0,1].set_ylabel('Jitter [ms]')
    axs[0,1].set_title('Jitter vs Inverse Derivative PT Voltage')
    axs[0,1].set_ylim([0, 25])

    # axs[0,1].axhline(y=pt_sampling_time_ms*2, xmin=0, xmax=1, color='red', linestyle='--', label='2x PT Sampling Period', alpha=0.7)

    axs[1,1].plot(ptvolts, dpt_mean_delay*1e3, 'o')
    axs[1,1].set_xlabel('Pilot Tone Voltage [V]')
    axs[1,1].set_ylabel('Mean Delay [ms]')
    axs[1,1].set_title('Mean Delay vs Inverse Derivative PT Voltage')

    plt.show()

    np.savez_compressed(os.path.join('output_recons', DATA_DIR, f'jitter_{datetime.datetime.now()}.npz'), 
                        ptvolts=ptvolts, pk_diffs=pt_stats, derivative_pk_diffs=dpt_stats, pt_sampling_time=pt_['pt_sampling_time'])