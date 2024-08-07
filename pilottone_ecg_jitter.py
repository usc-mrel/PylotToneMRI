import ismrmrd
import rtoml
import os
import fnmatch
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
sys.path.append('./src')
from pilottone.pt import beat_rejection, interval_peak_matching

if __name__ == '__main__':


    # Read config
    with open('config.toml', 'r') as cf:
        cfg = rtoml.load(cf)

    DATA_ROOT = cfg['DATA_ROOT']
    DATA_DIR = cfg['data_folder']
    raw_file = cfg['raw_file']
    gpu_device = cfg['gpu_num']

    data_dir_path = os.path.join(DATA_ROOT, DATA_DIR, 'raw/h5')
    if raw_file.isnumeric():
        raw_file_ = fnmatch.filter(os.listdir(data_dir_path), f'meas_MID*{raw_file}*.h5')[0]
        ismrmrd_data_fullpath = os.path.join(data_dir_path, raw_file_)
    elif raw_file.startswith('meas_MID'):
        raw_file_ = raw_file
        ismrmrd_data_fullpath = os.path.join(data_dir_path, raw_file)
    else:
        print('Could not find the file. Exiting...')
        exit(-1)

    # Read the data in
    print(f'Reading {ismrmrd_data_fullpath}...')
    with ismrmrd.Dataset(ismrmrd_data_fullpath) as dset:
        n_wf = dset.number_of_waveforms()
        print(f'There are {n_wf} waveforms in the dataset. Reading...')

        wf_list = []
        for ii in range(n_wf):
            wf_list.append(dset.read_waveform(ii))
        
        print('Waveforms read.')
        hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
        # Read the first and acquisitions for their timestamps
        n_acq = dset.number_of_acquisitions()
        acq_s = dset.read_acquisition(0)
        acq_e = dset.read_acquisition(n_acq-1)

    ## Process ECG waveform
    ecg_waveform = []
    ecg_trigs = []
    wf_init_timestamp = 0
    for wf in wf_list:
        if wf.getHead().waveform_id == 0:
            ecg_waveform.append(wf.data[0,:])
            ecg_trigs.append(wf.data[4,:])
            if wf_init_timestamp == 0:
                wf_init_timestamp = wf.time_stamp
                ecg_sampling_time = wf_list[0].getHead().sample_time_us*1e-6 # [us] -> [s]

    ecg_waveform = (np.asarray(np.concatenate(ecg_waveform, axis=0), dtype=float)-2048)
    ecg_waveform = ecg_waveform/np.percentile(ecg_waveform, 99.9)
    ecg_trigs = (np.concatenate(ecg_trigs, axis=0)/2**14).astype(int)
    time_ecg = np.arange(ecg_waveform.shape[0])*ecg_sampling_time - (acq_s.acquisition_time_stamp - wf_init_timestamp)*1e-3

    ## Read PT
    resp_waveform = []
    pt_card_triggers = []

    for wf in wf_list:
        if wf.getHead().waveform_id == 1025:
            resp_waveform = wf.data[0,:]
            pt_cardiac = ((wf.data[1,:].astype(float) - 2**31)/2**31)
            pt_cardiac_trigs = np.round(((wf.data[2,:] - 2**31)/2**31)).astype(int)
            pt_cardiac_derivative = ((wf.data[3,:].astype(float) - 2**31)/2**31)
            pt_derivative_trigs = np.round((wf.data[4,:] - 2**31)/2**31).astype(int)

            pt_sampling_time = wf.getHead().sample_time_us*1e-6
            break


    t_acq_start = acq_s.acquisition_time_stamp*2.5e-3 # [2.5ms] -> [s]
    t_acq_end = acq_e.acquisition_time_stamp*2.5e-3
    time_acq = np.linspace(t_acq_start, t_acq_end, n_acq) # Interpolate for TR, as TR will not be a multiple of time resolution.
    time_pt = time_acq - t_acq_start
    samp_time_pt = time_acq[1] - time_acq[0]

    plt.figure()
    plt.plot(time_ecg, ecg_waveform)
    plt.plot(time_ecg[ecg_trigs==1], ecg_waveform[ecg_trigs==1], '*')
    plt.plot(time_pt, pt_cardiac_trigs, 'x', label='PT Triggers')

    skip_time = 0.6 # [s]

    # ECG Triggers
    # ecg_peak_locs,_ = find_peaks(ecg_waveform[time_ecg > skip_time], prominence=0.7)
    ecg_peak_locs = np.nonzero(ecg_trigs[time_ecg > skip_time])[0]
    ecg_peak_locs += np.sum(time_ecg <= skip_time)

    # PT Triggers
    dt_pt = (time_pt[1] - time_pt[0])
    Dmin = int(np.ceil(0.65/(dt_pt))) # Min distance between two peaks, should not be less than 0.6 secs (100 bpm max assumed)
    # pt_cardiac_peak_locs,_ = find_peaks(pt_cardiac[time_pt > skip_time], prominence=0.4, distance=Dmin)
    pt_cardiac_peak_locs = np.nonzero(pt_cardiac_trigs[time_pt > skip_time])[0]
    pt_cardiac_peak_locs += np.sum(time_pt <= skip_time)

    # PT Derivative Triggers
    # pt_cardiac_derivative_peak_locs,_ = find_peaks(pt_cardiac_derivative[time_pt > skip_time], prominence=0.6, distance=Dmin)
    pt_cardiac_derivative_peak_locs = np.nonzero(pt_derivative_trigs[time_pt > skip_time])[0]
    pt_cardiac_derivative_peak_locs += np.sum(time_pt <= skip_time)

    # "Arryhtmia detection" by heart rate variation
    hr_accept_list = beat_rejection(pt_cardiac_peak_locs*dt_pt, "post")
    hr_accept_list_derivative = beat_rejection(pt_cardiac_derivative_peak_locs*dt_pt, "pre")
    # TODO: Is pre post even correct? Why does it change? Need to investigate.
    print(f'Rejection ratio for pt peaks is {100*(len(hr_accept_list) - np.sum(hr_accept_list))/len(hr_accept_list):.2f} percent.\n')
    print(f'Rejection ratio for derivative pt peaks is {100*(len(hr_accept_list_derivative) - np.sum(hr_accept_list_derivative))/len(hr_accept_list_derivative):.2f} percent.\n')


    # peak_diff, pt_peaks_selected = prepeak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    # derivative_peak_diff, pt_derivative_peaks_selected = prepeak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)

    peak_diff, miss_pks, extra_pks = interval_peak_matching(time_pt, pt_cardiac_peak_locs, time_ecg, ecg_peak_locs)
    pt_peaks_selected = pt_cardiac_peak_locs

    derivative_peak_diff, derivative_miss_pks, derivative_extra_pks = interval_peak_matching(time_pt, pt_cardiac_derivative_peak_locs, time_ecg, ecg_peak_locs)
    pt_derivative_peaks_selected = pt_cardiac_derivative_peak_locs

    # Create trigger waveforms from peak locations.
    pt_cardiac_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_derivative_trigs = np.zeros((n_acq,), dtype=np.uint32)
    pt_cardiac_trigs[pt_peaks_selected] = 1
    pt_derivative_trigs[pt_derivative_peaks_selected] = 1

    # Print some useful info

    print(f'Peak difference {np.mean(peak_diff*1e3):.1f} \u00B1 {np.std(peak_diff*1e3):.1f}')
    print(f'Derivative peak difference {np.mean(derivative_peak_diff*1e3):.1f} \u00B1 {np.std(derivative_peak_diff*1e3):.1f}')

    print(f'Number of ECG triggers: {ecg_peak_locs.shape[0]}.')
    print(f'Number of PT triggers: {pt_cardiac_peak_locs.shape[0]}.')
    print(f'Number of missed PT triggers: {miss_pks.shape[0]}.')
    print(f'Number of extraneous PT triggers: {extra_pks.shape[0]}.')
    print(f'Number of derivative PT triggers: {pt_cardiac_derivative_peak_locs.shape[0]}.')
    print(f'Number of missed derivative PT triggers: {derivative_miss_pks.shape[0]}.')
    print(f'Number of extraneous derivative PT triggers: {derivative_extra_pks.shape[0]}.')


    # Plots
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