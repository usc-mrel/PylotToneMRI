# %%
import argparse
import ctypes
import os
from pathlib import Path
from typing import Union

import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import rtoml
from scipy.io import loadmat
from scipy.signal import savgol_filter

import pylottone as pt
import pylottone.mrdhelper as mrdhelper
from pylottone.constants import PILOTTONE_WAVEFORM_ID
from pylottone.editer import autopick_sensing_coils
from pylottone.reconstruction.coils import (
    apply_prewhitening,
    calculate_prewhitening,
)
from pylottone.selectionui import get_multiple_filepaths
from pylottone.signal import angle_dependant_filtering
from pylottone.trajectory import remove_readout_os
from pylottone.triggering import extract_triggers, pt_ecg_jitter

# %%
# Read the data in

def main(ismrmrd_data_fullpath, cfg) -> Union[str, None]:
    f_pt = cfg['pilottone']['pt_freq']
    remove_os = cfg['saving']['remove_os']

    data_dir = os.path.join('/', *(os.path.dirname(ismrmrd_data_fullpath).split('/')[:-2]))
    print(f"Data dir: {data_dir}")

    raw_file = ismrmrd_data_fullpath.split('/')[-1]
    ismrmrd_data_fullpath, ismrmrd_noise_fullpath = mrdhelper.siemens_mrd_finder(data_dir, '', raw_file)

    acq_list, wf_list, hdr = mrdhelper.read_mrd(ismrmrd_data_fullpath)
    n_acq = len(acq_list)

    # get the k-space trajectory based on the metadata hash.
    traj_name = hdr.userParameters.userParameterString[1].value

    # load the .mat file containing the trajectory
    traj = loadmat(os.path.join(data_dir, traj_name), squeeze_me=True)

    n_unique_angles = int(traj['param']['repetitions'])
    pre_discard = int(traj['param']['pre_discard'])

    kx = traj['kx'][:,:]
    ky = traj['ky'][:,:]
    kx = np.vstack((np.zeros((pre_discard, n_unique_angles)), traj['kx'][:,:]))
    ky = np.vstack((np.zeros((pre_discard, n_unique_angles)), traj['ky'][:,:]))
    dt = float(traj['param']['dt'])

    data = np.array([arm.data[:,:] for arm in acq_list]).transpose((2, 0, 1))

    # %%
    n_channels = data.shape[2]
    sensing_coils = np.array(cfg['pilottone']['sensing_coils'], dtype=int)
    mri_coils = np.arange(n_channels)
    mri_coils = mri_coils[~np.isin(mri_coils, sensing_coils)]

    coil_name = []

    for clbl in hdr.acquisitionSystemInformation.coilLabel:
        coil_name.append(clbl.coilName)

    coil_name = np.asarray(coil_name)
    print(f'Coil names: {coil_name}')
    print(f"Coils to be used as sniffers: {coil_name[sensing_coils]}")

    f0 = hdr.experimentalConditions.H1resonanceFrequency_Hz
    df = 1/(dt*data.shape[0])

    t_acq_start = acq_list[0].acquisition_time_stamp*2.5e-3 # [2.5ms] -> [s]
    t_acq_end = acq_list[-1].acquisition_time_stamp*2.5e-3
    time_acq = np.linspace(t_acq_start, t_acq_end, n_acq) # Interpolate for TR, as TR will not be a multiple of time resolution.
    time_pt = time_acq - t_acq_start
    samp_time_pt = time_acq[1] - time_acq[0]

    ksp_measured = data[:,:,mri_coils]
    # ksp_sniffer  = data[:,:,sensing_coils]

    ## Process ECG waveform
    ecg, _ = mrdhelper.waveforms_asarray(wf_list)
    if ecg is not None:
        ecg_waveform = ecg['ecg_waveform']
        ecg_waveform = pt.check_waveform_polarity(ecg_waveform, 0.5)*ecg_waveform
        time_ecg = ecg['time_ecg'] - acq_list[0].acquisition_time_stamp*2.5e-3
        ecg_trigs = ecg['ecg_trigs']
    else:
        print('No ECG waveform found, skipping the validation part.')
    # plt.figure()
    # plt.plot(time_ecg, ecg_waveform)
    # plt.plot(time_ecg[ecg_trigs==1], ecg_waveform[ecg_trigs==1], '*')


    # %% [markdown]
    # ## PT correction

    # %%

    f_diff = f0 - f_pt

    pt_raw, ksp_ptsubbed = pt.extract_raw_pt(ksp_measured, kx, ky, n_unique_angles, acq_list[0], f_diff, df, dt, method='wPCA', freq_correction=True, return_complex=True)

    pt_sig = np.abs(pt_raw)
    pt_sig = np.squeeze(pt_sig - np.mean(pt_sig, axis=1, keepdims=True))
    pt_sig = angle_dependant_filtering(pt_sig, n_unique_angles)

    # %% [markdown]
    # ## QA and ECG PT Jitter

    # %%

    f_samp = 1/samp_time_pt # [Hz]
    print(f"Using {cfg['pilottone']['cardiac']['initial_channel']} as the initial cardiac coil.")
    pt_extract_params = {'golay_filter_len': cfg['pilottone']['golay_filter_len'],
                        'respiratory': {
                                'freq_start': cfg['pilottone']['respiratory']['freq_start'],
                                'freq_stop': cfg['pilottone']['respiratory']['freq_stop'],
                                'corr_threshold': cfg['pilottone']['respiratory']['corr_threshold'],
                                'corr_init_ch': cfg['pilottone']['respiratory']['initial_channel'],
                                'separation_method': cfg['pilottone']['respiratory']['separation_method'], # 'sobi', 'pca'
                        },
                        'cardiac': {
                                    'freq_start': cfg['pilottone']['cardiac']['freq_start'],
                                    'freq_stop': cfg['pilottone']['cardiac']['freq_stop'],
                                    'corr_threshold': cfg['pilottone']['cardiac']['corr_threshold'],
                                    'corr_init_ch': np.nonzero(coil_name == cfg['pilottone']['cardiac']['initial_channel'])[0][0],                           
                                    'separation_method': cfg['pilottone']['cardiac']['separation_method'], # 'sobi', 'pca'
                                    'num_lags': 375, # SOBI number of lags
                        },
                        'debug': {
                            'selected_coils': cfg['pilottone']['debug']['selected_coils'],
                            'coil_legend': coil_name[mri_coils],
                            'show_plots': cfg['pilottone']['debug']['show_plots'],
                            'no_normalize': cfg['pilottone']['debug']['no_normalize'],
                        }
                    }

    sg_filter_len = 81

    pt_respiratory, pt_cardiac = pt.extract_pilottone_navs(pt_sig, f_samp, pt_extract_params)
    
    # %% Save the waveforms separately if requested

    if cfg['saving']['save_pt_separate']:
        print('Saving waveforms separately...')
        np.savez(os.path.join(data_dir, f"{ismrmrd_data_fullpath.split('/')[-1][:-3]}_ptwaveforms.npz"), 
                 pt_raw=pt_raw,
                 pt_respiratory=pt_respiratory, 
                 pt_cardiac=pt_cardiac,
                 time_pt=time_pt)

    pt_cardiac = cfg['pilottone']['cardiac']['sign']*pt_cardiac
    pt_cardiac[:20] = pt_cardiac[20]
    pt_cardiac[-20:] = pt_cardiac[-20]
    pt_cardiac -= np.percentile(pt_cardiac, 10)
    pt_cardiac /= np.percentile(pt_cardiac, 95)

    pt_cardiac_filtered = savgol_filter(pt_cardiac, sg_filter_len, 3, axis=0)
    pt_cardiac_derivative = np.hstack((0, np.diff(pt_cardiac_filtered)/(time_pt[1] - time_pt[0])))
    pt_cardiac_derivative[:20] = pt_cardiac_derivative[20]
    pt_cardiac_derivative[-20:] = pt_cardiac_derivative[-20]
    pt_cardiac_derivative -= np.percentile(pt_cardiac_derivative, 10)
    pt_cardiac_derivative /= np.percentile(pt_cardiac_derivative, 98)

    pt_cardiac_trigs = extract_triggers(time_pt, pt_cardiac, skip_time=1, prominence=0.4, max_hr=160)
    pt_derivative_trigs = extract_triggers(time_pt, pt_cardiac_derivative, skip_time=1, prominence=0.5, max_hr=160)

    if ecg is not None:
        _,_ = pt_ecg_jitter(time_pt, pt_cardiac, pt_cardiac_derivative,
                            time_ecg, ecg_waveform, 
                            pt_cardiac_trigs=pt_cardiac_trigs, pt_derivative_trigs=pt_derivative_trigs, ecg_trigs=ecg_trigs, 
                            skip_time=1, show_outputs=cfg['pilottone']['show_outputs'])
    elif ecg is None and cfg['pilottone']['show_outputs']:
        fig, axs = plt.subplots(2,1, figsize=(10, 6), sharex=True)
        axs[0].plot(time_pt, pt_cardiac, label='Cardiac')
        axs[0].plot(time_pt, pt_cardiac_derivative, label='Cardiac Derivative')
        axs[0].plot(time_pt[pt_cardiac_trigs==1], pt_cardiac[pt_cardiac_trigs==1], '*', label='Cardiac Triggers')
        axs[0].plot(time_pt[pt_derivative_trigs==1], pt_cardiac_derivative[pt_derivative_trigs==1], '*', label='Cardiac Derivative Triggers')
        axs[0].set_title('Cardiac Triggers')
        axs[0].set_xlabel('Time [s]')
        axs[0].legend()
        axs[1].plot(time_pt, pt_respiratory, label='Respiratory')
        axs[1].set_title('Respiratory')
        axs[1].set_xlabel('Time [s]')
        axs[1].legend()
        plt.show()


    # %% [markdown]
    # ## Save the waveforms into the original data
    if cfg['saving']['save_pt_waveforms']:
        # Concat, and normalize pt waveforms.
        dt_pt = (time_pt[1] - time_pt[0])
        pt_wf_data = np.vstack((pt_respiratory, pt_cardiac, pt_cardiac_trigs, pt_cardiac_derivative, pt_derivative_trigs))
        pt_wf_data = ((pt_wf_data/np.max(np.abs(pt_wf_data), axis=1, keepdims=True)*(2**31-1)) + 2**31).astype(np.uint32)

        pt_wf = ismrmrd.waveform.Waveform.from_array(pt_wf_data)
        pt_wf._head.sample_time_us = ctypes.c_float(dt_pt*1e6)
        pt_wf._head.waveform_id = ctypes.c_uint16(PILOTTONE_WAVEFORM_ID)
        pt_wf._head.time_stamp = acq_list[0].acquisition_time_stamp

        with ismrmrd.Dataset(ismrmrd_data_fullpath) as dset:
            dset.append_waveform(pt_wf)

        print('Done writing the waveform.')


    # %% [markdown]
    # # Save the PT subtracted k-space

    if cfg['saving']['save_model_subtracted']:

        if cfg['pilottone']['prewhiten']:

            # Read the noise data in
            print(f'Reading {ismrmrd_noise_fullpath}...')
            with ismrmrd.Dataset(ismrmrd_noise_fullpath) as dset_noise:
                n_cal_acq = dset_noise.number_of_acquisitions()
                print(f'There are {n_cal_acq} acquisitions in the file. Reading...')

                cal_list = []
                for ii in range(n_cal_acq):
                    cal_list.append(dset_noise.read_acquisition(ii))

            noise_list = []

            for cal_ in cal_list:
                if cal_.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noise_list.append(cal_.data)

            noise = np.transpose(np.asarray(noise_list), (1,0,2)).reshape((noise_list[0].shape[0], -1))[mri_coils,:]
            print('Prewhitening the raw data...')
            dmtx = calculate_prewhitening(noise)

            ksp_ptsubbed = apply_prewhitening(np.transpose(ksp_ptsubbed, (2,0,1)), dmtx).transpose((1,2,0))

        if cfg['pilottone']['discard_badcoils']:
            mri_coils2, _ = autopick_sensing_coils(ksp_measured, f_emi=f_diff, bw_emi=100e3, bw_sig=200e3, f_samp=1/dt, n_sensing=5)
            ksp_ptsubbed = ksp_ptsubbed[:,:,mri_coils2]
            mri_coils = mri_coils[mri_coils2]
            
        if remove_os:
            ksp_ptsubbed = remove_readout_os(ksp_ptsubbed)

        output_dir_fullpath = os.path.join(data_dir, 'raw', 'h5_proc')
        output_data_fullpath = os.path.join(output_dir_fullpath, f"{ismrmrd_data_fullpath.split('/')[-1][:-3]}_mdlsub.h5")
        print('Saving to ' + output_data_fullpath)

        Path.mkdir(Path(output_dir_fullpath), exist_ok=True)
        user_params = {'processing': 'ModelSubtraction'}
        mrdhelper.save_processed_raw_data(output_data_fullpath, hdr, acq_list, wf_list, 
                                             ksp_ptsubbed, mri_coils, pt_wf if cfg['saving']['save_pt_waveforms'] else None, user_params)

        return output_data_fullpath

if __name__ == '__main__':
    # Check if filepaths are provided as arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filepaths', nargs='+', help='List of filepaths to process.')
    argparser.add_argument('-c', '--config', nargs='?', default='config.toml', help='Config file to be used during processing.')

    args = argparser.parse_args()

    with open(args.config, 'r') as cf:
        cfg = rtoml.load(cf)

    if args.filepaths:
        filepaths = args.filepaths
        print(f'Processing {len(filepaths)} files.')
        print(filepaths)
    else:
        # Get filepaths if not provided
        filepaths = get_multiple_filepaths(dir=os.path.join(cfg['DATA_ROOT'], cfg['data_folder'], 'raw'))

    for ismrmrd_data_fullpath in filepaths:
        main(ismrmrd_data_fullpath, cfg)