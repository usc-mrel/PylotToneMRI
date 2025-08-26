# %%
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time
from pathlib import Path

import ismrmrd
import numpy as np
import rtoml
from scipy.io import loadmat
from scipy.signal.windows import tukey
from scipy.sparse.linalg import svds

import pylottone.mrdhelper as mrdhelper
from pylottone.editer import apply_editer, autopick_sensing_coils
from pylottone.selectionui import get_multiple_filepaths
from pylottone.trajectory import remove_readout_os


def process_channel_shared(args):
    """Process a single channel using shared memory arrays."""
    ch, ksp_shm_name, ksp_shape, ksp_dtype, sniffer_shm_name, sniffer_shape, sniffer_dtype, editer_params, w = args
    
    # Attach to shared memory
    ksp_shm = shared_memory.SharedMemory(name=ksp_shm_name)
    ksp_measured = np.ndarray(ksp_shape, dtype=ksp_dtype, buffer=ksp_shm.buf)
    
    sniffer_shm = shared_memory.SharedMemory(name=sniffer_shm_name)
    ksp_sniffer2 = np.ndarray(sniffer_shape, dtype=sniffer_dtype, buffer=sniffer_shm.buf)
    
    # Process the channel
    est_emi_ch, _ = apply_editer(ksp_measured[:, :, ch], ksp_sniffer2, editer_params, w)
    
    # Clean up
    ksp_shm.close()
    sniffer_shm.close()
    
    return est_emi_ch

def main(ismrmrd_data_fullpath, cfg) -> str:
    mp.set_start_method('spawn', force=True)
    DATA_ROOT = cfg['DATA_ROOT']
    DATA_DIR = cfg['data_folder']
    prewhiten = cfg['editer']['prewhiten']
    autoselect = cfg['editer']['autosniffer_select']
    gpu_device = cfg['editer']['gpu_device']
    remove_os = cfg['saving']['remove_os']
    raw_file = ismrmrd_data_fullpath.split('/')[-1]
    ismrmrd_data_fullpath, ismrmrd_noise_fullpath = mrdhelper.siemens_mrd_finder(DATA_ROOT, DATA_DIR, raw_file)

    # %%
    # Read the data in
    acq_list, wf_list, hdr = mrdhelper.read_mrd(ismrmrd_data_fullpath)
    n_acq = len(acq_list)
    # get the k-space trajectory based on the metadata hash.
    traj_name = hdr.userParameters.userParameterString[1].value

    # load the .mat file containing the trajectory
    traj = loadmat(os.path.join(DATA_ROOT, DATA_DIR, traj_name), squeeze_me=True)

    n_unique_angles = int(traj['param']['repetitions'])

    kx = traj['kx'][:,:]
    ky = traj['ky'][:,:]
    dt = float(traj['param']['dt'])
    msize = int(10 * traj['param']['fov'] / traj['param']['spatial_resolution'])
    pre_discard = int(traj['param']['pre_discard'])

    # Convert raw data and trajectory into convenient arrays
    ktraj = np.stack((kx, -ky), axis=2)
    # find max ktraj value
    kmax = np.max(np.abs(kx + 1j * ky))
    # swap 0 and 1 axes to make repetitions the first axis (repetitions, interleaves, 2)
    ktraj = np.swapaxes(ktraj, 0, 1)
    ktraj = 0.5 * (ktraj / kmax) * msize

    data = np.array([arm.data[:,:] for arm in acq_list]).transpose((2, 0, 1))
    coord = np.array([ktraj[ii%n_unique_angles,:,:] for ii in range(n_acq)], dtype=np.float32).transpose((2, 1, 0))



    # %%
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

    noise = np.transpose(np.asarray(noise_list), (1,0,2)).reshape((noise_list[0].shape[0], -1))

    if prewhiten:
        from pylottone.reconstruction.coils import (
            apply_prewhitening,
            calculate_prewhitening,
        )

        print('Prewhitening the raw data...')
        dmtx = calculate_prewhitening(noise)
        data = apply_prewhitening(np.transpose(data, (2,0,1)), dmtx).transpose((1,2,0))


    # %%

    coil_name = []

    for clbl in hdr.acquisitionSystemInformation.coilLabel:
        coil_name.append(clbl.coilName)

    coil_name = np.asarray(coil_name)


    f0 = hdr.experimentalConditions.H1resonanceFrequency_Hz


    ksp_window = tukey(data.shape[0]*2, 0.01)
    ksp_window = ksp_window[data.shape[0]:, None, None]

    # TEST auto coil selection

    f_pt = cfg['editer']['interference_freq'] # [Hz]
    f_diff = f0 - f_pt
    n_channels = data.shape[2]

    if autoselect:
        mri_coils, sensing_coils = autopick_sensing_coils(data, f_emi=f_diff, bw_emi=100e3, bw_sig=200e3, f_samp=1/dt, n_sensing=8)

    else:
        sensing_coils = np.array(cfg['editer']['sensing_coils'], dtype=int)
        mri_coils = np.arange(n_channels)
        mri_coils = mri_coils[~np.isin(mri_coils, sensing_coils)]


    print(f"Coils to be used as sniffers: {coil_name[sensing_coils.astype(int)]}")

    ksp_measured = data[:,:,mri_coils]*ksp_window
    ksp_sniffer  = data[:,:,sensing_coils]*ksp_window


    # %%
    ## PCA sniffer coils before weighting and subtracting from k-space to "denoise" them. Compare SNR with and without.

    U, S, V = svds(ksp_sniffer.reshape((ksp_sniffer.shape[0]*ksp_sniffer.shape[1], -1)), k=1)
    ksp_sniffer2 = (U@np.diag(S))@V
    ksp_sniffer2 = ksp_sniffer2.reshape((ksp_sniffer.shape[0], ksp_sniffer.shape[1], ksp_sniffer.shape[2]))

    # %%

    # ===============================================================
    # Prepare EDITER weights and inputs
    # ===============================================================
    start_time = time.time()

    print('Running EDITER...')
    dk = [3, 0]

    w = np.concatenate((np.zeros((pre_discard, coord.shape[2])), np.sqrt(coord[0,:,:]**2 + coord[1,:,:]**2))).astype(np.float32)
    n_pe = round(1000 / hdr.sequenceParameters.TR[0])  # 4 * Nlines / Nrep

    editer_params = {
        'grouping_method': "uniform",  # "uniform", "corr_orig"
        'max_lines_per_group': n_pe,   # Max number of lines in a group
        'dk': dk,                      # Convolution kernel size in kx and ky directions
        'gpu': gpu_device,             # Use GPU acceleration
    }

    chs = range(ksp_measured.shape[2])

    # Create shared memory for large arrays
    ksp_shm = shared_memory.SharedMemory(create=True, size=ksp_measured.nbytes)
    ksp_shared = np.ndarray(ksp_measured.shape, dtype=ksp_measured.dtype, buffer=ksp_shm.buf)
    ksp_shared[:] = ksp_measured[:]
    
    sniffer_shm = shared_memory.SharedMemory(create=True, size=ksp_sniffer2.nbytes)
    sniffer_shared = np.ndarray(ksp_sniffer2.shape, dtype=ksp_sniffer2.dtype, buffer=sniffer_shm.buf)
    sniffer_shared[:] = ksp_sniffer2[:]
    
    # Prepare arguments for each process
    process_args = [
        (ch, ksp_shm.name, ksp_measured.shape, ksp_measured.dtype,
         sniffer_shm.name, ksp_sniffer2.shape, ksp_sniffer2.dtype,
         editer_params, w)
        for ch in chs
    ]

    try:
        with mp.Pool(processes=len(chs)//2) as pool:
            results = pool.map(process_channel_shared, process_args)
    finally:
        # Clean up shared memory
        ksp_shm.close()
        ksp_shm.unlink()
        sniffer_shm.close()
        sniffer_shm.unlink()

    emi_hat = np.stack(results, axis=2)
    ksp_emicorr = ksp_measured - emi_hat

    print(f"Elapsed time: {time.time() - start_time} seconds")

    # %% [markdown]
    # # Create a new MRD dataset, use the original as a template, and write corrected k-space into it.

    # %%
    n_samp = ksp_emicorr.shape[0]

    if remove_os:
        remove_readout_os(ksp_emicorr)
        n_samp = n_samp // 2


    output_dir_fullpath = os.path.join(DATA_ROOT, DATA_DIR, 'raw', 'h5_proc')
    output_data_fullpath = os.path.join(output_dir_fullpath, f'{raw_file[:-3]}_editer.h5')
    print('Saving to ' + output_data_fullpath)

    Path.mkdir(Path(output_dir_fullpath), exist_ok=True)
    user_params = {'processing': 'EDITER',
                   'EDITER_kx': editer_params['dk'][0],
                   'EDITER_ky': editer_params['dk'][1],
                   'EDITER_maxNoLines': editer_params['max_lines_per_group'],
                   'EDITER_groupingAlgo': editer_params['grouping_method']
                    }
    mrdhelper.save_processed_raw_data(output_data_fullpath, hdr, acq_list, wf_list, ksp_emicorr, mri_coils, user_params=user_params)
    
    return output_data_fullpath

        
if __name__ == '__main__':

    # Check if filepaths are provided as arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filepaths', nargs='+', help='List of filepaths to process')
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
