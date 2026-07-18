# %%
import argparse
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import time
from pathlib import Path

import ismrmrd
import numpy as np
import tomllib
from scipy.io import loadmat
from scipy.signal.windows import tukey

import pylottone.mrdhelper as mrdhelper
from pylottone.editer import apply_editer, autopick_sensing_coils

try:
    from pylottone.selectionui import get_multiple_filepaths
except ImportError:
    get_multiple_filepaths = None
from pylottone.trajectory import remove_readout_os
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

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
    kernel_temporal_footprint = cfg['editer']['temporal_footprint']
    denoise_sniffers = cfg['editer']['denoise_sniffers']
    denoise_rank = int(cfg['editer'].get('denoise_rank', 1))
    debug_plots = cfg['editer']['show_plots']
    
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
    logging.info(f'Reading {ismrmrd_noise_fullpath}...')
    with ismrmrd.Dataset(ismrmrd_noise_fullpath) as dset_noise:
        n_cal_acq = dset_noise.number_of_acquisitions()
        logging.info(f'There are {n_cal_acq} acquisitions in the file. Reading...')

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

        logging.info('Prewhitening the raw data...')
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


    logging.info(f"Coils to be used as sniffers: {coil_name[sensing_coils.astype(int)]}")

    ksp_measured = data[:,:,mri_coils]*ksp_window
    ksp_sniffer  = data[:,:,sensing_coils]*ksp_window


    # Denoising is applied per temporal window inside EDITER.
    ksp_sniffer2 = ksp_sniffer

    # %%

    # ===============================================================
    # Prepare EDITER weights and inputs
    # ===============================================================
    start_time = time.time()

    logging.info('Running EDITER...')
    dk = [cfg['editer']['kernel_length'], 0]
    logging.debug(f'EDITER parameters: {cfg["editer"]["kernel_length"]}, {kernel_temporal_footprint}, {gpu_device}')
    # Weighted lsq by square of the ksp to avoid biasing result by strong k-space center
    w = np.concatenate((np.zeros((pre_discard, coord.shape[2])), np.sqrt(coord[0,:,:]**2 + coord[1,:,:]**2))).astype(np.float32)
    n_pe = round(kernel_temporal_footprint*1e3 / hdr.sequenceParameters.TR[0])  # convert time to number of acquisitions

    editer_params = {
        'grouping_method': "uniform",  # "uniform", "corr_orig"
        'max_lines_per_group': n_pe,   # Max number of lines in a group
        'dk': dk,                      # Convolution kernel size in kx and ky directions
        'denoise_sniffers': denoise_sniffers,  # Whether to denoise sniffer coils using per-window low-rank SVD
        'denoise_rank': denoise_rank,  # Number of singular components to keep during sniffer denoising
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

    logging.info(f"Elapsed time: {time.time() - start_time} seconds")

    if debug_plots:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox
        from pylottone.signal import to_hybrid_kspace

        max_slice_idx = ksp_measured.shape[2] - 1
        initial_slice_idx = 0

        measured_hybrid = np.abs(to_hybrid_kspace(ksp_measured))
        corrected_hybrid = np.abs(to_hybrid_kspace(ksp_emicorr))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(measured_hybrid.shape[0], d=dt))
        window_size = 10

        def get_freq_window_indices(center_freq):
            center_idx = int(np.argmin(np.abs(freq_axis - center_freq)))
            half_window = window_size // 2
            start_idx = max(0, center_idx - half_window)
            end_idx = min(freq_axis.size, start_idx + window_size)
            start_idx = max(0, end_idx - window_size)
            return np.arange(start_idx, end_idx)

        center_window_idx = get_freq_window_indices(0.0)
        interference_window_idx = get_freq_window_indices(f_diff)

        def get_slice_metrics(slice_idx):
            measured_slice = measured_hybrid[:, :, slice_idx]
            corrected_slice = corrected_hybrid[:, :, slice_idx]

            measured_center = float(np.mean(np.abs(measured_slice[center_window_idx, :])))
            measured_interference = float(np.mean(np.abs(measured_slice[interference_window_idx, :])))
            corrected_center = float(np.mean(np.abs(corrected_slice[center_window_idx, :])))
            corrected_interference = float(np.mean(np.abs(corrected_slice[interference_window_idx, :])))

            measured_ratio = measured_interference / measured_center if measured_center != 0 else np.nan
            corrected_ratio = corrected_interference / corrected_center if corrected_center != 0 else np.nan

            return {
                'measured_center': measured_center,
                'measured_interference': measured_interference,
                'measured_ratio': measured_ratio,
                'corrected_center': corrected_center,
                'corrected_interference': corrected_interference,
                'corrected_ratio': corrected_ratio,
            }

        def format_metrics(slice_idx):
            metrics = get_slice_metrics(slice_idx)
            return (
                f'Slice {slice_idx}\n'
                f'Before: center={metrics["measured_center"]:.4g}, '
                f'interference={metrics["measured_interference"]:.4g}, '
                f'ratio={metrics["measured_ratio"]:.4g}\n'
                f'After:  center={metrics["corrected_center"]:.4g}, '
                f'interference={metrics["corrected_interference"]:.4g}, '
                f'ratio={metrics["corrected_ratio"]:.4g}\n'
                f'All channels avg ratio before={measured_channel_ratio_avg:.4g}, '
                f'after={corrected_channel_ratio_avg:.4g}'
            )

        channel_metrics = [get_slice_metrics(slice_idx) for slice_idx in range(max_slice_idx + 1)]
        measured_channel_ratio_avg = float(np.nanmean([metrics['measured_ratio'] for metrics in channel_metrics]))
        corrected_channel_ratio_avg = float(np.nanmean([metrics['corrected_ratio'] for metrics in channel_metrics]))

        def get_slice_clim(slice_idx):
            slice_values = np.concatenate(
                (
                    measured_hybrid[:, :, slice_idx].ravel(),
                    corrected_hybrid[:, :, slice_idx].ravel(),
                )
            )
            vmin, vmax = np.percentile(slice_values, [1, 99])
            if vmin == vmax:
                vmax = vmin + 1.0
            return float(vmin), float(vmax)

        n_lines = ksp_measured.shape[1]
        tr_sec = hdr.sequenceParameters.TR[0] * 1e-3
        time_axis = np.linspace(0.0, (n_lines - 1) * tr_sec, n_lines)
        x_extent = [time_axis[0], time_axis[-1]] if n_lines > 1 else [0.0, tr_sec]
        y_extent = [0, measured_hybrid.shape[0] - 1]
        footprint_marks = np.arange(
            kernel_temporal_footprint,
            x_extent[1] + kernel_temporal_footprint,
            kernel_temporal_footprint,
        )

        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.18, hspace=0.25)

        corrected_title_suffix = (
            f'tf={kernel_temporal_footprint:g}s, '
            f'klen={editer_params["dk"][0]}, '
            f'denoise={denoise_sniffers}'
            f' (rank={denoise_rank})' if denoise_sniffers else ''
        )

        clim_vmin, clim_vmax = get_slice_clim(initial_slice_idx)

        measured_img = axes[0].imshow(
            measured_hybrid[:, :, initial_slice_idx],
            aspect='auto',
            origin='lower',
            cmap='gray',
            vmin=clim_vmin,
            vmax=clim_vmax,
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
        )
        axes[0].set_title(f'Measured k-space (slice {initial_slice_idx})')
        axes[0].set_ylabel('Hybrid k-space sample')

        corrected_img = axes[1].imshow(
            corrected_hybrid[:, :, initial_slice_idx],
            aspect='auto',
            origin='lower',
            cmap='gray',
            vmin=clim_vmin,
            vmax=clim_vmax,
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
        )
        axes[1].set_title(f'Corrected k-space (slice {initial_slice_idx}) | {corrected_title_suffix}')
        axes[1].set_xlabel('Time [s]')
        axes[1].set_ylabel('Hybrid k-space sample')

        for mark in footprint_marks:
            axes[0].axvline(mark, color='tab:red', linestyle='--', linewidth=1.0, alpha=0.6)
            axes[1].axvline(mark, color='tab:red', linestyle='--', linewidth=1.0, alpha=0.6)

        fig.colorbar(measured_img, ax=axes, location='right')

        metrics_text = fig.text(
            0.52,
            0.02,
            format_metrics(initial_slice_idx),
            ha='center',
            va='bottom',
            fontsize=9,
            family='monospace',
        )

        control_state = {'updating': False, 'slice_idx': initial_slice_idx}

        text_ax = fig.add_axes([0.20, 0.06, 0.12, 0.05])
        prev_ax = fig.add_axes([0.08, 0.06, 0.08, 0.05])
        next_ax = fig.add_axes([0.34, 0.06, 0.08, 0.05])

        slice_textbox = TextBox(text_ax, 'Slice', initial=str(initial_slice_idx))
        prev_button = Button(prev_ax, '<')
        next_button = Button(next_ax, '>')

        def apply_slice(slice_idx, sync_text=True):
            slice_idx = max(0, min(max_slice_idx, int(slice_idx)))
            clim_vmin, clim_vmax = get_slice_clim(slice_idx)
            control_state['slice_idx'] = slice_idx
            metrics_text.set_text(format_metrics(slice_idx))
            metrics = get_slice_metrics(slice_idx)
            measured_img.set_data(measured_hybrid[:, :, slice_idx])
            corrected_img.set_data(corrected_hybrid[:, :, slice_idx])
            measured_img.set_clim(clim_vmin, clim_vmax)
            corrected_img.set_clim(clim_vmin, clim_vmax)
            axes[0].set_title(f'Measured k-space (slice {slice_idx})')
            axes[1].set_title(f'Corrected k-space (slice {slice_idx}) | {corrected_title_suffix}')
            logging.info(
                'Debug slice %d | before center=%.4g interference=%.4g ratio=%.4g | '
                'after center=%.4g interference=%.4g ratio=%.4g | '
                'avg ratio before=%.4g after=%.4g',
                slice_idx,
                metrics['measured_center'],
                metrics['measured_interference'],
                metrics['measured_ratio'],
                metrics['corrected_center'],
                metrics['corrected_interference'],
                metrics['corrected_ratio'],
                measured_channel_ratio_avg,
                corrected_channel_ratio_avg,
            )
            if sync_text and slice_textbox.text != str(slice_idx):
                control_state['updating'] = True
                slice_textbox.set_val(str(slice_idx))
                control_state['updating'] = False
            fig.canvas.draw_idle()

        def on_text_submit(text):
            if control_state['updating']:
                return
            try:
                slice_idx = int(text)
            except ValueError:
                return
            apply_slice(slice_idx, sync_text=False)

        def on_prev(event):
            apply_slice(control_state['slice_idx'] - 1)

        def on_next(event):
            apply_slice(control_state['slice_idx'] + 1)

        slice_textbox.on_submit(on_text_submit)
        prev_button.on_clicked(on_prev)
        next_button.on_clicked(on_next)
        plt.show()

    # %% [markdown]
    # # Create a new MRD dataset, use the original as a template, and write corrected k-space into it.

    # %%
    n_samp = ksp_emicorr.shape[0]

    if remove_os:
        remove_readout_os(ksp_emicorr)
        n_samp = n_samp // 2

    output_dir_fullpath = os.path.join(DATA_ROOT, DATA_DIR, 'raw', 'h5_proc')
    output_data_fullpath = os.path.join(output_dir_fullpath, f'{raw_file[:-3]}_editer.h5')
    logging.info('Saving to ' + output_data_fullpath)

    Path.mkdir(Path(output_dir_fullpath), exist_ok=True)
    user_params = {'processing': 'EDITER',
                   'EDITER_kx': editer_params['dk'][0],
                   'EDITER_ky': editer_params['dk'][1],
                   'EDITER_denoiseSniffers': int(denoise_sniffers),
                   'EDITER_temporalFootprint': editer_params['max_lines_per_group'],
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
    
    with open(args.config, 'rb') as cf:
        cfg = tomllib.load(cf)

    if args.filepaths:
        filepaths = args.filepaths
        logging.info(f'Processing {len(filepaths)} files.')
        logging.info(filepaths)
    else:
        # Get filepaths if not provided
        if get_multiple_filepaths is None:
            raise ImportError(
                "UI dependencies not installed. Either provide filepaths with -f/--filepaths \n"
                "or install with: pip install pylottone[ui]"
            )
        filepaths = get_multiple_filepaths(dir=os.path.join(cfg['DATA_ROOT'], cfg['data_folder'], 'raw'))

    for ismrmrd_data_fullpath in filepaths:
        main(ismrmrd_data_fullpath, cfg)
