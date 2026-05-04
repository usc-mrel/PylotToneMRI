# %%
import argparse
import logging
import os
from typing import Union

import numpy as np
import rtoml
from scipy.io import loadmat

import pylottone as pt
import pylottone.mrdhelper as mrdhelper
from pylottone.selectionui import get_multiple_filepaths
from pylottone.signal import angle_dependant_filtering, find_freq_qifft
from pylottone.pt import est_dtft
from scipy.sparse.linalg import svds

import ismrmrd

def _get_string_param_var(hdr, param_name):
    for str_param in hdr.userParameters.userParameterString:
        if str_param.name == param_name:
            return str_param.value
    return None

# %%
# Read the data in
def load_trajectory(metadata, metafile_paths: list[str]) -> dict | None:
    # get the k-space trajectory based on the metadata hash.
    for str_param in metadata.userParameters.userParameterString:
        if str_param.name == "tSequenceVariant":
            traj_name = str_param.value[:32] # Get first 32 chars, because a bug sometimes causes this field to have /OSP added to the end.
            break
    else:
        logging.error("Sequence hash is not found in metadata user parameters.")
        return None

    # load the .mat file containing the trajectory
    # Search for the file in the metafile_paths
    for path in metafile_paths:
        metafile_fullpath = os.path.join(path, traj_name + ".mat")
        if os.path.isfile(metafile_fullpath):
            logging.info(f"Loading metafile {traj_name} from {path}...")
            traj = loadmat(metafile_fullpath, squeeze_me=True)
            return traj
    else:
        logging.error(f"Trajectory file {traj_name}.mat not found in specified paths.")
        return None

def extract_pt_from_cartesian(ksp_measured, fov: float, acq: ismrmrd.Acquisition,
                        f_diff: float, df: float, dt: float,
                        freq_correction: bool = True, return_complex: bool = False) -> tuple[np.ndarray, np.ndarray]:

        n_acq = ksp_measured.shape[1]
        # ================================
        # Demodulate any shifts
        # ================================
        phase_mod_rads = pt.trajectory.calc_cartesian_fovshift_phase(fov, dt, acq)
        # Apply the negative of the phase
        ksp_measured_ = ksp_measured*phase_mod_rads

        fcorrmin = 0
        if freq_correction:
            fcorrmin = find_freq_qifft(ksp_measured_[:,:,:], df, f_diff, 3e3, 4, (2))

        ksp_window = np.ones(ksp_measured_.shape[0])
        ksp_measured_ = ksp_measured_*ksp_window[:,None,None]

        time_acq = np.arange(0, ksp_measured_.shape[0])*dt

        if freq_correction:
            w_corr = np.exp(-2j*np.pi*np.arange(ksp_measured_.shape[0])[:,None]*dt*fcorrmin[None,:])[:,:,None]
        else:
            w_corr = 1

        X = np.reshape(ksp_measured_*w_corr, (ksp_measured_.shape[0], -1))
        b, _,_ = svds(X, k=1)

        B = np.reshape(b @ np.conj(b.T) @ X, (ksp_measured_.shape[0], n_acq, -1))*np.conj(w_corr)

        ksp_ptsubbed_ = ksp_measured_ - B

        _, pt_sig_fit = est_dtft(time_acq, ksp_measured_, np.array([f_diff])-fcorrmin, ksp_window)

        ksp_ptsubbed = ksp_ptsubbed_*np.conj(phase_mod_rads)

        if return_complex:
            pt_sig = np.squeeze(pt_sig_fit)
        else:
            pt_sig_fit = np.abs(pt_sig_fit)
            pt_sig = np.squeeze(pt_sig_fit - np.mean(pt_sig_fit, axis=1, keepdims=True))
        return pt_sig, ksp_ptsubbed

def main(ismrmrd_data_fullpath, cfg) -> Union[str, None]:
    f_pt = cfg['pilottone']['pt_freq']

    data_dir = os.path.join('/', *(os.path.dirname(ismrmrd_data_fullpath).split('/')[:-2]))
    print(f"Data dir: {data_dir}")

    raw_file = ismrmrd_data_fullpath.split('/')[-1]
    ismrmrd_data_fullpath, ismrmrd_noise_fullpath = mrdhelper.siemens_mrd_finder(data_dir, '', raw_file)

    acq_list, wf_list, hdr = mrdhelper.read_mrd(ismrmrd_data_fullpath)
    n_acq = len(acq_list)

    # Check if Cartesian

    is_cartesian = False
    seq_var_str = _get_string_param_var(hdr, "tSequenceVariant")
    if seq_var_str is None:
        logging.error("Sequence variant string parameter not found in metadata.")
        return
    if seq_var_str == 'SK\\SS':
        is_cartesian = True
        logging.info("Cartesian acquisition...")
        dt = acq_list[0].sample_time_us*1e-6
    else:
        # get the k-space trajectory based on the metadata hash.
        traj = load_trajectory(hdr, [data_dir])
        if traj is None:
            logging.error("Failed to load trajectory.")
            return

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

    ## Process ECG waveform

    # Switch to this function entirely after testing
    wf_dict = mrdhelper.waveforms_asarray2(wf_list)
    if 'ecg' in wf_dict:
        ecg_waveform = wf_dict['ecg'][1][:,0]
        ecg_waveform = pt.check_waveform_polarity(ecg_waveform, 0.5, method='width')*ecg_waveform
        time_ecg = wf_dict['ecg'][0] - acq_list[0].acquisition_time_stamp*2.5e-3
        ecg_trigs = wf_dict['ecg'][1][:, -1]
    else:
        print('No ECG waveform found.')
    if 'pulseox' in wf_dict:
        time_pulseox = wf_dict['pulseox'][0] - acq_list[0].acquisition_time_stamp*2.5e-3
        pulseox_waveform = wf_dict['pulseox'][1]
        pulseox_trigs = wf_dict['pulseox'][2]
    else:
        print('No Pulse Oximeter waveform found.')
    if 'ext1' in wf_dict:
        time_ext1 = wf_dict['ext1'][0] - acq_list[0].acquisition_time_stamp*2.5e-3
        ext1_trigs = wf_dict['ext1'][1]
    else:
        print('No external trigger waveform found.')
    if 'resp' in wf_dict:
        time_resp = wf_dict['resp'][0] - acq_list[0].acquisition_time_stamp*2.5e-3
        resp_waveform = wf_dict['resp'][1]
    else:
        print('No respiratory waveform found.')

    # %% [markdown]
    # ## PT correction

    # %%

    f_diff = f0 - f_pt

    if is_cartesian:
        fov = hdr.encoding[0].encodedSpace.fieldOfView_mm.x*1e-3 # [m]
        pt_raw, _ = extract_pt_from_cartesian(ksp_measured, fov, acq_list[0], f_diff, df, dt, freq_correction=True, return_complex = True)
        pt_sig = np.abs(pt_raw)
        pt_sig = np.squeeze(pt_sig - np.mean(pt_sig, axis=1, keepdims=True))

    else:
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

    _, _,pt_respiratory, Vcard, accept_list_cardiac, pt_cardiac = pt.calibrate_pt(pt_sig, f_samp, pt_extract_params)

    # S = (np.linalg.inv(Vcard)[:,0][:,None]@pt_cardiac[None,:]).T


    print('Saving waveforms...')
    np.savez(os.path.join(cfg['DATA_ROOT'], "waveforms/", cfg['data_folder'], f"{ismrmrd_data_fullpath.split('/')[-1][:-3]}_ptwaveforms_{f_pt/1e6}MHz.npz"),
                pt_raw=pt_raw,
                pt_respiratory=pt_respiratory,
                pt_cardiac=pt_cardiac,
                # S=S,
                Vcard=Vcard,
                coil_name=coil_name[mri_coils],
                accept_list_cardiac=accept_list_cardiac,
                time_pt=time_pt,
                time_ecg=time_ecg if 'ecg' in wf_dict else None,
                ecg_trigs=ecg_trigs if 'ecg' in wf_dict else None,
                ecg_waveform=ecg_waveform if 'ecg' in wf_dict else None,
                time_pulseox=time_pulseox if 'pulseox' in wf_dict else None,
                pulseox_waveform=pulseox_waveform if 'pulseox' in wf_dict else None,
                pulseox_trigs=pulseox_trigs if 'pulseox' in wf_dict else None,
                time_ext1=time_ext1 if 'ext1' in wf_dict else None,
                ext1_trigs=ext1_trigs if 'ext1' in wf_dict else None,
                time_resp=time_resp if 'resp' in wf_dict else None,
                resp_waveform=resp_waveform if 'resp' in wf_dict else None,
                )



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
