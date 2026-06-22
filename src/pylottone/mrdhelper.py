import copy
import fnmatch
import logging
import os
import re
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import loadmat

from pylottone.constants import ECG_WAVEFORM_ID, PULSEOX_WAVEFORM_ID, PILOTTONE_WAVEFORM_ID, PILOTTONE_CH, EXT1_WAVEFORM_ID, RESPPT_WAVEFORM_ID

if TYPE_CHECKING:
    import ismrmrd

try:
    import ismrmrd
except ImportError:
    ismrmrd = None

logging.basicConfig(level=logging.INFO)


def _require_ismrmrd() -> None:
    if ismrmrd is None:
        raise ImportError("ismrmrd is required for MRD file I/O. Install with: pip install pylottone[mrd]")

def siemens_mrd_finder(data_root: str, data_folder: str, raw_file: str, h5folderext: str = '', rawfile_ext: str = '') -> str:
    """
    Finds the full paths of the Siemens MRD data file and noise file.

    Parameters
    ----------
    data_root : str
        The root directory of the data.
    data_folder : str
        The folder containing the data.
    raw_file : str
        The name or identifier of the raw file.
    h5folderext : str, optional
        The extension of the h5 folder. Defaults to ''.

    Returns
    -------
    ismrmrd_data_fullpath : str
        String containing the full path of the MRD data file.
    ismrmrd_noise_fullpath : str
        String containing the full path of the noise file.
    Raises
    ------
    Warning
        If the file cannot be found.
    """

    data_dir_path = os.path.join(data_root, data_folder, f'raw/h5{h5folderext}')
    noise_dir_path = os.path.join(data_root, data_folder, 'raw/noise')

    if raw_file.isnumeric():
        raw_file_ = fnmatch.filter(os.listdir(data_dir_path), f'meas_MID*{raw_file}*{rawfile_ext}.h5')[0]
    elif raw_file.startswith('meas_MID'):
        raw_file_ = raw_file
    else:
        warnings.warn('Could not find the file.', warnings.Error)
    
    ismrmrd_data_fullpath = os.path.join(data_dir_path, raw_file_)
    ismrmrd_noise_fullpath = os.path.join(noise_dir_path, f'noise_{raw_file_}')

    return ismrmrd_data_fullpath, ismrmrd_noise_fullpath
    
def read_waveforms(filepath: str, dataset_name: str = 'dataset') -> "tuple[list[ismrmrd.Waveform], ismrmrd.xsd.ismrmrdHeader]":
    '''Reads all waveforms from an ISMRMRD dataset.
        Parameters
        ----------
        filename : str
            MRD File name.
        
        Returns
        -------
        waveform_list : list
            List of waveforms.
        xml_header : ismrmrd.xsd.ismrmrdHeader
            XML header.
    '''
    _require_ismrmrd()
    print(f'Reading {filepath}...')
    with ismrmrd.File(filepath) as mrd:
        waveform_list = mrd[dataset_name].waveforms[:]
        print(f'There are {len(waveform_list)} waveforms in the dataset. Reading...')
        xml_header = mrd[dataset_name].header
    print('Waveforms read.')

    return waveform_list, xml_header

def waveforms_asarray(waveform_list: "list[ismrmrd.Waveform]", ecg_channel: int=0, ext_as_ecg: bool=False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    '''Converts a list of waveforms to numpy arrays for ECG and PT.
        Parameters
        ----------
        waveform_list : list
            List of waveforms.
        
        Returns
        -------
        ecg : dict[np.array, np.array, float]
            Numpy array of waveforms.
        pt : dict[np.array, np.array, np.array, np.array, np.array, float]
    '''

    ecg_waveform = []
    ecg_trigs = []
    ext1_ts = []
    ecg_ts = []
    resp_waveform = []
    respbeat_waveform = []
    ecg_init_timestamp = 0
    pt_init_timestamp = 0
    respbeat_init_timestamp = 0
    respbeat_ts = []
    for wf in waveform_list:
        if ext_as_ecg and wf.getHead().waveform_id == EXT1_WAVEFORM_ID:
            ecg_waveform.append(wf.data[0,:]+2048)
            ecg_trigs.append(wf.data[1,:]*8)
            if ecg_init_timestamp == 0:
                ecg_init_timestamp = wf.time_stamp
                ecg_sampling_time = wf.getHead().sample_time_us*1e-6
            ext1_ts.append(wf.time_stamp + np.arange(wf.data.shape[1])*ecg_sampling_time/2.5e-3)

        elif not ext_as_ecg and wf.getHead().waveform_id == ECG_WAVEFORM_ID:
            ecg_waveform.append(wf.data[ecg_channel,:])
            ecg_trigs.append(wf.data[4,:])
            if ecg_init_timestamp == 0:
                ecg_init_timestamp = wf.time_stamp
                ecg_sampling_time = wf.getHead().sample_time_us*1e-6 # [us] -> [s]
            ecg_ts.append(wf.time_stamp + np.arange(wf.data.shape[1])*ecg_sampling_time/2.5e-3)
        # If there are multiple PT waveforms, last one will overwrite the previous ones.
        elif wf.getHead().waveform_id == PILOTTONE_WAVEFORM_ID:
            resp_waveform = wf.data[PILOTTONE_CH['RESP'],:]
            pt_cardiac = ((wf.data[PILOTTONE_CH['CARDIAC'],:].astype(float) - 2**31)/2**31)
            pt_cardiac_trigs = np.round(((wf.data[PILOTTONE_CH['CARDIAC_TRIGGERS'],:] - 2**31)/2**31)).astype(int)
            pt_cardiac_derivative = ((wf.data[PILOTTONE_CH['CARDIAC_DERIVATIVE'],:].astype(float) - 2**31)/2**31)
            pt_derivative_trigs = np.round((wf.data[PILOTTONE_CH['DERIVATIVE_TRIGGERS'],:] - 2**31)/2**31).astype(int)

            pt_sampling_time = wf.getHead().sample_time_us*1e-6
            pt_init_timestamp = wf.time_stamp
        elif wf.getHead().waveform_id == RESPPT_WAVEFORM_ID:
            respbeat_waveform.append(wf.data[0,:])
            if respbeat_init_timestamp == 0:
                respbeat_init_timestamp = wf.time_stamp
                respbeat_sampling_time = wf.getHead().sample_time_us*1e-6
            respbeat_ts.append(wf.time_stamp + np.arange(wf.data.shape[1])*respbeat_sampling_time/2.5e-3)

    if len(ecg_waveform) == 0:
        warnings.warn('No ECG waveform found.')
        ecg_ = None
    else:
        ecg_waveform = (np.asarray(np.concatenate(ecg_waveform, axis=0), dtype=float)-2048)
        ecg_waveform = ecg_waveform/np.percentile(ecg_waveform, 99.9)
        ecg_trigs = (np.concatenate(ecg_trigs, axis=0)/2**14).astype(int)
        if ext_as_ecg:
            time_ecg = np.concatenate(ext1_ts, axis=0)*2.5e-3
        else:
            time_ecg = np.arange(ecg_waveform.shape[0])*ecg_sampling_time + ecg_init_timestamp*2.5e-3
        ecg_ = {'time_ecg': time_ecg, 'ecg_waveform': ecg_waveform, 'ecg_trigs': ecg_trigs, 'ecg_sampling_time': ecg_sampling_time, 'ecg_init_timestamp': ecg_init_timestamp}
    
    if len(resp_waveform) == 0 and len(respbeat_waveform) == 0:
        warnings.warn('No PT waveform found.')
        return ecg_, None
    
    pt_ = {}
    if len(respbeat_waveform) > 0:
        respbeat_waveform = (np.asarray(np.concatenate(respbeat_waveform, axis=0), dtype=float))
        respbeat_waveform -= np.mean(respbeat_waveform)
        respbeat_waveform /= np.percentile(respbeat_waveform, 99.9)
        time_respbeat = np.concatenate(respbeat_ts, axis=0)*2.5e-3
        pt_['time_respbeat'] = time_respbeat
        pt_['respbeat_waveform'] = respbeat_waveform
        pt_['respbeat_sampling_time'] = respbeat_sampling_time
        pt_['respbeat_init_timestamp'] = respbeat_init_timestamp

    if len(resp_waveform) > 0:
        time_pt = np.arange(resp_waveform.shape[0])*pt_sampling_time + pt_init_timestamp*2.5e-3
        pt_.update({'time_pt': time_pt, 'resp_waveform': resp_waveform, 'pt_cardiac': pt_cardiac, 'pt_cardiac_trigs': pt_cardiac_trigs, 'pt_cardiac_derivative': pt_cardiac_derivative, 'pt_derivative_trigs': pt_derivative_trigs, 'pt_sampling_time': pt_sampling_time, 'pt_init_timestamp': pt_init_timestamp})

    return ecg_, pt_

def waveforms_asarray2(wf_list: "list[ismrmrd.Waveform]") -> dict:
    '''An alternative function to sort and convert a list of waveforms to numpy arrays.
        Parameters
        ----------
        wf_list : list[ismrmrd.Waveform]
            List of waveforms.
        
        Returns
        -------
        waveform_dict : dict
            Dictionary of waveforms. Possible keys are 'ecg', 'pulseox', 'resp', 'ext1'. 
            Keys will only be present if the corresponding waveform is found.
    '''
    ecg = []
    resp_pt = []
    ext1 = []
    t_ext1 = []
    pulseox = []
    t_init_pox = 0
    pulseox_sample_time = 0
    t_init_ecg = 0
    ecg_sample_time = 0
    t_init_resp = 0
    resp_sample_time = 0
    t_init_ext1 = 0
    ext1_sample_time = 0

    for wf in wf_list:
        if wf.waveform_id == ECG_WAVEFORM_ID:
            ecg.append(wf.data)
            if t_init_ecg == 0:
                t_init_ecg = wf.time_stamp*2.5e-3
                ecg_sample_time = wf.sample_time_us*1e-6
        elif wf.waveform_id == PULSEOX_WAVEFORM_ID:
            pulseox.append(wf.data)
            if t_init_pox == 0:
                t_init_pox = wf.time_stamp*2.5e-3
                pulseox_sample_time = wf.sample_time_us*1e-6
        elif wf.waveform_id == RESPPT_WAVEFORM_ID:
            resp_pt.append(wf.data)
            if t_init_resp == 0:
                t_init_resp = wf.time_stamp*2.5e-3
                resp_sample_time = wf.sample_time_us*1e-6
        elif wf.waveform_id == EXT1_WAVEFORM_ID:
            ext1.append(wf.data)
            if t_init_ext1 == 0:
                t_init_ext1 = wf.time_stamp*2.5e-3
                ext1_sample_time = wf.sample_time_us*1e-6
            t_ext1.extend(wf.time_stamp*2.5e-3 + np.arange(wf.data.shape[1])*ext1_sample_time)

    waveform_dict = {}
    ecg = np.concatenate(ecg, axis=1).T if len(ecg) > 0 else np.array([])
    if len(ecg) > 0:
        is_flat = np.all(ecg[:, 0] == ecg[0, 0], axis=0) or \
            np.all(ecg[:, 1] == ecg[0, 1], axis=0) or \
            np.all(ecg[:, 2] == ecg[0, 2], axis=0) or \
            np.all(ecg[:, 3] == ecg[0, 3], axis=0)                  # Check if the waveform is flat

        if not np.isnan(ecg).all() and not is_flat:
            ecg = ecg[:, :].astype(np.float32)
            ecg -= np.percentile(ecg, 5, axis=0)
            ecg /= np.max(np.abs(ecg), axis=0, keepdims=True)
            t_ecg = np.arange(ecg.shape[0])*ecg_sample_time + t_init_ecg
            waveform_dict['ecg'] = (t_ecg, ecg)

    pulseox = np.concatenate(pulseox, axis=1).T if len(pulseox) > 0 else np.array([])
    if len(pulseox) > 0:
        is_flat = np.all(np.diff(pulseox, axis=0) == 0)
        if not np.isnan(pulseox).all() and not is_flat and not np.all(pulseox[:,1]):
            pulseox_trigs = pulseox[:, 1].astype(np.int32)
            pulseox_trigs[pulseox_trigs > 0] = 1
            pulseox = pulseox[:, 0].astype(np.float32)
            pulseox -= np.percentile(pulseox, 5, axis=0)
            pulseox /= np.max(np.abs(pulseox), axis=0, keepdims=True)
            t_pox = np.arange(pulseox.shape[0])*pulseox_sample_time + t_init_pox
            waveform_dict['pulseox'] = (t_pox, pulseox, pulseox_trigs)

    resp_pt = np.concatenate(resp_pt, axis=1).T if len(resp_pt) > 0 else np.array([])
    if len(resp_pt) > 0:
        resp_pt = resp_pt[:, 0].astype(np.float32)
        resp_pt -= np.mean(resp_pt, axis=0, keepdims=True)
        resp_pt /= np.max(np.abs(resp_pt), axis=0, keepdims=True)
        t_resp = np.arange(resp_pt.shape[0])*resp_sample_time + t_init_resp
        waveform_dict['resp'] = (t_resp, resp_pt)

    ext1 = np.concatenate(ext1, axis=1).T if len(ext1) > 0 else np.array([])
    if len(ext1) > 0:
        is_flat = np.all(np.diff(ext1, axis=0) == 0)
        if not np.isnan(ext1).all() and not is_flat:
            ext1 = ext1[:, 0].astype(np.float32)
            ext1[ext1 > 0] = 1
            ext1_trgs = (np.concatenate(([0],np.diff(ext1, axis=0))) > 0).astype(np.float32)
            ext1 = np.concatenate((ext1[:, np.newaxis], ext1_trgs[:, np.newaxis]), axis=1)
            # t_ext1 = np.arange(ext1.shape[0])*ext1_sample_time + t_init_ext1
            t_ext1 = np.array(t_ext1)
            waveform_dict['ext1'] = (t_ext1, ext1)

    return waveform_dict

def read_mrd(ismrmrd_data_fullpath: str) -> "tuple[list[ismrmrd.Acquisition], list[ismrmrd.Waveform], ismrmrd.xsd.ismrmrdHeader]":
    '''Reads an ISMRMRD dataset.
        Parameters
        ----------
        ismrmrd_data_fullpath : str
            MRD File name.'
        
        Returns
        -------
        acq_list : list
            List of acquisitions.
        wf_list : list
            List of waveforms.
        hdr : ismrmrd.xsd.ismrmrdHeader
            XML header.
        '''
    _require_ismrmrd()
    start = time.time()
    print('=' * 50)
    print(f'Reading {ismrmrd_data_fullpath}...')

    with ismrmrd.File(ismrmrd_data_fullpath, mode='r') as mrd:
        if mrd['dataset'].has_acquisitions():
            print('Reading acquisitions...')
            # Read all acquisitions from the dataset
            acq_list = mrd['dataset'].acquisitions[:]
            print(f'There are {len(acq_list)} acquisitions in the dataset.')
        if mrd['dataset'].has_waveforms():
            wf_list = mrd['dataset'].waveforms[:]
            print(f'There are {len(wf_list)} waveforms in the dataset.')
        else:
            wf_list = []
            print('No waveforms found in the dataset.')
        if mrd['dataset'].has_header():
            hdr = mrd['dataset'].header
        else:
            warnings.warn('No header found in the dataset. Using default header.')
            hdr = ismrmrd.xsd.ismrmrdHeader()

    end = time.time()
    print(f'Finished reading {ismrmrd_data_fullpath} in {end - start:.2f} seconds.')
    print('=' * 50)

    return acq_list, wf_list, hdr

def read_adj(ismrmrd_noise_fullpath: str) -> "tuple[np.ndarray, np.ndarray, np.ndarray, ismrmrd.xsd.ismrmrdHeader]":
    '''Reads the coil sensitivity maps and noise from an ISMRMRD dataset.
        Parameters
        ----------
        ismrmrd_noise_fullpath : str
            MRD File name.
        
        Returns
        -------
        data_csm : np.ndarray
            Coil sensitivity maps.
        data_bc : np.ndarray
            Body coil data.
        hdr : ismrmrd.xsd.ismrmrdHeader
            XML header.
    '''

    _require_ismrmrd()
    acq_list_noise, _, hdr_noise = read_mrd(ismrmrd_noise_fullpath)
    acq_csm = [acq_ for acq_ in acq_list_noise if acq_.isFlagSet(ismrmrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA)]
    acq_noise = [acq_.data for acq_ in acq_list_noise if acq_.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)]
    noise = np.transpose(np.asarray(acq_noise), (1,0,2)).reshape((acq_noise[0].shape[0], -1))

    print(f"Number of CSMs: {len(acq_csm)}")
    n_pe = hdr_noise.encoding[0].encodingLimits.kspace_encoding_step_1.maximum + 1
    n_par = hdr_noise.encoding[0].encodingLimits.kspace_encoding_step_2.maximum + 1
    print(f"CSM shape: {acq_csm[0].data.shape[0]}, {acq_csm[0].data.shape[1]}, {n_pe}, {n_par}")

    data_csm = np.zeros(
        (acq_csm[0].available_channels, acq_csm[0].data.shape[1], 
            n_pe, n_par), 
        dtype=np.complex64)
    data_bc = np.zeros(
        (2, acq_csm[0].data.shape[1], 
            n_pe, n_par), 
        dtype=np.complex64)
    for acq_ in acq_csm:
        if acq_.active_channels == 2:
            data_bc[:, :, acq_.idx.kspace_encode_step_1, acq_.idx.kspace_encode_step_2] = acq_.data
        else:
            data_csm[:, :, acq_.idx.kspace_encode_step_1, acq_.idx.kspace_encode_step_2] = acq_.data

    # Pad in k-space to match the encoded size
    data_bc = np.pad(data_bc, ((0,), (0,), ((hdr_noise.encoding[0].encodedSpace.matrixSize.y - n_pe)//2,), ((hdr_noise.encoding[0].encodedSpace.matrixSize.z - n_par)//2,)))
    data_csm = np.pad(data_csm, ((0,), (0,), ((hdr_noise.encoding[0].encodedSpace.matrixSize.y - n_pe)//2,), ((hdr_noise.encoding[0].encodedSpace.matrixSize.z - n_par)//2,)))

    return data_csm, data_bc, noise, hdr_noise
def get_volt_from_protoname(proto_name: str) -> float:
    """
    Extract the PT volt if written in the protocol name as _XXXV_ or _XXXmV_.
    
    Parameters:
    proto_name (str): Protocol name containing the voltage information.
    
    Returns:
    pt_volt (float): Extracted voltage in volts (V). Returns NaN if no voltage information is found.
    """
    proto_fields = proto_name.lower().split('_')
    pt_volt = np.nan
    
    for fld in proto_fields:
        if 'mv' in fld:
            vval = re.findall(r'\d+\.?\d*', fld)
            if not vval:
                continue
            pt_volt = float(vval[0]) * 1e-3
            break

        if 'v' in fld:
            vval = re.findall(r'\d+\.?\d*', fld)
            if not vval:
                continue
            pt_volt = float(vval[0])
            break

    if np.isnan(pt_volt):
        print('Could not extract PT voltage from the protocol name.')

    return pt_volt

def save_processed_raw_data(output_data_fullpath: str, 
                               hdr: "ismrmrd.xsd.ismrmrdHeader", acq_list: "list[ismrmrd.Acquisition]", wf_list: "list[ismrmrd.Waveform]", 
                               ksp_processed: np.ndarray, mri_coils: np.ndarray, pt_wf: "ismrmrd.Waveform | None" = None, user_params: dict = {}) -> None:

    _require_ismrmrd()

    # Update new parameters to XML header.
    new_hdr = copy.deepcopy(hdr)

    for param_name, param_value in user_params.items():
        if type(param_value) is str:
            new_hdr.userParameters.userParameterString.append(ismrmrd.xsd.userParameterStringType(param_name, param_value))
        elif type(param_value) is int:
            new_hdr.userParameters.userParameterLong.append(ismrmrd.xsd.userParameterLongType(param_name, param_value))
        else:
            print(f'Parameter {param_name} with type {type(param_value)} is not supported. Skipping...')

    # new_hdr.userParameters.userParameterString.append(ismrmrd.xsd.userParameterStringType('processing', 'ModelSubtraction'))
    new_hdr.acquisitionSystemInformation.coilLabel = [hdr.acquisitionSystemInformation.coilLabel[ch_i] for ch_i in mri_coils]
    new_hdr.acquisitionSystemInformation.receiverChannels = len(new_hdr.acquisitionSystemInformation.coilLabel)

    # Copy and fix acquisition objects
    new_acq_list = []
    remove_os = True if ksp_processed.shape[0]*2 == acq_list[0].getHead().number_of_samples else False

    for acq_i, acq_ in enumerate(acq_list):
        new_head = copy.deepcopy(acq_.getHead())
        new_head.active_channels = len(new_hdr.acquisitionSystemInformation.coilLabel)
        new_head.available_channels = len(new_hdr.acquisitionSystemInformation.coilLabel)
        if remove_os:
            new_head.number_of_samples = ksp_processed.shape[0]
            new_head.center_sample = 5

        new_acq_list.append(ismrmrd.Acquisition(head=new_head, data=np.ascontiguousarray(ksp_processed[:,acq_i,:].squeeze().T.astype(np.complex64))))

    with ismrmrd.Dataset(output_data_fullpath, create_if_needed=True) as new_dset:
        for acq_ in new_acq_list:
            new_dset.append_acquisition(acq_)

        for wave_ in wf_list:
            new_dset.append_waveform(wave_)

        if pt_wf:
            new_dset.append_waveform(pt_wf)

        new_dset.write_xml_header(ismrmrd.xsd.ToXML(new_hdr))

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