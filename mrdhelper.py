from constants import ECG_WAVEFORM_ID, PILOTTONE_WAVEFORM_ID, PILOTTONE_CH
import ismrmrd
import numpy as np
import os
import fnmatch
import warnings
import time

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
    
def read_waveforms(filepath: str, dataset_name: str = 'dataset') -> list[ismrmrd.Waveform]:
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
    print(f'Reading {filepath}...')
    with ismrmrd.File(filepath) as mrd:
        waveform_list = mrd[dataset_name].waveforms[:]
        print(f'There are {len(waveform_list)} waveforms in the dataset. Reading...')
        xml_header = mrd[dataset_name].header
    print('Waveforms read.')

    return waveform_list, xml_header

def waveforms_asarray(waveform_list: list[ismrmrd.Waveform], ecg_channel: int=0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
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
    # TODO: There is no check if the waveform is empty. This will cause an error.
    ecg_waveform = []
    ecg_trigs = []
    resp_waveform = []
    ecg_init_timestamp = 0
    pt_init_timestamp = 0
    for wf in waveform_list:
        if wf.getHead().waveform_id == ECG_WAVEFORM_ID:
            ecg_waveform.append(wf.data[ecg_channel,:])
            ecg_trigs.append(wf.data[4,:])
            if ecg_init_timestamp == 0:
                ecg_init_timestamp = wf.time_stamp
                ecg_sampling_time = wf.getHead().sample_time_us*1e-6 # [us] -> [s]
        # If there are multiple PT waveforms, last one will overwrite the previous ones.
        elif wf.getHead().waveform_id == PILOTTONE_WAVEFORM_ID:
            resp_waveform = wf.data[PILOTTONE_CH['RESP'],:]
            pt_cardiac = ((wf.data[PILOTTONE_CH['CARDIAC'],:].astype(float) - 2**31)/2**31)
            pt_cardiac_trigs = np.round(((wf.data[PILOTTONE_CH['CARDIAC_TRIGGERS'],:] - 2**31)/2**31)).astype(int)
            pt_cardiac_derivative = ((wf.data[PILOTTONE_CH['CARDIAC_DERIVATIVE'],:].astype(float) - 2**31)/2**31)
            pt_derivative_trigs = np.round((wf.data[PILOTTONE_CH['DERIVATIVE_TRIGGERS'],:] - 2**31)/2**31).astype(int)

            pt_sampling_time = wf.getHead().sample_time_us*1e-6
            pt_init_timestamp = wf.time_stamp

    if len(ecg_waveform) == 0:
        warnings.warn('No ECG waveform found.')
        ecg_ = None
    else:
        ecg_waveform = (np.asarray(np.concatenate(ecg_waveform, axis=0), dtype=float)-2048)
        ecg_waveform = ecg_waveform/np.percentile(ecg_waveform, 99.9)
        ecg_trigs = (np.concatenate(ecg_trigs, axis=0)/2**14).astype(int)
        time_ecg = np.arange(ecg_waveform.shape[0])*ecg_sampling_time + ecg_init_timestamp*2.5e-3
        ecg_ = {'time_ecg': time_ecg, 'ecg_waveform': ecg_waveform, 'ecg_trigs': ecg_trigs, 'ecg_sampling_time': ecg_sampling_time, 'ecg_init_timestamp': ecg_init_timestamp}
    
    if len(resp_waveform) == 0:
        warnings.warn('No PT waveform found.')
        return ecg_, None
    
    time_pt = np.arange(resp_waveform.shape[0])*pt_sampling_time + pt_init_timestamp*2.5e-3
    pt_ = {'time_pt': time_pt, 'resp_waveform': resp_waveform, 'pt_cardiac': pt_cardiac, 'pt_cardiac_trigs': pt_cardiac_trigs, 'pt_cardiac_derivative': pt_cardiac_derivative, 'pt_derivative_trigs': pt_derivative_trigs, 'pt_sampling_time': pt_sampling_time, 'pt_init_timestamp': pt_init_timestamp}

    return ecg_, pt_

def read_mrd(ismrmrd_data_fullpath: str) -> tuple[list[ismrmrd.Acquisition], list[ismrmrd.Waveform], ismrmrd.xsd.ismrmrdHeader]:
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
    start = time.time()
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
    
    return acq_list, wf_list, hdr

def read_adj(ismrmrd_noise_fullpath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, ismrmrd.xsd.ismrmrdHeader]:
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