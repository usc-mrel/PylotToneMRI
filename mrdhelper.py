from constants import ECG_WAVEFORM_ID, PILOTTONE_WAVEFORM_ID, PILOTTONE_CH
import ismrmrd
import numpy as np
import os
import fnmatch
import warnings

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
    with ismrmrd.Dataset(filepath, dataset_name=dataset_name) as dset:
        n_wf = dset.number_of_waveforms()
        print(f'There are {n_wf} waveforms in the dataset. Reading...')

        waveform_list = []
        for ii in range(n_wf):
            waveform_list.append(dset.read_waveform(ii))
        xml_header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
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