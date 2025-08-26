import copy
from pylottone.constants import ECG_WAVEFORM_ID, PILOTTONE_WAVEFORM_ID, PILOTTONE_CH
import ismrmrd
import numpy as np
import os
import fnmatch
import warnings
import re

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
    print(f'Reading {ismrmrd_data_fullpath}...')
    with ismrmrd.Dataset(ismrmrd_data_fullpath) as dset:
        n_acq = dset.number_of_acquisitions()
        print(f'There are {n_acq} acquisitions in the file. Reading...')

        acq_list = []
        for ii in range(n_acq):
            acq_list.append(dset.read_acquisition(ii))

        try:
            n_wf = dset.number_of_waveforms()
            print(f'There are {n_wf} waveforms in the dataset. Reading...')
        except LookupError:
            n_wf = 0

        wf_list = []

        for ii in range(n_wf):
            wf_list.append(dset.read_waveform(ii))
        
        hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    
    return acq_list, wf_list, hdr

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
                               hdr: ismrmrd.xsd.ismrmrdHeader, acq_list: list[ismrmrd.Acquisition], wf_list: list[ismrmrd.Waveform], 
                               ksp_processed: np.ndarray, mri_coils: np.ndarray, pt_wf: None | ismrmrd.Waveform = None, user_params: dict = {}) -> None:

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
