"""
Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.
Author: Bilal Tasdelen
"""
from doctest import OutputChecker
import os
import h5py
import numpy as np
import subprocess
import argparse
from ui.selectionui import get_filepath
from PySide6.QtCore import QDir
import ismrmrd

parser = argparse.ArgumentParser(description='Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.')
parser.add_argument('start', nargs='?', help='Acquisition to start.', default=0, type=int)
parser.add_argument('end', nargs='?', help='Acquisition to end.', default=100, type=int)
args = parser.parse_args()
nstart = args.start
nend = args.end

ismrmrd_data_fullpath = get_filepath(dir=QDir.homePath())
print(f'File selected: {ismrmrd_data_fullpath}')
raw_file_ = os.path.basename(ismrmrd_data_fullpath)
print(f'Raw file: {raw_file_}')
data_dir_path = os.path.dirname(ismrmrd_data_fullpath)
print(f'Data directory path: {data_dir_path}')

print(f'Reading {ismrmrd_data_fullpath}...')
with ismrmrd.Dataset(ismrmrd_data_fullpath) as dset:
    n_acq = dset.number_of_acquisitions()
    print(f'There are {n_acq} acquisitions in the file. Reading...')

    acq_list = []
    for ii in range(n_acq):
        acq_list.append(dset.read_acquisition(ii))

    n_wf = dset.number_of_waveforms()
    print(f'There are {n_wf} waveforms in the dataset. Reading...')

    wf_list = []
    for ii in range(n_wf):
        wf_list.append(dset.read_waveform(ii))
    
    hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

output_data_fullpath = os.path.join(data_dir_path, raw_file_[:-3] + '_truncated.h5')
print(f'Creating truncated dataset {output_data_fullpath}...')
with ismrmrd.Dataset(output_data_fullpath, create_if_needed=True) as new_dset:
    for acq_ in acq_list[nstart:nend]:
        new_dset.append_acquisition(acq_)

    for wave_ in wf_list:
        new_dset.append_waveform(wave_)

    new_dset.write_xml_header(ismrmrd.xsd.ToXML(hdr))

print('Done.')
