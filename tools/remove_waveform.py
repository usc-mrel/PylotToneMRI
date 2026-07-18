"""
Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.
Author: Bilal Tasdelen
"""
import argparse
import glob
import os
from pathlib import Path
import h5py
import numpy as np
import subprocess

try:
    from pylottone.selectionui import get_multiple_filepaths
except ImportError:
    get_multiple_filepaths = None

parser = argparse.ArgumentParser(description='Remove waveform by its ID from an ISMRMRD dataset. Especially useful when we want to replace custom waveforms.')
parser.add_argument('waveform_id', nargs='?', help='Waveform ID. Default is 1025.', default=1025, type=int)
parser.add_argument(
    '-f',
    '--files',
    nargs='+',
    help='One or more MRD/H5 files to process. Supports glob patterns such as "*.h5".',
)
parser.add_argument("-r", '--repack', help='Repack h5 file. May reduce file size, but takes some time.',action='store_true')
args = parser.parse_args()
repack_file = args.repack
waveform_id = args.waveform_id

def _expand_input_paths(file_inputs: list[str]) -> list[str]:
    expanded_paths: list[str] = []

    for file_input in file_inputs:
        expanded_input = os.path.expanduser(file_input)
        matches = glob.glob(expanded_input, recursive=True)
        if matches:
            expanded_paths.extend(matches)
        else:
            expanded_paths.append(expanded_input)

    unique_paths = list(dict.fromkeys(expanded_paths))
    return [str(Path(path).expanduser()) for path in unique_paths]


def _remove_waveform_from_file(ismrmrd_data_fullpath: str, waveform_id: int, repack: bool) -> None:
    ismrmrd_data_fullpath = str(Path(ismrmrd_data_fullpath).expanduser())
    print(f'File selected: {ismrmrd_data_fullpath}')
    raw_file_ = os.path.basename(ismrmrd_data_fullpath)
    print(f'Raw file: {raw_file_}')
    data_dir_path = os.path.dirname(ismrmrd_data_fullpath)
    print(f'Data directory path: {data_dir_path}')

    with h5py.File(ismrmrd_data_fullpath, 'a') as f:
        wfs = f['/dataset/waveforms']
        wfs2 = []
        print(f'Finding and removing waveforms with ID {waveform_id}...')
        for wf in wfs:
            if wf[0][8] != waveform_id:
                wfs2.append(wf)

        del f['/dataset/waveforms']

        f.create_dataset('/dataset/waveforms', maxshape=(None,), chunks=True, data=np.array(wfs2))

    if repack is True:
        print('Repacking the file, this may take a while....')

        rpk_fname = '2rpk_' + raw_file_
        raw_file_path = os.path.join(data_dir_path, raw_file_)
        rpk_file_path = os.path.join(data_dir_path, rpk_fname)
        os.rename(raw_file_path, rpk_file_path)
        print(subprocess.run(['h5repack', rpk_fname, raw_file_], cwd=data_dir_path))
        os.remove(rpk_file_path)


if args.files:
    ismrmrd_data_fullpaths = _expand_input_paths(args.files)
    print(f'Processing {len(ismrmrd_data_fullpaths)} file(s).')
else:
    if get_multiple_filepaths is None:
        raise ImportError(
            "UI dependencies not installed. Either provide filepaths with -f or install with: pip install pylottone[ui]"
        )
    ismrmrd_data_fullpaths = get_multiple_filepaths(dir=os.path.expanduser('~'))

if not ismrmrd_data_fullpaths:
    raise SystemExit('No input files selected.')

for ismrmrd_data_fullpath in ismrmrd_data_fullpaths:
    _remove_waveform_from_file(ismrmrd_data_fullpath, waveform_id, repack_file)