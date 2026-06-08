'''
This script is used to run the entire pipeline for a list of files.
It extracts the pilottone and editer data, runs the reconstruction server on both, and saves the output.
It will use the config.toml for the configuration.
'''
import argparse
import os

import tomllib

import main_editer_correct
import main_pilottone_extract
from pylottone.reconstruction import send_to_recon_server

try:
    from pylottone.selectionui import get_multiple_filepaths
except ImportError:
    get_multiple_filepaths = None

# Check if filepaths are provided as arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('-f', '--filepaths', nargs='+', help='List of filepaths to process')
argparser.add_argument('-c', '--config', nargs='?', default='config.toml', help='Config file to be used during processing.')

args = argparser.parse_args()

with open(args.config, 'rb') as cf:
    cfg = tomllib.load(cf)

if args.filepaths:
    filepaths = args.filepaths
    print(f'Processing {len(filepaths)} files.')
    print(filepaths)
else:
    # Get filepaths if not provided
    if get_multiple_filepaths is None:
        raise ImportError(
            "UI dependencies not installed. Either provide filepaths with -f or install with: pip install pylottone[ui]"
        )
    filepaths = get_multiple_filepaths(dir=os.path.join(cfg['DATA_ROOT'], cfg['data_folder'], 'raw'))

for ismrmrd_data_fullpath in filepaths:
    outpath_pt = main_pilottone_extract.main(ismrmrd_data_fullpath, cfg)
    outpath_editer = main_editer_correct.main(ismrmrd_data_fullpath, cfg)
    send_to_recon_server.main(outpath_pt, cfg)
    send_to_recon_server.main(outpath_editer, cfg)