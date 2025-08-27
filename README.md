# HAPTIC: High-amplitude Pilot Tone with Interference Cancellation
PylotToneMRI is a Python Toolbox with an emphasis on high-amplitude pilot tone applied to spiral MR imaging.

# Installation

Create an isolated environment using your favorite software (`uv`, `venv`, `conda`, etc.). Example for `venv`:

```bash
python -m venv ${venv_name}
source ${venv_name}/bin/activate
```

### For development:
Clone this repository and navigate into it:
```bash
git clone https://github.com/usc-mrel/PylotToneMRI.git
cd PylotToneMRI
```

Install via `pip` in dev mode:

```bash
pip install -e .
```
## For usage as a library:

It can be directly installed from GitHub via `pip`:
```bash
pip install git+https://github.com/usc-mrel/PylotToneMRI
```

**Note:** Toolbox is only tested for Python>=3.10 and Python<=3.12.

# Usage

Most inputs and outputs are in [ISMRMRD](https://ismrmrd.readthedocs.io/en/latest/) format. Some scripts do modify the input raw data, so in case something goes wrong, it is important to back-up original raw data. For a more detailed explanation of raw data, please refer to subsection [Directory Hierarchy for Raw Data](#directory-hierarchy-for-raw-data).

Most scripts take the configuration file in `toml` format, for specifying input data and some important parameters. An example config file, `example_config.toml` can be used as the template.

Example for supplying the config file: `python script_to_run.py -c config_file_path.toml`. If no input config is supplied, `config.toml` is used as the default path.

Most scripts also accepts list of inputs to be processed as a list of filepaths using `-f` or `--filepaths` switch. If no filepath is provided, scripts will open a UI to select one or multiple files for processing.
 
## Summary of important notebooks/scripts

### Following are under `examples/` directory:

`main_pilottone_extract.py`: This is the main script that loads the raw data, extracts pilot tone, and saves the extracted waveforms back into the same raw data as an MRD waveform. Later parts of this notebook assumes ECG is acquired in the raw data, so if it is not the case, one can also run the parts that extract pilot tone, without comparing to ECG. **Note:** This script provides a pipeline for a spiral acquisiton, as implemented here: [Real Time Spiral sequences in PyPulseq](https://github.com/usc-mrel/rtspiral_pypulseq). Can be used as a template for other type of acquisitions.

`main_editer_correct.py`: This script will process the raw data using EDITER and saves the corrected raw data. 

`run_all.py`: A convenience script that applies pilot tone extraction and EDITER processing on the supplied raw data, and send the results to reconstruction server.

`main_process_reference.py`: Can be used to process raw data with no PT.

### Following are under `tools/` directory:

`respiratory_from_image.py`: From reconstructed images in MRD format, asks the user to place a line plot, which is then used for estimating respiratory waveform from the reconstructed images.

`remove_waveform.py`: Removes the waveforms with given ID from an MRD raw data. Can also repack the data to reclaim space.

`truncate_acquisitions.py`: Truncates the raw data to shorten the acquisiton time.

-----

`send_to_recon_server.py`: This script is part of the package and also provides an entry point that can be called as `send_to_recon_server`. It configures and runs the MRD client, which in turn sends the waveforms and the raw data to the reconstruction server. A server toolkit that includes some reconstructions, including several ones capable of processing pilot tone is provided [here](https://github.com/usc-mrel/python-ismrmrd-server).

There are several notebooks under `notebooks/` directory for mostly debugging or interactive usage purposes.

## Directory Hierarchy for Raw Data

The code expect raw data in the following hierarchy:

    DATA_ROOT\
        |- data_folder\
            |- SEQUENCEHASH.mat
            |- raw\
                |- h5\
                    |- raw_file.h5
                |- h5_proc\
                    |- raw_file_editer.h5
                    |- raw_file_ptsub.h5
                |- noise\
                    |- noise_raw_file.h5

`SEQUENCEHASH.mat` is the metadata file generated during sequence design. Refer to [rtspiral_pypulseq](https://github.com/usc-mrel/rtspiral_pypulseq) for the details.

Processed raw data (either by EDITER or model subtraction) is put into `h5_proc` folder, with appropriate suffix to the file name.