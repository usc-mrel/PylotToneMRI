# PyPilotTone
Pilot Tone Python Toolbox with an emphasis on high-amplitude pilot tone and spiral imaging.

# Installation

Create a conda/mamba environment using `environment.yml`:

```
conda env create -f environment.yml
```

Eeasiest way to run the notebooks is using VS Code, and following the prompts asking for the installation of `jupyterlab` and `ipykernel`. Otherwise, those two packages need to be installed manually.

# Usage

Most inputs and outputs are in [ISMRMRD](https://ismrmrd.readthedocs.io/en/latest/) format. Some scripts do modify the input raw data, so in case something goes wrong, it is important to back-up original raw data. For a more detailed explanation of raw data, please refer to subsection [Directory Hierarchy for Raw Data](#directory-hierarchy-for-raw-data).

All scripts use the same configuration file, `config.toml`, for specifying input data and some important parameters. An example config file, `example_config.toml` can be used as the template.

Following are the important notebooks/scripts for the typical usage:

`main_pilottone_extract.ipynb`: This is the main notebook that loads the raw data, extracts pilot tone, and saves the extracted waveforms back into the same raw data as an MRD waveform. Later parts of this notebook assumes ECG is acquired in the raw data, so if it is not the case, one can also run the parts that extract pilot tone, without comparing to ECG. **Note:** This script assumes the acquisition is from Pulseq, and loads the data as such.

`run_server_recon.ipynb`: This small notebook configures and runs the MRD client, which in turn sends the waveforms and the raw data to the reconstruction server. A server toolkit that includes some reconstructions, including several ones capable of processing pilot tone is provided [here](https://github.com/usc-mrel/python-ismrmrd-server).

`pilottone_ecg_jitter.py`: This script loads waveforms in a raw data, and compares ECG and pilot tone quality using jitter metric.

`respiratory_from_image.py`: From reconstructed images in MRD format, asks the user to place a line plot, which is then used for estimating respiratory waveform from the reconstructed images.

`remove_waveform.py`: Removes the waveforms with given ID from an MRD raw data. Can also repack the data to reclaim space.

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