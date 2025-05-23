import ismrmrd
import h5py
import numpy as np
from ui.selectionui import get_selection
import matplotlib.pyplot as plt
import matplotlib.animation as animation

filepath = 'output_recons/vol0912_20240626/viewsharing_meas_MID00285_FID16394_pulseq2D_fire_spiralga_100mV_24MHz_mdlsub.mrd'

# Load the MRD file

# If run multiple times, the recon.mrd file will have multiple reconstructed images
# Find the most recent recon run
with h5py.File(filepath, 'r') as d:
    dset_names = list(d.keys())
    print(f'File {filepath} contains {len(dset_names)} groups (reconstruction runs):')

if len(dset_names) > 1:
    group = get_selection(dset_names)
else:
    group = dset_names[0]

t_sel = [3, 5]
video_name = f'{filepath[:-4]}_{t_sel[0]}s_{t_sel[1]}s_{group}.mp4'
wf_list = []
with ismrmrd.Dataset(filepath, group, False) as dset:
    subgroups = dset.list()

    # Images are organized by series number in subgroups that start with 'images_'
    imgGroups = [group for group in list(subgroups) if (group.find('image_') != -1)]
    print(f'Group {group} contains {len(imgGroups)} image series:')
    print(' ', '\n  '.join(imgGroups))

    # Read and append images
    imgs = []
    time_stamps = []
    n = dset.number_of_images(imgGroups[0])
    for ii in range(n):
        frame = dset.read_image(imgGroups[0], ii)
        imgs.append(np.squeeze(frame.data))
        time_stamps.append(frame.acquisition_time_stamp)
    img_0 = dset.read_image(imgGroups[0], 0)

    n_wf = dset.number_of_waveforms()

    for ii in range(n_wf):
        wf_list.append(dset.read_waveform(ii))
    
imgs = np.array(imgs).transpose(1, 2, 0)
time_frame = np.array(time_stamps, dtype=float)*2.5e-3
time_frame -= time_frame[0]
# Update frame rate
framerate = 1/(np.mean(np.diff(time_frame)))
I_sel_I = ((time_frame > t_sel[0]) & (time_frame < t_sel[1])).squeeze()
Nframes = np.sum(I_sel_I)


fig, ax = plt.subplots()
ax.set_axis_off()
im_ax = []
for ii in range(Nframes):
    im_ = ax.imshow(imgs[:, :, I_sel_I[0]+ii], cmap='gray', animated=True)
    if ii == 0:
        ax.imshow(imgs[:, :, I_sel_I[0]+ii], cmap='gray', animated=True)
    im_ax.append([im_])

ani = animation.ArtistAnimation(fig, im_ax, interval=1e3/framerate, blit=True)
MWriter = animation.FFMpegWriter(fps=framerate)
plt.show()
print(f'Saving in {video_name}...')
ani.save(video_name, writer=MWriter)

