import numpy as np
from scipy.optimize import fminbound
from scipy.signal.windows import blackman
from scipy.ndimage import zoom
from scipy import ndimage
from pylottone.signal import cfftn, cifftn, rssq
import h5py
import matplotlib.pyplot as plt
from pylottone.vis import ndv
import xml.etree.ElementTree as ET

def circle3d(radius, n_points=100) -> np.ndarray:
    """Generate a 3D circle at a given position with a specified radius."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    x = np.zeros_like(y)
    return np.array([x, y, z]).T

def rotate_circle(circle: np.ndarray, azimuth: float, polar: float) -> np.ndarray:
    """Rotate a circle in 3D space based on azimuth and polar angles."""
    azimuth_rad = np.radians(azimuth)
    polar_rad = np.radians(polar)
    
    # Rotation matrices
    R_azimuth = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
                          [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
                          [0, 0, 1]])
    
    R_polar = np.array([[1, 0, 0],
                        [0, np.cos(polar_rad), -np.sin(polar_rad)],
                        [0, np.sin(polar_rad), np.cos(polar_rad)]])
    
    return circle @ (R_azimuth @ R_polar).T
    
def plot_quadrants(ax, array, fixed_coord, cmap, extent:float =64):
    """For a given 3d *array* plot a plane with *fixed_coord*, using four quadrants."""
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    min_val = array.min()
    max_val = array.max()

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == 'x':
            Y, Z = extent*(np.mgrid[0:ny // 2, 0:nz // 2]-ny/2)/ny
            X = (extent / 2) * np.zeros_like(Y)
            Y_offset = (i // 2) * extent / 2
            Z_offset = (i % 2) * extent / 2
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'y':
            X, Z = extent*(np.mgrid[0:nx // 2, 0:nz // 2] - nx/2)/nx
            Y = (extent / 2) * np.zeros_like(X)
            X_offset = (i // 2) * extent / 2
            Z_offset = (i % 2) * extent / 2
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'z':
            X, Y = extent*(np.mgrid[0:nx // 2, 0:ny // 2] - nx/2)/nx
            Z = (extent / 2) * np.zeros_like(X)
            X_offset = (i // 2) * extent / 2
            Y_offset = (i % 2) * extent / 2
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)


def figure_3D_array_slices(array, ax, cmap=None, extent:float =64):
    """Plot a 3d array using three intersecting centered planes."""

    ax.set_box_aspect(array.shape)
    plot_quadrants(ax, array, 'x', cmap=cmap, extent=extent)
    plot_quadrants(ax, array, 'y', cmap=cmap, extent=extent)
    plot_quadrants(ax, array, 'z', cmap=cmap, extent=extent)


class Coil:
    def __init__(self, coil_element, sensmap):
        self.name = coil_element.attrib['Name']
        self.position = np.array([float(coil_element.attrib['XPos']),
                                  float(coil_element.attrib['YPos']),
                                  float(coil_element.attrib['ZPos'])])
        self.radius = float(coil_element.attrib['Radius'])
        self.azimuth = float(coil_element.attrib['Azimuth'])
        self.polar = float(coil_element.attrib['Polar'])
        self.extent = float(coil_element.attrib['Extent'])
        self.sensmap = sensmap
    
    def __repr__(self):
        return f"Coil(position={self.position}, radius={self.radius}, azimuth={self.azimuth}, polar={self.polar})"
    
    def show(self, show_sensmap: bool = True, ax=None, callshow: bool = True):
        """Visualize the coil in 3D."""

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        if show_sensmap:
            figure_3D_array_slices(np.abs(self.sensmap.transpose(2, 1, 0)), ax, cmap='turbo', extent=self.extent)
        
        ax.text(*self.position, self.name, fontsize=12, color='red')
        circ = circle3d(self.radius, n_points=201)
        circ = rotate_circle(circ, self.azimuth, self.polar)
        circ += self.position
        ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color='blue', linewidth=2)
        X, Y, Z = np.meshgrid(np.linspace(-self.extent/2, self.extent/2, 64),
                             np.linspace(-self.extent/2, self.extent/2, 64),
                             np.linspace(-self.extent/2, self.extent/2, 64))
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        
        ax.set_title('Coil Position')
        ax.set_xlabel('X Position [mm]')
        ax.set_ylabel('Y Position [mm]')
        ax.set_zlabel('Z Position [mm]')
        # ax.set_xlim([-self.extent/2, self.extent/2])
        # ax.set_ylim([-self.extent/2, self.extent/2])
        # ax.set_zlim([-self.extent/2, self.extent/2])
        if callshow:
            plt.show()
    
    def show_sensmap(self):
        """Visualize the sensitivity map of the coil."""
        ndv(np.abs(self.sensmap))

class CoilArray(dict):
    def __init__(self, coils: list[Coil] = []):
        super().__init__()
        for coil in coils:
            if not isinstance(coil, Coil):
                raise TypeError(f"Expected Coil instance, got {type(coil)}")
            self[coil.name] = coil

        self.extent = coils[0].extent if coils else 0.0
        self.sens_shape = coils[0].sensmap.shape if coils else (0, 0, 0)
    
    def load_from_xml(self, xml_file: str, sensmap: np.ndarray):
        """Load coils from an XML file and associate them with a sensitivity map."""
        tree = ET.parse(xml_file)
        coils_xml = tree.findall('BIOTSAVARTLOOP')
        for ci, coil_element in enumerate(coils_xml):
            coil = Coil(coil_element, sensmap[ci, :, :, :])
            self[coil.name] = coil
        
        self.extent = float(coils_xml[0].attrib['Extent'])

    def show(self, show_sensmap: bool = True):
        """Visualize all coils in the array."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for coil in self.values():
            coil.show(ax=ax, show_sensmap=False, callshow=False)

        if show_sensmap:
            sensmap_rssq = rssq(np.abs([c.sensmap for c in self.values()]), axis=0).transpose(2, 1, 0)
            figure_3D_array_slices(sensmap_rssq, ax, cmap='turbo', extent=self.extent)

        plt.show()



def calculate_prewhitening(noise, scale_factor=1.0):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to
                   adjust for effective noise bandwith and difference in
                   sampling rate between noise calibration and actual measurement:
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    :returns w: Prewhitening matrix, ``[coil, coil]``, w*data is prewhitened
    '''

    noise_int = noise.reshape((noise.shape[0], noise.size//noise.shape[0]))
    M = float(noise_int.shape[1])
    dmtx = (1/(M-1))*noise_int.dot(noise_int.conj().T)
    dmtx = np.linalg.inv(np.linalg.cholesky(dmtx))
    dmtx = dmtx*np.sqrt(2)*np.sqrt(scale_factor)
    return dmtx

def apply_prewhitening(data,dmtx):
    '''Apply the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix

    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''

    s = data.shape
    return (dmtx.dot(data.reshape(data.shape[0],-1))).reshape(s)

def bc_adapt_comb(bcip: np.ndarray, bcquad: np.ndarray) -> np.ndarray:
    """
    Adaptively combine in-phase and quadrature body coil signals.
    
    Parameters
    ----------
    bcip : np.ndarray
        In-phase body coil signal
    bcquad : np.ndarray
        Quadrature body coil signal
        
    Returns
    -------
    np.ndarray
        Combined body coil signal
    """
    def fcomb(IP, QUAD, phi):
        return IP + QUAD * np.exp(1j * phi)
    
    def fpmin(phi):
        return -np.sum(np.abs(fcomb(bcip, bcquad, phi)))
    
    # Find optimal phase difference between -pi and pi
    ang_diff = fminbound(fpmin, -np.pi, np.pi)
    print(f"Optimal phase difference: {ang_diff*180/np.pi:.4f} degrees")
    # Combine signals using optimal phase
    bc_comb = fcomb(bcip, bcquad, ang_diff)
    
    return bc_comb

def get_surface_corrected_maps(im_csm: np.ndarray, im_bc: np.ndarray, mask_threshold: float=0.05) -> np.ndarray:
    """
    Compute surface corrected maps from coil sensitivity maps and body coil data.
    Parameters
    ----------
    im_csm : np.ndarray
        Coil sensitivity maps, shape: [coil, x, y, z]
    im_bc : np.ndarray
        Body coil data, shape: [2, x, y, z] (in-phase and quadrature)
    mask_threshold : float
        Max BC signal's fraction to threshold for noise, default is 0.05.
    Returns
    -------
    im_csm: np.ndarray
        Surface corrected maps, shape: [coil, x, y, z]
    """

    bc_comb = bc_adapt_comb(im_bc[0,:,:,:], im_bc[1,:,:,:])
    n_ch = im_csm.shape[0]
    # Compute raw (unprocessed) sensitivity maps
    ksize = 32
    # Create 3D Blackman window
    blackman_1d = blackman(ksize)
    blackman_2d = np.outer(blackman_1d, blackman_1d)
    blackman_3d = blackman_2d[:, :, np.newaxis] * blackman_1d[np.newaxis, np.newaxis, :]
    # Resize to 64x64x64
    kfilt = zoom(blackman_3d, (64/ksize, 64/ksize, 64/ksize))

    # Apply forward and inverse FFTs with filtering
    bc_comb_filt = cfftn(cifftn(bc_comb[None,:,:,:], axes=(1,2,3)) * kfilt, axes=(1,2,3))
    surface_img_filt = cfftn(cifftn(im_csm, axes=(1,2,3)) * kfilt, axes=(1,2,3))

    # Find maximum value across all dimensions
    mbc = np.max(np.abs(bc_comb_filt))

    # Create threshold mask and repeat for all channels
    mask2 = np.abs(bc_comb_filt) > mbc * mask_threshold
    mask2 = np.repeat(mask2, n_ch, axis=0)

    # Scale the filtered surface image 
    im_scl = surface_img_filt / bc_comb_filt
    im_scl[~mask2] = 0

    return cfftn(cifftn(im_scl, axes=(1,2,3))*kfilt, axes=(1,2,3))

def calculate_csm_walsh(img, smoothing=5, niter=3):
    '''Calculates the coil sensitivities for 2D data using an iterative version of the Walsh method

    :param img: Input images, ``[coil, y, x]``
    :param smoothing: Smoothing block size (default ``5``)
    :parma niter: Number of iterations for the eigenvector power method (default ``3``)

    :returns csm: Relative coil sensitivity maps, ``[coil, y, x]``
    :returns rho: Total power in the estimated coils maps, ``[y, x]``
    '''

    assert img.ndim == 3, "Coil sensitivity map must have exactly 3 dimensions"

    ncoils = img.shape[0]
    ny = img.shape[1]
    nx = img.shape[2]

    # Compute the sample covariance pointwise
    Rs = np.zeros((ncoils,ncoils,ny,nx),dtype=img.dtype)
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q,:,:] = img[p,:,:] * np.conj(img[q,:,:])

    # Smooth the covariance
    for p in range(ncoils):
        for q in range(ncoils):
            Rs[p,q] = smooth(Rs[p,q,:,:], smoothing)

    # At each point in the image, find the dominant eigenvector
    # and corresponding eigenvalue of the signal covariance
    # matrix using the power method
    rho = np.zeros((ny, nx))
    csm = np.zeros((ncoils, ny, nx),dtype=img.dtype)
    for y in range(ny):
        for x in range(nx):
            R = Rs[:,:,y,x]
            v = np.sum(R,axis=0)
            lam = np.linalg.norm(v)
            v = v/lam

            for iter in range(niter):
                v = np.dot(R,v)
                lam = np.linalg.norm(v)
                v = v/lam

            rho[y,x] = lam
            csm[:,y,x] = v

    return (csm, rho)


def calculate_csm_inati_iter(im, smoothing=5, niter=5, thresh=1e-3,
                             verbose=False):
    """ Fast, iterative coil map estimation for 2D or 3D acquisitions.

    Parameters
    ----------
    im : ndarray
        Input images, [coil, y, x] or [coil, z, y, x].
    smoothing : int or ndarray-like
        Smoothing block size(s) for the spatial axes.
    niter : int
        Maximal number of iterations to run.
    thresh : float
        Threshold on the relative coil map change required for early
        termination of iterations.  If ``thresh=0``, the threshold check
        will be skipped and all ``niter`` iterations will be performed.
    verbose : bool
        If true, progress information will be printed out at each iteration.

    Returns
    -------
    coil_map : ndarray
        Relative coil sensitivity maps, [coil, y, x] or [coil, z, y, x].
    coil_combined : ndarray
        The coil combined image volume, [y, x] or [z, y, x].

    Notes
    -----
    The implementation corresponds to the algorithm described in [1]_ and is a
    port of Gadgetron's ``coil_map_3d_Inati_Iter`` routine.

    For non-isotropic voxels it may be desirable to use non-uniform smoothing
    kernel sizes, so a length 3 array of smoothings is also supported.

    References
    ----------
    .. [1] S Inati, MS Hansen, P Kellman.  A Fast Optimal Method for Coil
        Sensitivity Estimation and Adaptive Coil Combination for Complex
        Images.  In: ISMRM proceedings; Milan, Italy; 2014; p. 4407.
    """

    im = np.asarray(im)
    if im.ndim < 3 or im.ndim > 4:
        raise ValueError("Expected 3D [ncoils, ny, nx] or 4D "
                         " [ncoils, nz, ny, nx] input.")

    if im.ndim == 3:
        # pad to size 1 on z for 2D + coils case
        images_are_2D = True
        im = im[:, np.newaxis, :, :]
    else:
        images_are_2D = False

    # convert smoothing kernel to array
    if isinstance(smoothing, int):
        smoothing = np.asarray([smoothing, ] * 3)
    smoothing = np.asarray(smoothing)
    if smoothing.ndim > 1 or smoothing.size != 3:
        raise ValueError("smoothing should be an int or a 3-element 1D array")

    if images_are_2D:
        smoothing[2] = 1  # no smoothing along z in 2D case

    # smoothing kernel is size 1 on the coil axis
    smoothing = np.concatenate(([1, ], smoothing), axis=0)

    ncha = im.shape[0]

    try:
        # numpy >= 1.7 required for this notation
        D_sum = im.sum(axis=(1, 2, 3))
    except:
        D_sum = im.reshape(ncha, -1).sum(axis=1)

    v = 1/np.linalg.norm(D_sum)
    D_sum *= v
    R = 0

    for cha in range(ncha):
        R += np.conj(D_sum[cha]) * im[cha, ...]

    eps = np.finfo(im.real.dtype).eps * np.abs(im).mean()
    for it in range(niter):
        if verbose:
            print("Coil map estimation: iteration %d of %d" % (it+1, niter))
        if thresh > 0:
            prevR = R.copy()
        R = np.conj(R)
        coil_map = im * R[np.newaxis, ...]
        coil_map_conv = smooth(coil_map, box=smoothing)
        D = coil_map_conv * np.conj(coil_map_conv)
        R = D.sum(axis=0)
        R = np.sqrt(R) + eps
        R = 1/R
        coil_map = coil_map_conv * R[np.newaxis, ...]
        D = im * np.conj(coil_map)
        R = D.sum(axis=0)
        D = coil_map * R[np.newaxis, ...]
        try:
            # numpy >= 1.7 required for this notation
            D_sum = D.sum(axis=(1, 2, 3))
        except:
            D_sum = im.reshape(ncha, -1).sum(axis=1)
        v = 1/np.linalg.norm(D_sum)
        D_sum *= v

        imT = 0
        for cha in range(ncha):
            imT += np.conj(D_sum[cha]) * coil_map[cha, ...]
        magT = np.abs(imT) + eps
        imT /= magT
        R = R * imT
        imT = np.conj(imT)
        coil_map = coil_map * imT[np.newaxis, ...]

        if thresh > 0:
            diffR = R - prevR
            vRatio = np.linalg.norm(diffR) / np.linalg.norm(R)
            if verbose:
                print("vRatio = {}".format(vRatio))
            if vRatio < thresh:
                break

    coil_combined = (im * np.conj(coil_map)).sum(0)

    if images_are_2D:
        # remove singleton z dimension that was added for the 2D case
        coil_combined = coil_combined[0, :, :]
        coil_map = coil_map[:, 0, :, :]

    return coil_map, coil_combined


def smooth(img, box=5):
    '''Smooths coil images

    :param img: Input complex images, ``[y, x] or [z, y, x]``
    :param box: Smoothing block size (default ``5``)

    :returns simg: Smoothed complex image ``[y,x] or [z,y,x]``
    '''

    t_real = np.zeros(img.shape)
    t_imag = np.zeros(img.shape)

    ndimage.uniform_filter(img.real,size=box,output=t_real)
    ndimage.uniform_filter(img.imag,size=box,output=t_imag)

    simg = t_real + 1j*t_imag

    return simg

def load_sensmap(filename: str = 'sensmaps.h5') -> np.ndarray:
    """Load sensitivity map from an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        m = f['maps']['magnitude']
        p = f['maps']['phase']
        n_ch = len(m)
        mag = [m[f"{i:02d}"][:] for i in range(n_ch)]
        pha = [p[f"{i:02d}"][:] for i in range(n_ch)]

    return np.asarray(mag, dtype=np.float32) * np.exp(1j * np.asanyarray(pha, dtype=np.float32))