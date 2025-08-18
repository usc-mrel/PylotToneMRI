import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle
import ipywidgets as widgets
from ipywidgets import Layout
import numpy as np
from typing import List, Tuple, Optional
from IPython.display import display
from mpl_image_segmenter import ImageSegmenter

def get_circle_radius(sphere_radius: float, d: float) -> float:
    """
    Calculate the radius of a circle on a spherical shell at distance d from center.
    Uses the Pythagorean theorem: r^2 + d^2 = R^2, where:
    - R is the sphere radius
    - d is the distance from the center
    - r is the radius of the circle at that distance
    
    Parameters
    ----------
    sphere_radius : float
        Radius of the sphere
    d : float
        Distance from the center along the axis
        
    Returns
    -------
    float
        Radius of the circle at distance d
    """
    if abs(d) > sphere_radius:
        return 0.0
    return np.sqrt(sphere_radius**2 - d**2)

class SpherePlacer:
    def __init__(self, image_data: np.ndarray, vmin: float = None, vmax: float = None):
        """
        Initialize interactive slice viewer with clickable ROIs.
        
        Parameters
        ----------
        image_data : np.ndarray
            3D image data array (z, y, x)
        vmin : float, optional
            Minimum value for display window
        vmax : float, optional
            Maximum value for display window
        """
        self.image = image_data
        self.nx, self.ny, self.nz = image_data.shape
        
        if vmin is None:
            vmin = np.percentile(np.abs(image_data), 5)
        if vmax is None:
            vmax = np.percentile(np.abs(image_data), 95)
            
        self.vmin = vmin
        self.vmax = vmax
        
        self.fig, self.axs = plt.subplots(1, 2)
        self.fig.set_size_inches(10, 5)
        
        self._setup_sliders()
        self._setup_image_objects()
        self._setup_circles()
        self._setup_callbacks()
        
        self.axs[0].set_title('Axial')
        self.axs[1].set_title('Coronal')
        self.axs[0].axis('off')
        self.axs[1].axis('off')
        
        plt.tight_layout()
        
        self.sphere_center = (self.nx//2, self.ny//2, self.nz//2)

    def _setup_sliders(self):
        """Initialize the slice selection sliders"""
        self.slice_slider_axial = widgets.IntSlider(
            value=self.ny//2, 
            min=0, 
            max=self.ny-1, 
            step=1, 
            description='Axial Slice',
            continuous_update=True
        )
        
        self.slice_slider_coronal = widgets.IntSlider(
            value=self.nx//2, 
            min=0, 
            max=self.nx-1, 
            step=1, 
            description='Coronal Slice',
            continuous_update=True
        )
        
    def _setup_image_objects(self):
        """Initialize the image display objects"""
        self.imobj = []
        self.imobj.append(self.axs[0].imshow(
            self.image[:, self.ny//2, :],
            cmap='gray',
            vmin=self.vmin,
            vmax=self.vmax
        ))
        self.imobj.append(self.axs[1].imshow(
            self.image[self.nx//2, :, :],
            cmap='gray',
            vmin=self.vmin,
            vmax=self.vmax
        ))
        
    def _setup_circles(self):
        """Initialize the clickable circles"""
        self.circles = []
        for ax in self.axs:
            circle = Circle((32, 32), 10, color='red', fill=True, alpha=0.5)
            ax.add_patch(circle)
            self.circles.append(circle)
            
    def _setup_callbacks(self):
        """Set up all callback functions"""
        self.slice_slider_axial.observe(
            lambda change: self._update_slice(change['new'], self.imobj[0], 0),
            names='value'
        )
        self.slice_slider_coronal.observe(
            lambda change: self._update_slice(change['new'], self.imobj[1], 1),
            names='value'
        )
        self.fig.canvas.mpl_connect(
            'button_press_event',
            self._onclick
        )
        
    def _update_slice(self, val: int, imobj: matplotlib.image.AxesImage, slice_index: int):
        """Update the displayed slice"""
        if slice_index == 0:
            imobj.set_data(self.image[:, val, :])
            self.circles[0].set_radius(
                get_circle_radius(10, self.sphere_center[2] - val)
            )
        elif slice_index == 1:
            imobj.set_data(self.image[val, :, :])
            self.circles[1].set_radius(
                get_circle_radius(10, self.sphere_center[1] - val)
            )
            
    def _onclick(self, event: matplotlib.backend_bases.MouseEvent):
        """Handle click events for circle placement"""
        if event.inaxes is not None:
            ax = event.inaxes
            axs_idx = self.axs.tolist().index(ax)
            print(f"Clicked on {ax.get_title()}, {axs_idx} at ({event.xdata:.1f}, {event.ydata:.1f})")
            self.circles[axs_idx].set_center((event.xdata, event.ydata))
            slc_i_ax = self.slice_slider_axial.value
            slc_i_cor = self.slice_slider_coronal.value
            # Set the other common coordinate
            if axs_idx == 0:
                self.sphere_center = (event.xdata, event.ydata, slc_i_ax)
                self.circles[1].set_center((event.xdata, slc_i_ax))
                self.circles[1].set_radius(
                    get_circle_radius(10, event.ydata-slc_i_cor)
                )
                self.circles[0].set_radius(10)
            else:
                self.sphere_center = (event.xdata, slc_i_cor, event.ydata)
                self.circles[0].set_center((event.xdata, slc_i_cor))
                self.circles[0].set_radius(
                    get_circle_radius(10, event.ydata-slc_i_ax)
                )
                self.circles[1].set_radius(10)
            
    def show(self):
        """Display the interactive viewer"""
        display(widgets.VBox([self.slice_slider_axial, self.slice_slider_coronal]))
        plt.show()
        
    def get_circle_centers(self) -> List[Tuple[float, float]]:
        """
        Get the current circle center coordinates
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (x,y) coordinates for each circle
        """
        return [circle.get_center() for circle in self.circles]
    
    def get_sphere_center(self)-> Tuple[float, float, float]:
        """
        Get the center coordinates of the sphere defined by the circles
        
        Returns
        -------
        Tuple[float, float, float]
            (x, y, z) coordinates of the sphere center
        """
        return (self.circles[0].get_center()[1], self.circles[1].get_center()[1], self.circles[0].get_center()[0])


class SignalInterferenceFreehandSegmenter:
    """Interactive multi-class image segmentation interface."""
    
    def __init__(self, 
                 image: np.ndarray,
                 classes: List[str] = ['signal', 'interference'],
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 mask_alpha: float = 0.76,
                 initial_slice: int = 32,
                 cmap: str = 'gray'):
        """
        Initialize the multi-class segmenter.
        
        Parameters
        ----------
        image : np.ndarray
            3D image data to segment
        classes : List[str]
            List of class names for segmentation
        vmin : float, optional
            Minimum value for display window
        vmax : float, optional 
            Maximum value for display window
        mask_alpha : float
            Transparency of segmentation mask
        initial_slice : int
            Starting slice index
        cmap : str
            Colormap for image display
        """
        # Calculate display window if not provided
        if vmin is None:
            vmin = np.percentile(np.abs(image), 5)
        if vmax is None:
            vmax = np.percentile(np.abs(image), 95)
            
        # Create widgets
        self.class_selector = widgets.Dropdown(
            options=classes,
            description="class"
        )
        
        self.slice_selector = widgets.IntSlider(
            value=initial_slice,
            min=0,
            max=image.shape[0]-1,
            step=1,
            description='Slice',
            continuous_update=True
        )
        
        self.erasing_button = widgets.Checkbox(
            value=False,
            description="Erasing"
        )
        
        # Create segmenter
        self.segmenter = ImageSegmenter(
            image,
            classes=classes,
            mask_alpha=mask_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap
        )
        self.segmenter.current_class = 1
        self.segmenter.image_index = initial_slice
        
        # Set up callbacks
        self._setup_callbacks()
        controls = widgets.HBox([
            self.erasing_button,
            self.class_selector,
            self.slice_selector
        ])
        display(controls)
        display(self.segmenter)
        
    def _update(self, change):
        """Update segmenter state based on widget changes."""
        self.segmenter.current_class = self.class_selector.index + 1
        self.segmenter.erasing = self.erasing_button.value
        self.segmenter.image_index = self.slice_selector.value
        
    def _setup_callbacks(self):
        """Set up widget callbacks."""
        self.erasing_button.observe(self._update, names="value")
        self.class_selector.observe(self._update, names="value") 
        self.slice_selector.observe(self._update, names="value")
        
    @property
    def mask(self):
        """Get the current segmentation mask."""
        return self.segmenter.mask
    
    def get_masks(self):
        """Get the current segmentation masks for all classes."""
        interference_mask = self.segmenter.mask == 2
        signal_mask = self.segmenter.mask == 1
        slc_i = self.slice_selector.value
        interference_mask[:,:,:] = interference_mask[slc_i,:,:]
        signal_mask[:,:,:] = signal_mask[slc_i,:,:]
        return signal_mask, interference_mask

class PickPixel4Correlation:
    """Interactive viewer for 3D volumes with slice selection and coordinate printing."""
    
    def __init__(self, volume_show: np.ndarray, volume_use:np.ndarray, coil_names: list, ref_vals: np.array, cmap: str = 'gray',
                 vmin: float = None, vmax: float = None,
                 figsize: tuple = (12, 8)):
        """
        Initialize volume viewer.
        
        Parameters
        ----------
        volume : np.ndarray
            4D volume data to display (z, y, x, c)
        cmap : str
            Colormap for display
        vmin, vmax : float
            Display window min/max values
        figsize : tuple
            Figure size in inches
        """
        self.volume_s = volume_show
        self.volume = volume_use
        self.cmap = cmap
        self.coil_names = coil_names
        self.ref_vals = ref_vals
        
        # Calculate display window if not provided
        if vmin is None:
            vmin = np.percentile(self.volume_s, 5)
        if vmax is None:
            vmax = np.percentile(self.volume_s, 95)
        self.vmin = vmin
        self.vmax = vmax
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(1,2 ,figsize=figsize)
        self.fig.canvas.header_visible = False
        
        # Create slice slider
        self.slice_slider = widgets.IntSlider(
            value=volume_show.shape[0]//2,
            min=0,
            max=volume_show.shape[0]-1,
            description='Slice:',
            continuous_update=True
        )
        
        # Display initial slice
        self.img = self.ax[0].imshow(
            self.volume_s[self.slice_slider.value],
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax
        )
        
        # Set up callbacks
        self.slice_slider.observe(self._update_slice, names='value')
        self.fig.canvas.mpl_connect('button_press_event', self._onclick)
        
        # Set axis labels
        self.ax[0].set_xlabel('X')
        self.ax[0].set_ylabel('Y')
        self.ax[1].plot(ref_vals/ref_vals.max(), np.arange(len(self.coil_names)), 'x')

        self.plt, = self.ax[1].plot(np.zeros_like(np.arange(len(self.coil_names)), dtype=np.float64), np.arange(len(self.coil_names)), 'o')

        self.ax[1].set_yticks(np.arange(len(coil_names)), coil_names)

        
    def _update_slice(self, change):
        """Update displayed slice when slider changes."""
        self.img.set_data(self.volume_s[change['new']])
        self.ax[0].set_title(f'Slice {change["new"]}')
        self.fig.canvas.draw_idle()
        
    def _onclick(self, event):
        """Handle mouse click events."""
        if event.inaxes == self.ax[0]:
            x, y = int(event.xdata), int(event.ydata)
            z = self.slice_slider.value
            val = self.volume[z, y, x, :]
            self.plt.set_xdata(val/val.max())
            self.ax[1].set_xlim(0, (val/val.max()).max()*1.1)
            cc_ = np.corrcoef(self.ref_vals, val)[0, 1]
            self.ax[1].set_title(f'Correlation with reference: {cc_:.3f}.')
            # self.ax.set_title(f'Slice {z} - Clicked at ({y}, {x})')
            # print(f'Clicked coordinates (z,y,x): ({z},{y},{x}), value: {val:.3f}')
            
    def show(self):
        """Display the interactive viewer."""
        display(self.slice_slider)
        plt.show()

def ndv(data, ax: matplotlib.axes.Axes = None,  YX = [-2,-1], voxel_shape=None,  overlay_img=None, overlay_cmap='gray', colorbar=False, slider_values=None, clim=None, figsize=None, **kwargs):
    '''
    Opens a multi-dimensional array viewer widget in Jupyter

    Args:
        data (array): The n-dimensional data to be viewed
        ax: matplotlib axes to plot on (default: None, creates new figure)
        YX: two-element array indicating the data axes to be viewed on start (default: [-2,-1])
        voxel_shape: n-element array indicating the voxel shape (default: None / all ones)
        slider_values: n-element array of initial slider values (default: None / all zeros)
        clim: two-element array indicating the lower and upper limits of the color axis
        figsize: passed to matplotlib.pyplot.figure
        **kwargs: passed to matplotlib.pyplot.imshow

    Original code taken from https://github.com/danionella/ndview/tree/master
    '''
    
    dims = data.shape
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    if overlay_img is not None:
        if overlay_img.shape != data.shape:
            raise ValueError("Overlay image must have the same shape as the data.")
    
    plt.show()
    if not clim: 
        clim = [min(0, data.min()), data.max()]
    if not slider_values: 
        slider_values = [0 for i in range(len(dims))]
    im = []
    im_overlay = []
    sliders = []
    
    def rbCallback(event=None):
        if rbX.value == rbY.value:
            if event['owner'] == rbX: 
                rbY.value = event['old']
            elif event['owner'] == rbY: 
                rbX.value = event['old']
        ax.clear()
        refreshimage()
        sliderCallback()
        ax.set_xlabel(f'axis {rbX.value}')
        ax.set_ylabel(f'axis {rbY.value}')
        #plt.ylabel('axis')

    def sliderCallback(event=None):
        im.set_data(getslice())
        if overlay_img is not None:
            im_overlay.set_data(getslice_overlay())
        ax.figure.canvas.draw_idle()

    def getslice():
        subs = [sliders[i].value for i in range(len(sliders))]
        subs[rbY.value] = slice(None)
        subs[rbX.value] = slice(None)
        out = data[tuple(subs)]
        if rbX.value < rbY.value: 
            out = out.T
        return out

    def getslice_overlay():
        subs = [sliders[i].value for i in range(len(sliders))]
        subs[rbY.value] = slice(None)
        subs[rbX.value] = slice(None)
        out = overlay_img[tuple(subs)]
        if rbX.value < rbY.value: 
            out = out.T
        return out


    def refreshimage():
        nonlocal im
        nonlocal im_overlay
        aspect = voxel_shape[rbY.value]/voxel_shape[rbX.value] if voxel_shape is not None else 'auto'
        if overlay_img is not None:
            im_overlay = ax.imshow(getslice_overlay(), cmap=overlay_cmap, aspect=aspect, vmin=overlay_img.min(), vmax=overlay_img.max())
            im = ax.imshow(getslice(), alpha=0.5, aspect=aspect, vmin=clim_slider.value[0], vmax=clim_slider.value[1], **kwargs)
        else:
            im = ax.imshow(getslice(), aspect=aspect, vmin=clim_slider.value[0], vmax=clim_slider.value[1], **kwargs)

        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def clim_sliderCallback(event):
        im.set_clim(clim_slider.value)
    
    for i in range(len(dims)):
        slider = widgets.IntSlider(value=slider_values[i],min=0,max=dims[i]-1,description=f'[{dims[i]}]',layout=Layout(width='500px', height='17px'))
        slider.observe(sliderCallback, names='value')
        sliders.append(slider)
        
    rbY = widgets.RadioButtons(options=[i for i in range(len(dims))],layout={'width':'40px'},value=(len(dims)+YX[0])%len(dims))
    rbX = widgets.RadioButtons(options=[i for i in range(len(dims))],layout={'width':'40px'},value=(len(dims)+YX[1])%len(dims))
    rbY.observe(rbCallback, names='value')
    rbX.observe(rbCallback, names='value')
    
    clim_slider = widgets.FloatRangeSlider(value=clim,min=min(0, clim[0]),max=clim[1]*1.5,step=1/1000,description='clim:',readout_format='.3', layout=Layout(width='500px', height='30px'))
    clim_slider.observe(clim_sliderCallback, names='value')
    
    hb = widgets.HBox([widgets.VBox([widgets.Label('Y'), rbY]), widgets.VBox([widgets.Label('X'), rbX]), widgets.VBox([clim_slider, widgets.VBox(sliders)])])
    display(hb)
    
    rbCallback()
    plt.tight_layout()
    
