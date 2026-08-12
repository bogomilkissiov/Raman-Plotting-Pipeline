import os
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from cycler import cycler
from renishawWiRE import WDFReader
from spectra_class import spectra
import processing_helpers as ph
from gruvbox_theme import GRUVBOX, apply_gruvbox_theme

########################
# PROCESSING FUNCTIONS #
########################
def wdf_to_spectra(wdf_filepath : str) -> list[spectra]:
    """
    Parses a Renishaw WDF map file and returns a list of spectra objects.
    Each spectra object contains:
      - position_vector: [x, y] stage coordinates
      - wavenumber_vector: 1D array of wavenumbers
      - intensity_vector: 1D array of intensities
    """
    reader = WDFReader(wdf_filepath)
    wavenumbers = reader.xdata
    spectra_data = reader.spectra

    if spectra_data.ndim == 1:
        spectra_data = spectra_data.reshape(1, -1)
    elif spectra_data.ndim > 2:
        spectra_data = spectra_data.reshape(-1, spectra_data.shape[-1])
        
    x_coords = np.ravel(reader.xpos) if reader.xpos is not None else np.zeros(len(spectra_data))
    y_coords = np.ravel(reader.ypos) if reader.ypos is not None else np.zeros(len(spectra_data))
        
    spectra_list = []
    for i in range(len(spectra_data)):
        pos = [x_coords[i], y_coords[i]]
        spectra_list.append(spectra(pos, wavenumbers, spectra_data[i]))
        
    return spectra_list

######################
# PLOTTING FUNCTIONS #
######################
def calculate_intensity_variances(spectra_list : list[spectra]) -> np.ndarray:
    """
    Calculates the variance in intensity at each wavenumber across a list of spectra objects.
    Returns a 2xN numpy array where:
      - row 0: wavenumbers
      - row 1: intensity variances
    """
    if not spectra_list:
        return np.empty((2, 0))
    
    wavenumbers = spectra_list[0].wavenumbers
    all_intensities = np.array([spec.intensities for spec in spectra_list])
    variances = np.nanvar(all_intensities, axis=0)
    
    return np.vstack((wavenumbers, variances))

def plot_intensity_variances(spectra_list : list[spectra],
    title: str = "Wavenumber vs Intensity Variance",
    color: str = None,
    save_path: str = None,
    show: bool = True):
    """
    Plots wavenumber vs intensity variance across a list of spectra objects using the Gruvbox theme.
    """
    variance_matrix = calculate_intensity_variances(spectra_list)
    if variance_matrix.size == 0:
        print("Warning: Empty spectra list provided.")
        return None, None
        
    wavenumbers = variance_matrix[0]
    variances = variance_matrix[1]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    line_color = color if color else GRUVBOX["yellow"]
    ax.plot(wavenumbers, variances, color=line_color, linewidth=1.5, label="Variance")
    
    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Intensity Variance")
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
        
    return fig, ax

def intensity_heatmap(
    spectra_list: list[spectra],
    wavenumber_idx: int,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "gruvbox_heat",
    title: str = None,
    save_path: str = None,
    show: bool = False):
    """
    Creates a spatial heatmap of intensity for a specific wavenumber index.
    The squares of the different spectra are related spatially by the coordinate vectors.
    
    Args:
        spectra_list: List of spectra objects.
        wavenumber_idx: Integer index of the wavenumber to plot (e.g., 1 for the 2nd wavenumber).
        vmin: Minimum intensity threshold for the colormap.
        vmax: Maximum intensity threshold for the colormap.
        cmap: Colormap to use (default: 'gruvbox_heat', or 'gruvbox_rainbow').
        title: Optional title for the plot.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
    """
    if not spectra_list:
        print("Warning: Empty spectra list provided.")
        return None, None
        
    x_coords = np.array([spec.x for spec in spectra_list])
    y_coords = np.array([spec.y for spec in spectra_list])
    intensities = np.array([spec.intensities[wavenumber_idx] for spec in spectra_list])
    target_wavenumber = spectra_list[0].wavenumbers[wavenumber_idx]
    
    x_rounded = np.round(x_coords, decimals=3)
    y_rounded = np.round(y_coords, decimals=3)
    
    ux = np.sort(np.unique(x_rounded))
    uy = np.sort(np.unique(y_rounded))
    
    Z = np.full((len(uy), len(ux)), np.nan)
    for x, y, intensity in zip(x_coords, y_coords, intensities):
        ix = np.searchsorted(ux, np.round(x, decimals=3))
        iy = np.searchsorted(uy, np.round(y, decimals=3))
        Z[iy, ix] = intensity
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # pcolormesh with 'nearest' shading makes a square for each coordinate point
    mesh = ax.pcolormesh(ux, uy, Z, shading='nearest', vmin=vmin, vmax=vmax, cmap=cmap)
    
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Raw Intensity")
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Intensity Heatmap @ {target_wavenumber:.2f} cm$^{{-1}}$")
        
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
        
    return fig, ax

def intensity_extrema(spectra_list: list[spectra]) -> tuple[float, float]:
    """
    Finds the lowest and highest raw intensity values across a list of spectra objects.
    
    Args:
        spectra_list: List of spectra objects.
        
    Returns:
        tuple[float, float]: (min_intensity, max_intensity)
    """
    if not spectra_list:
        return (0.0, 0.0)
        
    min_intensity = min(np.min(spec.intensities) for spec in spectra_list)
    max_intensity = max(np.max(spec.intensities) for spec in spectra_list)
    
    return float(min_intensity), float(max_intensity)

def animate_heatmaps(
    spectra_list: list[spectra],
    save_path: str,
    cmap: str = "gruvbox_heat",
    title: str = None,
    step_size: int = 10,
    fps: int = 15):
    """
    Creates an animated GIF of intensity heatmaps sweeping across wavenumbers.
    The brightness is normalized to the min and max intensities for the whole spectra group.
    
    Args:
        spectra_list: List of spectra objects.
        save_path: Mandatory file path to save the .gif output.
        cmap: Colormap to use.
        title: Base title for the plots.
        step_size: How much the wavenumber index jumps between frames.
        fps: Frames per second for the output gif.
    """
    if not spectra_list:
        print("Warning: Empty spectra list provided.")
        return
        
    if not save_path.endswith('.gif'):
        save_path += '.gif'
        
    vmin, vmax = intensity_extrema(spectra_list)
    num_wavenumbers = len(spectra_list[0].wavenumbers)
    
    frames = []
    total_frames = len(range(0, num_wavenumbers, step_size))
    print(f"Generating animation with {total_frames} frames...")
    
    for idx in range(0, num_wavenumbers, step_size):
        fig, ax = intensity_heatmap(
            spectra_list=spectra_list,
            wavenumber_idx=idx,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            show=False
        )
        if fig is None:
            continue
            
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches="tight", dpi=100) 
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig) 
        
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Animation successfully saved to {save_path}")