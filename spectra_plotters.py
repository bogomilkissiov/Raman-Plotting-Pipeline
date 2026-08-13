import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from spectra_class import spectrum, spectra
from gruvbox_theme import GRUVBOX

############################
# FANCY PLOTTING FUNCTIONS #
############################
def calculate_intensity_variances(spectral_data) -> np.ndarray:
    """
    Calculates the variance in intensity at each wavenumber for your data. Assumes
    that all spectra are measured in the same wavenumber range. Takes in
    either a `spectra` object or a list of `spectrum` objects. Spectra objects work
    faster. Assumes that your measurements are done in the same wavenumber range.
    Returns a 2xN numpy array where:
      - row 0: wavenumbers
      - row 1: intensity variances
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])

    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if isinstance(spectral_data[0],spectrum):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    
    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    wavenumbers = np.nanmean(spectral_data.wavenumbers, axis=0)
    variances = np.nanvar(spectral_data.intensities, axis=0)
    
    return np.vstack((wavenumbers, variances))

def plot_intensity_variances(
    spectral_data,
    title: str = "Wavenumber vs Intensity Variance",
    color: str = None,
    save_path: str = None,
    show: bool = True):
    """
    Plots wavenumber vs intensity variance across for spectral data.
    Takes in either `spectra` objects or a list of `spectrum` objects.
    Assumes that all spectra are measured in the same wavenumber range.
    """
    variance_matrix = calculate_intensity_variances(spectral_data)
    
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

def intensity_extrema(spectral_data) -> tuple[float, float]:
    """
    Finds the lowest and highest raw intensity values across a spectra object, 
    spectrum object, or list of spectrum objects.
    
    Args:
        spectral_data: `spectra` object, `spectrum` object, or list of `spectrum` objects.
        
    Returns:
        tuple[float, float]: (min_intensity, max_intensity)
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])

    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if isinstance(spectral_data[0],spectrum):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    min_intensity = np.nanmin(spectral_data.intensities)
    max_intensity = np.nanmax(spectral_data.intensities)
    
    return float(min_intensity), float(max_intensity)

def intensity_heatmap(
    spectral_data,
    wavenumber_index: int,
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
        spectral_data: `spectra` object, `spectrum` object, or list of `spectrum` objects.
        wavenumber_index: Integer index of the wavenumber to plot (e.g., 1 for the 2nd wavenumber).
        vmin: Minimum intensity threshold for the colormap.
        vmax: Maximum intensity threshold for the colormap.
        cmap: Colormap to use (default: 'gruvbox_heat', or 'gruvbox_rainbow').
        title: Optional title for the plot.
        save_path: Optional path to save the plot.
        show: Whether to display the plot.
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])

    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if isinstance(spectral_data[0],spectrum):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    
    if spectral_data.positions is None:
        raise ValueError("No spatial position data (X/Y) available in the provided spectra object.")

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    x_coords = spectral_data.x
    y_coords = spectral_data.y
    intensities = spectral_data.intensities[:, wavenumber_index]
    target_wavenumber = np.nanmean(spectral_data.wavenumbers, axis=0)[wavenumber_index]
    
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

def animate_heatmaps(
    spectral_data,
    save_path: str,
    cmap: str = "gruvbox_heat",
    title: str = None,
    step_size: int = 10,
    fps: int = 15):
    """
    Creates an animated GIF of intensity heatmaps sweeping across wavenumbers.
    The brightness is normalized to the min and max intensities for the whole spectra group.
    
    Args:
        spectral_data: `spectra` object, `spectrum` object, or list of `spectrum` objects.
        save_path: Mandatory file path to save the .gif output.
        cmap: Colormap to use.
        title: Base title for the plots.
        step_size: How much the wavenumber index jumps between frames.
        fps: Frames per second for the output gif.
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            print("Warning: Empty spectra list provided.")
            return
        if isinstance(spectral_data[0], spectrum):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
        
    if not save_path.endswith('.gif'):
        save_path += '.gif'
        
    vmin, vmax = intensity_extrema(spectral_data)
    num_wavenumbers = spectral_data.wavenumbers.shape[1] if spectral_data.wavenumbers.ndim > 1 else len(spectral_data.wavenumbers)
    
    frames = []
    total_frames = len(range(0, num_wavenumbers, step_size))
    print(f"Generating animation with {total_frames} frames...")
    
    for idx in range(0, num_wavenumbers, step_size):
        fig, ax = intensity_heatmap(
            spectral_data=spectral_data,
            wavenumber_index=idx,
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