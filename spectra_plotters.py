import io
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2 as imageio
from spectra_class import spectrum, spectra
from gruvbox_theme import GRUVBOX
# this program was almost entirely vibecoded
# stitch together multiple single spectra plots
def plot_spectra(
    spectral_data,
    index_range: list[int] = [0, 1],
    filepath: str = None,
    show: bool = True,
    linewidth: float = 0.4):
    """
    Plots each spectrum in `spectral_data` within `index_range` into a composite subplot 
    grid using NumPy matrices and the Gruvbox theme.
    
    Args:
        spectral_data: `spectrum` object, list of `spectrum` objects, or `spectra` object.
        index_range: Optional list/tuple of 2 integers [start, end] specifying which spectra to plot by index (default: [0, 1], plotting only the first spectrum).
        filepath: Optional path to save the composite plot figure.
        show: Whether to display the plot (default: True).
        linewidth: Line width for spectral traces (default: 0.4).
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")

    if index_range is None:
        index_range = [0, 1]
    elif len(index_range) != 2:
        raise ValueError("index_range must contain exactly two integers: [start_index, end_index].")

    start_idx, end_idx = int(index_range[0]), int(index_range[1])
    total_spectra = spectral_data.intensities.shape[0]

    if start_idx < 0 or end_idx <= start_idx:
        raise ValueError(f"Invalid index_range: [{start_idx}, {end_idx}]. Start must be >= 0 and < end.")
    if start_idx >= total_spectra:
        raise IndexError(f"Start index {start_idx} is out of range for dataset with {total_spectra} spectra.")

    # Slice according to index_range
    wavenumbers = spectral_data.wavenumbers[start_idx:end_idx]
    intensities = spectral_data.intensities[start_idx:end_idx]
    num_spectra = intensities.shape[0]

    if num_spectra == 0:
        raise ValueError("No spectra found in the specified index_range.")

    # Grid layout calculation
    ncols = int(np.ceil(np.sqrt(num_spectra)))
    nrows = int(np.ceil(num_spectra / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), squeeze=False)
    axes_flat = axes.flatten()

    cycle = [
        GRUVBOX.get("yellow", "#fabd2f"),
        GRUVBOX.get("aqua", "#8ec07c"),
        GRUVBOX.get("blue", "#83a598"),
        GRUVBOX.get("orange", "#fe8019"),
        GRUVBOX.get("purple", "#d3869b"),
        GRUVBOX.get("green", "#b8bb26"),
        GRUVBOX.get("red", "#fb4934"),
    ]

    has_positions = spectral_data.positions is not None
    if has_positions:
        x_coords = spectral_data.x[start_idx:end_idx]
        y_coords = spectral_data.y[start_idx:end_idx]

    for i in range(num_spectra):
        actual_idx = start_idx + i
        ax = axes_flat[i]
        line_color = cycle[actual_idx % len(cycle)]
        
        ax.plot(wavenumbers[i], intensities[i], color=line_color, linewidth=linewidth)
        ax.set_xlabel("Wavenumber ($cm^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        
        if has_positions:
            x_pos = x_coords[i]
            y_pos = y_coords[i]
            ax.set_title(f"Spectrum #{actual_idx + 1} (x={x_pos:.2f}, y={y_pos:.2f})")
        else:
            ax.set_title(f"Spectrum #{actual_idx + 1}")

    # Hide any unused subplot axes in the grid
    for j in range(num_spectra, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

# 2D PCA plot with color for PC3
def plot_PC2(
    spectral_data,
    color: str = None,
    point_size: float = 15,
    alpha: float = 0.95,
    pc3_color: bool = False,
    filepath: str = None,
    show: bool = True):
    """
    Plots the first two principal components from Raman spectral data.
    Assumes all data is collected in the same wavenumber range.
    Args:
        spectral_data: `spectrum` object, list of `spectrum` objects, or `spectra` object.
        color: Color of data points (default: Gruvbox yellow). Accepts hex, named colors, or Gruvbox keys.
        point_size: Size of data points (default: 15).
        alpha: Opacity of data points (default: 0.99).
        pc3_color: Whether to color the points by the third principal component.
        filepath: Optional path to save the composite plot figure.
        show: Whether to display the plot (default: True).
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])

    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")
    
    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    X = spectral_data.intensities
    if X.shape[0] < 2:
        raise ValueError("PCA requires at least 2 spectra to calculate principal components.")

    # Center intensity data
    X_centered = X - np.mean(X, axis=0)

    # Compute PCA via SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = U * S

    total_var = np.sum(S ** 2)
    var_explained = (S ** 2) / total_var if total_var > 0 else np.zeros_like(S)

    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    var_pc1 = var_explained[0] * 100
    var_pc2 = var_explained[1] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    pt_color = GRUVBOX.get(color, color) if color else GRUVBOX.get("yellow", "#fabd2f")

    if pc3_color and scores.shape[1] >= 3:
        pc3 = scores[:, 2]
        var_pc3 = var_explained[2] * 100
        scatter = ax.scatter(
            pc1, pc2,
            c=pc3,
            cmap="gruvbox_rainbow",
            edgecolors="none",
            linewidths=0,
            s=point_size,
            alpha=alpha
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f"PC3 ({var_pc3:.1f}% Variance)")
    else:
        ax.scatter(
            pc1, pc2,
            color=pt_color,
            edgecolors="none",
            linewidths=0,
            s=point_size,
            alpha=alpha
        )

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% Variance)")
    ax.set_title("2D PCA Score Plot")

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax

# 3D PCA plot
def plot_PC3(
    spectral_data,
    color: str = None,
    point_size: float = 15,
    alpha: float = 0.99,
    pc4_color: bool = False,
    filepath: str = None,
    show: bool = True):
    """
    Plots the first three principal components from Raman spectral data.
    Assumes all data is collected in the same wavenumber range.
    Args:
        spectral_data: `spectrum` object, list of `spectrum` objects, or `spectra` object.
        color: Color of data points (default: Gruvbox yellow). Accepts hex, named colors, or Gruvbox keys.
        point_size: Size of data points (default: 15).
        alpha: Opacity of data points (default: 0.99).
        pc4_color: Whether to color the points by the fourth principal component.
        filepath: Optional path to save the composite plot figure.
        show: Whether to display the plot (default: True).
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])

    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")
    
    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    X = spectral_data.intensities
    if X.shape[0] < 3:
        raise ValueError("PCA requires at least 3 spectra to calculate principal components.")
    
    # Center intensity data
    X_centered = X - np.mean(X, axis=0)

    # Compute PCA via SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    scores = U * S

    total_var = np.sum(S ** 2)
    var_explained = (S ** 2) / total_var if total_var > 0 else np.zeros_like(S)

    pc1 = scores[:, 0]
    pc2 = scores[:, 1]
    pc3 = scores[:, 2]
    var_pc1 = var_explained[0] * 100
    var_pc2 = var_explained[1] * 100
    var_pc3 = var_explained[2] * 100

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Gruvbox 3D pane and axis styling
    ax.set_facecolor(GRUVBOX.get("bg0", "#282828"))
    fig.patch.set_facecolor(GRUVBOX.get("bg0", "#282828"))
    ax.xaxis.pane.set_facecolor(GRUVBOX.get("bg1", "#3c3836"))
    ax.yaxis.pane.set_facecolor(GRUVBOX.get("bg1", "#3c3836"))
    ax.zaxis.pane.set_facecolor(GRUVBOX.get("bg1", "#3c3836"))
    ax.xaxis.pane.set_edgecolor(GRUVBOX.get("bg4", "#7c6f64"))
    ax.yaxis.pane.set_edgecolor(GRUVBOX.get("bg4", "#7c6f64"))
    ax.zaxis.pane.set_edgecolor(GRUVBOX.get("bg4", "#7c6f64"))

    pt_color = GRUVBOX.get(color, color) if color else GRUVBOX.get("yellow", "#fabd2f")

    if pc4_color and scores.shape[1] >= 4:
        pc4 = scores[:, 3]
        var_pc4 = var_explained[3] * 100
        scatter = ax.scatter(
            pc1, pc2, pc3,
            c=pc4,
            cmap="gruvbox_rainbow",
            edgecolors="none",
            linewidths=0,
            s=point_size,
            alpha=alpha
        )
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
        cbar.set_label(f"PC4 ({var_pc4:.1f}% Variance)")
    else:
        ax.scatter(
            pc1, pc2, pc3,
            color=pt_color,
            edgecolors="none",
            linewidths=0,
            s=point_size,
            alpha=alpha
        )

    ax.set_xlabel(f"PC1 ({var_pc1:.1f}% Variance)")
    ax.set_ylabel(f"PC2 ({var_pc2:.1f}% Variance)")
    ax.set_zlabel(f"PC3 ({var_pc3:.1f}% Variance)")
    ax.set_title("3D PCA Score Plot")

    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax

# 3D PCA plot with angle shift animation (2-frame)
def plot_PC3_animated(
    spectral_data,
    color: str = None,
    point_size: float = 15,
    alpha: float = 0.99,
    pc4_color: bool = False,
    filepath: str = None,
    show: bool = True,
    angle_shift: float = 1.0,
    fps: int = 4):
    """
    Creates a 2-frame oscillating GIF of a 3D PCA plot with a slight viewing angle 
    shift (wiggle stereoscopy) to create an illusion of 3D depth.
    
    Args:
        spectral_data: `spectrum` object, list of `spectrum` objects, or `spectra` object.
        color: Color of data points (default: Gruvbox yellow).
        point_size: Size of data points (default: 15).
        alpha: Opacity of data points (default: 0.7).
        pc4_color: Whether to color the points by the fourth principal component.
        filepath: Optional path to save the .gif animation.
        show: Whether to display the animation (default: True).
        angle_shift: Degrees of azimuth shift between the two frames (default: 1.0).
        fps: Frames per second for the oscillating gif (default: 4).
    """
    # Generate the base 3D PCA plot figure
    fig, ax = plot_PC3(
        spectral_data,
        color=color,
        point_size=point_size,
        alpha=alpha,
        pc4_color=pc4_color,
        filepath=None,
        show=False
    )

    elev = ax.elev if ax.elev is not None else 30
    azim = ax.azim if ax.azim is not None else -60

    # Render frame 1 (left angle) and frame 2 (right angle)
    frames = []
    angles = [azim - angle_shift / 2, azim + angle_shift / 2]

    for a in angles:
        ax.view_init(elev=elev, azim=a)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    plt.close(fig)

    # Save to gif if requested
    if filepath:
        if not filepath.endswith(".gif"):
            filepath += ".gif"
        imageio.mimsave(filepath, frames, fps=fps, loop=0)

    # Display oscillating animation if requested
    if show:
        from matplotlib.animation import ArtistAnimation
        anim_fig, anim_ax = plt.subplots(figsize=(9, 7))
        anim_ax.axis("off")
        anim_fig.patch.set_facecolor(GRUVBOX.get("bg0", "#282828"))
        ims = [[anim_ax.imshow(f, animated=True)] for f in frames]
        ani = ArtistAnimation(anim_fig, ims, interval=1000 / fps, blit=True, repeat=True)
        plt.show()
        return ani

    return frames

# calculate intensity variance between spectra
def calculate_intensity_variances(spectral_data) -> np.ndarray:
    """
    Calculates the variance in intensity at each wavenumber for your data. Assumes
    that all spectra are measured in the same wavenumber range. Takes in
    either a `spectra` object, a `spectrum` object, or a list of `spectrum` objects.
    Returns a 2xN numpy array where:
      - row 0: wavenumbers
      - row 1: intensity variances
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")
    
    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    wavenumbers = np.nanmean(spectral_data.wavenumbers, axis=0)
    variances = np.nanvar(spectral_data.intensities, axis=0)
    
    return np.vstack((wavenumbers, variances))

# 2D plot of intensity variances (wavenumber vs variance)
def plot_intensity_variances(
    spectral_data,
    title: str = "Wavenumber vs Intensity Variance",
    color: str = None,
    filepath: str = None,
    show: bool = True):
    """
    Plots wavenumber vs intensity variance for spectral data.
    Takes in either `spectra` objects, a `spectrum` object, or a list of `spectrum` objects.
    Assumes that all spectra are measured in the same wavenumber range.
    
    Args:
        spectral_data: `spectra`, `spectrum`, or list of `spectrum` objects.
        title: Title of the plot.
        color: Optional line color.
        filepath: Optional path to save the plot.
        show: Whether to display the plot (default: True).
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
    
    if filepath:
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
        
    return fig, ax

# finds min and max intensities in dataset (for normalization purposes)
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
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    min_intensity = np.nanmin(spectral_data.intensities)
    max_intensity = np.nanmax(spectral_data.intensities)
    
    return float(min_intensity), float(max_intensity)

# plot intensity heatmap
def intensity_heatmap(
    spectral_data,
    wavenumber_index: int,
    grid_dimensions: list = None,
    vmin: float = None,
    vmax: float = None,
    cmap: str = "gruvbox_heat",
    title: str = None,
    filepath: str = None,
    show: bool = False,
    show_wavenumber_bar: bool = False,
    show_spectra_positions: bool = True):
    """
    Creates a spatial heatmap of intensity for a specific wavenumber index on a binned rectangular grid of square cells.
    The grid dimensions [nx, ny] enforce an nx:ny side proportion and cell density, fitted to the edge spectra.
    Each individual grid cell is guaranteed to be a square. Empty cells with no spectra are rendered as dark background.
    
    Args:
        spectral_data: `spectra` object, `spectrum` object, or list of `spectrum` objects.
        wavenumber_index: Integer index of the wavenumber to plot (e.g., 1 for the 2nd wavenumber).
        grid_dimensions: Optional list or tuple of [nx, ny] specifying the number of grid bins along X and Y (default: [20, 20]).
        vmin: Minimum intensity threshold for the colormap.
        vmax: Maximum intensity threshold for the colormap.
        cmap: Colormap to use (default: 'gruvbox_heat', or 'gruvbox_rainbow').
        title: Optional title for the plot.
        filepath: Optional path to save the plot.
        show: Whether to display the plot (default: False).
        show_wavenumber_bar: Whether to display a vertical wavenumber range indicator bar on the left (default: False).
        show_spectra_positions: Whether to plot true (x, y) spectra acquisition coordinates as subtle black dots (default: True).
        
    Returns:
        tuple: (fig, ax)
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")
    
    if spectral_data.positions is None:
        raise ValueError("No spatial position data (X/Y) available in the provided spectra object.")

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    x_coords = spectral_data.x
    y_coords = spectral_data.y

    # Check for missing or identical spatial coordinates (e.g., in Renishaw SPC files)
    if (np.all(x_coords == 0) and np.all(y_coords == 0)) or (np.all(x_coords == x_coords[0]) and np.all(y_coords == y_coords[0])):
        raise ValueError(
            "No spatial distribution data found in dataset (all spectra coordinates are identical or (0, 0)). "
            "Note: Renishaw WiRE exports to .spc format do not include stage position metadata. Use .wdf or .txt files for spatial heatmaps."
        )
    
    if grid_dimensions is None:
        grid_dimensions = [20, 20]
    elif len(grid_dimensions) != 2:
        raise ValueError("grid_dimensions must be a list or tuple of 2 numbers: [nx, ny].")
        
    nx, ny = int(grid_dimensions[0]), int(grid_dimensions[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_dimensions values must be positive integers.")
    
    intensities = spectral_data.intensities[:, wavenumber_index]
    target_wavenumber = np.nanmean(spectral_data.wavenumbers, axis=0)[wavenumber_index]
    
    span_x = float(np.nanmax(x_coords) - np.nanmin(x_coords))
    span_y = float(np.nanmax(y_coords) - np.nanmin(y_coords))
    
    # Determine square cell side length s such that all cells are square
    # and the grid aspect ratio is nx : ny while fitting to the edge spectra
    if span_x == 0 and span_y == 0:
        s = 1.0
    elif span_x == 0:
        s = span_y / ny
    elif span_y == 0:
        s = span_x / nx
    else:
        s = max(span_x / nx, span_y / ny)
        
    x_c = float(np.nanmin(x_coords) + np.nanmax(x_coords)) / 2.0
    y_c = float(np.nanmin(y_coords) + np.nanmax(y_coords)) / 2.0
    
    w_grid = nx * s
    h_grid = ny * s
    
    x_min_grid = x_c - w_grid / 2.0
    x_max_grid = x_c + w_grid / 2.0
    y_min_grid = y_c - h_grid / 2.0
    y_max_grid = y_c + h_grid / 2.0
    
    x_edges = np.linspace(x_min_grid, x_max_grid, nx + 1)
    y_edges = np.linspace(y_min_grid, y_max_grid, ny + 1)
    
    # Calculate 2D histogram with intensities as weights (sum of intensities per square)
    H, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges], weights=intensities)
    counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
    
    # Mask unpopulated cells with NaN so they render as completely dark background
    H[counts == 0] = np.nan
    
    if isinstance(cmap, str):
        cmap_obj = copy.copy(plt.get_cmap(cmap))
    else:
        cmap_obj = copy.copy(cmap)
    cmap_obj.set_bad(color=GRUVBOX["bg0"])
    
    # If vmin/vmax not provided, calculate strictly from occupied cells
    if vmin is None or vmax is None:
        valid_H = H[counts > 0]
        if valid_H.size > 0:
            if vmin is None:
                vmin = float(np.nanmin(valid_H))
            if vmax is None:
                vmax = float(np.nanmax(valid_H))
            if vmin == vmax:
                vmin -= 1.0
                vmax += 1.0
        else:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
            
    if show_wavenumber_bar:
        fig = plt.figure(figsize=(9.5, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[0.06, 1], wspace=0.25)
        ax_bar = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])

        wavenumber_min = float(np.nanmin(spectral_data.wavenumbers))
        wavenumber_max = float(np.nanmax(spectral_data.wavenumbers))

        ax_bar.set_facecolor(GRUVBOX.get("bg1", "#3c3836"))
        ax_bar.set_xlim(0, 1)
        ax_bar.set_ylim(wavenumber_min, wavenumber_max)
        ax_bar.set_ylabel("Wavenumber ($cm^{-1}$)")
        ax_bar.set_xticks([])

        ax_bar.axhspan(wavenumber_min, wavenumber_max, color=GRUVBOX.get("bg1", "#3c3836"), alpha=0.8)
        ax_bar.axhline(target_wavenumber, color=GRUVBOX.get("yellow", "#fabd2f"), linewidth=1.5)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # pcolormesh requires C with shape (len(y_edges)-1, len(x_edges)-1), so H.T is used
    mesh = ax.pcolormesh(x_edges, y_edges, H.T, shading='flat', vmin=vmin, vmax=vmax, cmap=cmap_obj)
    ax.set_aspect('equal')
    ax.set_facecolor(GRUVBOX["bg0"])

    # Plot true acquisition (x, y) coordinates as small, subtle black dots
    if show_spectra_positions:
        ax.scatter(x_coords, y_coords, color="black", s=2, alpha=0.7, edgecolors="none", zorder=4)
    
    cbar = fig.colorbar(mesh, ax=ax, pad=0.04)
    cbar.set_label("Summed Intensity")
    
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Intensity Heatmap @ {target_wavenumber:.2f} cm$^{{-1}}$")
        
    if filepath:
        plt.savefig(filepath, bbox_inches="tight", dpi=300)
        
    if show:
        plt.show()
        
    return fig, ax

# animate heatmaps (sweep across wavenumber range)
def animate_heatmaps(
    spectral_data,
    filepath: str,
    grid_dimensions: list = [20,20],
    cmap: str = "gruvbox_heat",
    title: str = None,
    step_size: int = 10,
    fps: int = 15,
    show_spectra_positions: bool = True):
    """
    Creates an animated GIF of intensity heatmaps sweeping across wavenumbers on a binned rectangular grid of square cells.
    The brightness is normalized across all frames for consistency, empty cells are rendered as dark background, and
    a vertical wavenumber progress bar is displayed on the left side.
    
    Args:
        spectral_data: `spectra` object, `spectrum` object, or list of `spectrum` objects.
        filepath: Mandatory file path to save the .gif output.
        grid_dimensions: Optional list or tuple of [nx, ny] specifying the number of grid bins along X and Y (default: [20, 20]).
        cmap: Colormap to use.
        title: Base title for the plots.
        step_size: How much the wavenumber index jumps between frames.
        fps: Frames per second for the output gif.
        show_spectra_positions: Whether to plot true (x, y) spectra acquisition coordinates as subtle black dots (default: True).
    """
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")
        
    if not filepath.endswith('.gif'):
        filepath += '.gif'
        
    if spectral_data.positions is None:
        raise ValueError("No spatial position data (X/Y) available in the provided spectra object.")

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
        
    x_coords = spectral_data.x
    y_coords = spectral_data.y

    # Check for missing or identical spatial coordinates (e.g., in Renishaw SPC files)
    if (np.all(x_coords == 0) and np.all(y_coords == 0)) or (np.all(x_coords == x_coords[0]) and np.all(y_coords == y_coords[0])):
        raise ValueError(
            "No spatial distribution data found in dataset (all spectra coordinates are identical or (0, 0)). "
            "Note: Renishaw WiRE exports to .spc format do not include stage position metadata. Use .wdf or .txt files for spatial heatmaps."
        )
        
    if len(grid_dimensions) != 2:
        raise ValueError("grid_dimensions must be a list or tuple of 2 numbers: [nx, ny].")
        
    nx, ny = int(grid_dimensions[0]), int(grid_dimensions[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_dimensions values must be positive integers.")
        
    span_x = float(np.nanmax(x_coords) - np.nanmin(x_coords))
    span_y = float(np.nanmax(y_coords) - np.nanmin(y_coords))
    
    if span_x == 0 and span_y == 0:
        s = 1.0
    elif span_x == 0:
        s = span_y / ny
    elif span_y == 0:
        s = span_x / nx
    else:
        s = max(span_x / nx, span_y / ny)
        
    x_c = float(np.nanmin(x_coords) + np.nanmax(x_coords)) / 2.0
    y_c = float(np.nanmin(y_coords) + np.nanmax(y_coords)) / 2.0
    
    w_grid = nx * s
    h_grid = ny * s
    
    x_min_grid = x_c - w_grid / 2.0
    x_max_grid = x_c + w_grid / 2.0
    y_min_grid = y_c - h_grid / 2.0
    y_max_grid = y_c + h_grid / 2.0
    
    x_edges = np.linspace(x_min_grid, x_max_grid, nx + 1)
    y_edges = np.linspace(y_min_grid, y_max_grid, ny + 1)
    
    counts, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges])
    
    num_wavenumbers = spectral_data.wavenumbers.shape[1] if spectral_data.wavenumbers.ndim > 1 else len(spectral_data.wavenumbers)
    
    # Compute global vmin and vmax across sampled frames strictly from occupied cells
    sampled_indices = list(range(0, num_wavenumbers, step_size))
    all_binned_vals = []
    for idx in sampled_indices:
        h, _, _ = np.histogram2d(x_coords, y_coords, bins=[x_edges, y_edges], weights=spectral_data.intensities[:, idx])
        valid_vals = h[counts > 0]
        if valid_vals.size > 0:
            all_binned_vals.extend(valid_vals.tolist())
            
    if all_binned_vals:
        vmin = float(np.nanmin(all_binned_vals))
        vmax = float(np.nanmax(all_binned_vals))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    else:
        vmin = 0.0
        vmax = 1.0
    
    frames = []
    total_frames = len(sampled_indices)
    print(f"Generating animation with {total_frames} frames...")
    
    for idx in sampled_indices:
        fig, ax = intensity_heatmap(
            spectral_data=spectral_data,
            wavenumber_index=idx,
            grid_dimensions=grid_dimensions,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=title,
            show=False,
            show_wavenumber_bar=True,
            show_spectra_positions=show_spectra_positions
        )
        if fig is None:
            continue
            
        # Save figure to in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100) 
        buf.seek(0)
        frames.append(Image.open(buf).convert('RGB'))
        plt.close(fig) 
        
    if frames:
        # Build a unified global 256-color palette across sampled frames
        # This completely prevents colorbar and palette flickering across frames in the GIF
        step = max(1, len(frames) // 10)
        sample_stack = Image.fromarray(np.vstack([np.array(im) for im in frames[::step]]))
        palette_img = sample_stack.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        quantized_frames = [im.quantize(palette=palette_img, dither=Image.Dither.NONE) for im in frames]
        
        duration_ms = int(1000 / fps)
        quantized_frames[0].save(
            filepath,
            save_all=True,
            append_images=quantized_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True
        )
    print(f"Animation successfully saved to {filepath}")

# hi :)