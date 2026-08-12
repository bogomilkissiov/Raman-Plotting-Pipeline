# Raman Plotting Pipeline

A Python toolkit for loading, preprocessing, and visualizing Raman spectra mapping data from Renishaw WDF files.

## Project Structure

- `spectra_class.py`: Defines the core `spectra` data class (`position_vector`, `wavenumber_vector`, `intensity_vector`).
- `processing_helpers.py`: Functions for despiking, restitching, denoising, baseline removal, and normalization.
- `spectra_plot_helpers.py`: Utilities for converting `.wdf` files to `spectra` objects, computing intensity variances, and generating heatmaps/animations.
- `map_analysis.py`: Main processing script.
- `gruvbox_theme.py`: Custom color themes for Matplotlib/Seaborn plots.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib seaborn renishawWiRE scipy imageio peakutils BaselineRemoval scikit-image
```
