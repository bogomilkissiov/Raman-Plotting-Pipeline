import os
import io
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from cycler import cycler
from renishawWiRE import WDFReader
from gruvbox_theme import GRUVBOX, apply_gruvbox_theme
from spectra_class import spectrum, spectra
import preprocessing as pp
import spectra_plotters as sp
import file_loaders as fl

# Data filepaths
wdf_path = "test_data/1s_100lp_map-2 (1).wdf"
spc_path = "test_data/1s_100lp_map-2 (1).spc"
txt_path = "test_data/1s_100lp_map-2 (1).txt"

# load data as list of spectrum objects
wdf_spectrum = fl.wdf_to_spectra(wdf_path, pack=False)
spc_spectrum = fl.spc_to_spectra(spc_path, pack=False)
txt_spectrum = fl.txt_to_spectra(txt_path, pack=False)

# load data as spectra object
wdf_spectra = fl.wdf_to_spectra(wdf_path, pack=True)
spc_spectra = fl.spc_to_spectra(spc_path, pack=True)
txt_spectra = fl.txt_to_spectra(txt_path, pack=True)

# Preprocess all datasets through the preprocessing pipeline
print("Running preprocessing pipeline on all datasets...")
wdf_spectrum = pp.preprocess_pipeline(wdf_spectrum)
spc_spectrum = pp.preprocess_pipeline(spc_spectrum)
txt_spectrum = pp.preprocess_pipeline(txt_spectrum)
wdf_spectra = pp.preprocess_pipeline(wdf_spectra)
spc_spectra = pp.preprocess_pipeline(spc_spectra)
txt_spectra = pp.preprocess_pipeline(txt_spectra)

# Datasets dictionary for comprehensive testing
datasets = {
    "wdf_spectrum_list": wdf_spectrum,
    "spc_spectrum_list": spc_spectrum,
    "txt_spectrum_list": txt_spectrum,
    "wdf_spectra": wdf_spectra,
    "spc_spectra": spc_spectra,
    "txt_spectra": txt_spectra,}

# ----------------------------------------------------
# 1. PCA & COMPOSITE PLOTS
# ----------------------------------------------------
pca_output_dir = os.path.join("test_plots", "PCA")
os.makedirs(pca_output_dir, exist_ok=True)

for name, data in datasets.items():
    print(f"Generating PCA plots for: {name}")
    
    # 1. plot_spectra (first spectrum only)
    sp.plot_spectra(
        data,
        index_range=[0, 1],
        filepath=os.path.join(pca_output_dir, f"{name}_plot_spectra.png"),
        show=False
    )
    
    # 2. plot_PC2 (with PC3 color)
    sp.plot_PC2(
        data,
        pc3_color=False,
        filepath=os.path.join(pca_output_dir, f"{name}_plot_PC2.png"),
        show=False
    )
    
    # 3. plot_PC3 (with PC4 color)
    sp.plot_PC3(
        data,
        pc4_color=False,
        filepath=os.path.join(pca_output_dir, f"{name}_plot_PC3.png"),
        show=False
    )
    
    # 4. plot_PC3_animated (with PC4 color, saved as GIF)
    sp.plot_PC3_animated(
        data,
        pc4_color=False,
        angle_shift=0.5,
        fps=10,
        filepath=os.path.join(pca_output_dir, f"{name}_plot_PC3_animated.gif"),
        show=False
    )

print("All PCA test plots generated successfully in test_plots/PCA/!\n")

# ----------------------------------------------------
# 2. INTENSITY VARIANCES & HEATMAPS
# ----------------------------------------------------
heatmap_output_dir = os.path.join("test_plots", "Heatmaps")
os.makedirs(heatmap_output_dir, exist_ok=True)

# Pick a wavenumber index for heatmap testing
random_wavenumber_index = 100

for name, data in datasets.items():
    print(f"Generating variance & heatmap plots for: {name}")
    
    # 1. plot_intensity_variances (works on all datasets)
    sp.plot_intensity_variances(
        data,
        title=f"{name} Intensity Variances",
        filepath=os.path.join(heatmap_output_dir, f"{name}_plot_intensity_variances.png"),
        show=False
    )
    
    # 2. intensity_heatmap (single wavenumber index)
    try:
        sp.intensity_heatmap(
            data,
            wavenumber_index=random_wavenumber_index,
            grid_dimensions=[20, 20],
            title=f"{name} Heatmap (Index {random_wavenumber_index})",
            filepath=os.path.join(heatmap_output_dir, f"{name}_intensity_heatmap.png"),
            show=False
        )
        print(f"✓ Heatmap generated for {name}")
    except ValueError as e:
        print(f"✓ Expected spatial validation caught for {name}: {e}")
    
    # 3. animate_heatmaps (wavenumber sweep GIF)
    try:
        sp.animate_heatmaps(
            data,
            filepath=os.path.join(heatmap_output_dir, f"{name}_animate_heatmaps.gif"),
            grid_dimensions=[20, 20],
            step_size=10,
            fps=15
        )
        print(f"✓ Animation generated for {name}")
    except ValueError as e:
        print(f"✓ Expected spatial validation caught for {name}: {e}")

print("All heatmap and variance test plots generated successfully in test_plots/Heatmaps/!")


