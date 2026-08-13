import os
import io
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from cycler import cycler
from renishawWiRE import WDFReader
from spectra_class import spectrum
import processing_helpers as ph
from gruvbox_theme import GRUVBOX, apply_gruvbox_theme
from spectra_plot_helpers import wdf_to_spectra, calculate_intensity_variances, plot_intensity_variances, intensity_heatmap, intensity_extrema, animate_heatmaps

fivesec_0 = "5s_100lp_map.wdf"
fivesec_1 = "5s_100lp_map-1.wdf"
twosec = "2s_100lp_map-2.wdf"
onesec = "1s_100lp_map-2.wdf"

# Read from disk ONCE
fivesec_0_raw = wdf_to_spectra(fivesec_0)
fivesec_1_raw = wdf_to_spectra(fivesec_1)
twosec_raw    = wdf_to_spectra(twosec)
onesec_raw    = wdf_to_spectra(onesec)

# Deep-copy in memory before processing
fivesec_0_processed = ph.preprocess_pipeline(copy.deepcopy(fivesec_0_raw))
fivesec_1_processed = ph.preprocess_pipeline(copy.deepcopy(fivesec_1_raw))
twosec_processed    = ph.preprocess_pipeline(copy.deepcopy(twosec_raw))
onesec_processed    = ph.preprocess_pipeline(copy.deepcopy(onesec_raw))

# plotting intensity variances for each map
plot_intensity_variances(fivesec_0_processed, title="5sec 100lp (0) Intensity Variances", save_path="5sec_100lp_(0)_intensity_variances.png", show=False)
plot_intensity_variances(fivesec_1_processed, title="5sec 100lp (1) Intensity Variances", save_path="5sec_100lp_(1)_intensity_variances.png", show=False)
plot_intensity_variances(twosec_processed, title="2sec 100lp Intensity Variances", save_path="2sec_100lp_intensity_variances.png", show=False)
plot_intensity_variances(onesec_processed, title="1sec 100lp Intensity Variances", save_path="1sec_100lp_intensity_variances.png", show=False)

# plotting a few intensity heat maps
