import copy
import numpy as np
import scipy.signal
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
from scipy.special import wofz
from scipy.optimize import curve_fit
from skimage.restoration import denoise_wavelet
import peakutils
import BaselineRemoval
from spectra_class import spectra

# ==========================================
# 1. SPIKE REMOVAL (Cosmic Rays & Dead Pixels)
# ==========================================
def med_z_score(spec: spectra):
    """ Function to calculate modified Z-scores """
    intensity = spec.intensities
    intensity_median = np.median(intensity)
    intensity_mad = np.median(np.abs(intensity - intensity_median))
    z_scores = 0.6745*(intensity - intensity_median) / (intensity_mad + 1e-12) 
    return z_scores

def whit_z_score(spec: spectra, absolute=True):
    """
    Extract the differenced series (ie. y[n+1] - y[n])
    and calculate the modified z-score from it
    """
    intensity = spec.intensities
    delta_intensity = np.diff(intensity) # n-1 sizing
    delta_intensity = np.insert(delta_intensity, 0, [0])
    
    temp_spec = spectra(spec.position_vector, spec.wavenumbers, delta_intensity)
    if not absolute:
        return med_z_score(temp_spec)
    return np.abs(med_z_score(temp_spec))

def find_adjacent_neighbors(series, item):
    """
    See which mislabeled items in series share a 
    common neighbor index with item or are truly neighbors
    """
    if not item:
        return []
    # see which neighbors have that same point
    common_neighbors = [k for k,v in series.items() if list(item.values())[0] in v]
    
    series_condensed = copy.deepcopy(series)
    for k in common_neighbors:
        del series_condensed[k]
    
    if len(common_neighbors) == 1:
        return common_neighbors
    
    try:
        lb_neighbor = min(common_neighbors)
        lb_neighbor = {lb_neighbor: min(series[lb_neighbor])}
    except Exception:
        lb_neighbor = []
        
    try:
        ub_neighbor = max(common_neighbors)
        ub_neighbor = {ub_neighbor: max(series[ub_neighbor])}
    except Exception:
        ub_neighbor = []
    
    return common_neighbors + find_adjacent_neighbors(series_condensed, lb_neighbor) + find_adjacent_neighbors(series_condensed, ub_neighbor)

def spike_removal(spec: spectra, threshold=10, m=4):
    """ 
    Spike removal at location signifying spikes
    m is ~= window size / 2
    """
    intensity = spec.intensities
    spikes_raw = whit_z_score(spec, absolute=False)
    spikes_dead = copy.deepcopy(spikes_raw) < -30
    spikes = np.abs(spikes_raw) > threshold
    intensity_spikeless = copy.deepcopy(intensity)
    
    spikes_in_question = {}
    mislabeled_spike_centers = []
    
    for i in np.argwhere(spikes == True).flatten():
        # find all the neighboring valid points (non-spikes)
        sub_intensity_i = np.argwhere(spikes[i-m:i+(m+1)] == 0).flatten() + (i-m)
        spikes_in_question[i] = sub_intensity_i
        
        # dead pixel spikes
        if spikes_dead[i] == True:
            pass
        # if theres only 1 valid point, probably misclassified
        elif len(sub_intensity_i) == 1:
            mislabeled_spike_centers.append({i: sub_intensity_i[0]})
    
    # remove all mislabeled points
    for spike in mislabeled_spike_centers:        
        spikey_wikeys = np.unique(find_adjacent_neighbors(spikes_in_question, spike))
        for k in spikey_wikeys:
            if k in spikes_in_question:
                del spikes_in_question[k]
    
    # average labeled points
    for k, v in spikes_in_question.items():
        if len(v) == 0:
            continue
        intensity_spikeless[k] = copy.deepcopy(np.mean(intensity[v]))
        
    return intensity_spikeless, spikes_raw

def despike_spectra(spec: spectra):
    """
    For any given single spectra, run the outlier removal step.
    Modifies spectra object in place.
    """
    f_int, f_spikes = spike_removal(spec)
    spec.intensities = f_int
    return spec

# ==========================================
# 2. STITCHING CORRECTION
# ==========================================
def restitch_spectra(spec: spectra):
    """ Fixes Horiba bad stitching artifacts by interpolating over sudden dips """
    wavenumbers = spec.wavenumbers
    intensity = spec.intensities
    # v > 1000 ensures we only look for stitch artifacts at higher wavenumbers/indices
    bad_stitch = [v for v in np.where(np.gradient(intensity) < -10)[0] if wavenumbers[v] > 1000 if v < len(wavenumbers)]
    
    # Check bounds to ensure we don't hit an IndexError when referencing secondhalf or an empty firsthalf
    bad_stitch = [b for b in bad_stitch if b > 0 and b < len(intensity) - 1]
    
    # empty, good stitching from horiba
    if len(bad_stitch) == 0: 
        return spec

    firsthalf = intensity[ :bad_stitch[0] ]
    secondhalf = intensity[ bad_stitch[-1]: ] + (firsthalf[-1] - intensity[ bad_stitch[-1] ])
    middledata = np.linspace(intensity[bad_stitch[0]], secondhalf[0], bad_stitch[-1] - bad_stitch[0] + 1)[:-1]
    f_int = np.concatenate((firsthalf, middledata, secondhalf))
    spec.intensities = f_int
    return spec

# ==========================================
# 3. WAVELET DENOISING
# ==========================================
def log10_transform(spec: spectra):
    intensity = spec.intensities
    min_val = np.min(intensity)
    shift = abs(min_val) + 1.0 if min_val <= 0 else 1.0
    return np.log10(intensity + shift), shift

def denoise_spectra(spec: spectra):
    """ Wavelet transform for noise reduction using BayesShrink """
    if np.max(spec.intensities) == np.min(spec.intensities):
        return spec
        
    log_intensity, shift = log10_transform(spec)
    # rescale_sigma expects a boolean flag
    denoised_intensity = denoise_wavelet(log_intensity, method='BayesShrink', mode='soft', wavelet_levels=1, wavelet='coif3', rescale_sigma=True)
    f_int = (10**denoised_intensity) - shift
    spec.intensities = f_int
    return spec

# ==========================================
# 4. BASELINE SUBTRACTION
# ==========================================
def arpls(spec: spectra, lam=1e4, ratio=0.05, itermax=100):
    """ Asymmetrically reweighted penalized least squares smoothing (Baek et al. 2015) """
    y = spec.intensities
    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / (s + 1e-12)))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z

def remove_baseline(spec: spectra, order=3, method='combo', l_=100):
    """
    Determines baseline via desired methodologies:
        - 'poly'    : adaptive polynomial fitting (PeakUtils)
        - 'modpoly' : modified polynomial fitting (Zhao et al. 2007)
        - 'airpls'  : airPLS (Zhang et al. 2010)
        - 'arpls'   : arPLS (Baek et al. 2015)
        - 'combo'   : peakutils + ZhangFit
    """
    intensity = spec.intensities
    wavenumbers = spec.wavenumbers
    
    if np.max(intensity) == np.min(intensity):
        return spec
        
    if method == 'airpls':
        baseobj = BaselineRemoval.BaselineRemoval(intensity)
        spectra_corrected = baseobj.ZhangFit(repitition=50)
    elif method == 'modpoly':
        baseobj = BaselineRemoval.BaselineRemoval(intensity)
        spectra_corrected = baseobj.IModPoly(order)
    elif method == 'arpls':
        baseline = arpls(spec)
        spectra_corrected = intensity - baseline
    elif method == 'combo':
        baseline = peakutils.baseline(intensity, order)
        spectra_corrected = intensity - baseline
        baseobj = BaselineRemoval.BaselineRemoval(spectra_corrected)
        spectra_corrected2 = baseobj.ZhangFit(repitition=50, lambda_=l_)
        spectra_corrected = spectra_corrected2
    else:
        baseline = peakutils.baseline(intensity, order)
        spectra_corrected = intensity - baseline

    if np.isnan(spectra_corrected).any():
        spectra_corrected = intensity - np.min(intensity)

    spec.intensities = spectra_corrected
    return spec

# ==========================================
# 5. SILICON BACKGROUND REMOVAL
# ==========================================
def voigt_profile(x, x0, sigma, gamma, height=1, offset=0):
    """ Voigt function: convolution of Cauchy-Lorentz and Gaussian distribution. """
    w = wofz((x - x0 + 1j * gamma) / (sigma * np.sqrt(2)))
    return height * np.real(w) / (sigma * np.sqrt(2*np.pi)) + offset

def silicon_background_2order(x, x01, x02, x03, sigma1, sigma2, sigma3, gamma1, gamma2, gamma3, height1, height2, height3):
    return (voigt_profile(x, x01, sigma1, gamma1, height1, offset=0) +
            voigt_profile(x, x02, sigma2, gamma2, height2, offset=0) +
            voigt_profile(x, x03, sigma3, gamma3, height3, offset=0))

def background_removal_silicon(spec: spectra):
    """ Single spectra background removal of silicon optical phonons. """
    wavenumbers = spec.wavenumbers
    intensity = spec.intensities
    mask = (wavenumbers >= 800) & (wavenumbers <= 1150)
    wn = wavenumbers[mask]
    data = intensity[mask]
    
    if len(wn) == 0:
        return spec
    
    params_tot = 12
    lower_bounds = [0 for i in range(params_tot)]
    upper_bounds = [np.inf for i in lower_bounds]
    rl = [920, 950, 970]
    ru = [940, 960, 985]
    for i in range(3):     # resonance location
        lower_bounds[i] = rl[i]
        upper_bounds[i] = ru[i]
    for i in range(3):    # HWHM of gaussian (width), HWHM of lorentz (width), height
        lower_bounds[i+3] = 1
        lower_bounds[i+6] = 1
        upper_bounds[i+3] = 50
        upper_bounds[i+6] = 50
        
    popt, cov = curve_fit(silicon_background_2order, wn, data, bounds=(lower_bounds, upper_bounds), maxfev=100000)
    data_background = silicon_background_2order(wavenumbers, *popt)
    
    spec.intensities = intensity - data_background
    return spec

# ==========================================
# 6. NORMALIZATION
# ==========================================
def normalize_spectra(spec: spectra):
    """ Normalize spectra by computing the z-score based off of the mean. """
    intensity = spec.intensities
    intensity_mean = np.mean(intensity)
    intensity_std = np.std(intensity)
    z_scores = (intensity - intensity_mean) / (intensity_std + 1e-12)
    spec.intensities = z_scores
    return spec

# ==========================================
# 7. PREPROCESSING PIPELINE (NEED #2)
# ==========================================
def preprocess_pipeline(data):
    """
    Takes in a single spectra object or a list of spectra objects.
    Runs them through the preprocessing pipeline modifying the objects in place.
    Pipeline steps: despike -> restitch -> denoise -> baseline removal -> normalization
    """
    is_single = False
    if isinstance(data, spectra):
        data = [data]
        is_single = True
        
    for spec in data:
        despike_spectra(spec)
        restitch_spectra(spec)
        denoise_spectra(spec)
        remove_baseline(spec)
        normalize_spectra(spec)
        
    if is_single:
        return data[0]
    return data
