import numpy as np
from renishawWiRE import WDFReader
from spectra_class import spectra

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

def spc_to_spectra(spc_filepath: str) -> list[spectra]:
    """
    Parses a SPC map file and returns a list of spectra objects.
    Each spectra object contains:
      - position_vector: [x, y] stage coordinates
      - wavenumber_vector: 1D array of wavenumbers
      - intensity_vector: 1D array of intensities
    """
    try:
        from spcfile import SPCFile
    except ImportError:
        raise ImportError("The 'spcfile' package is required to parse .spc files. Install it using 'pip install spcfile'.")

    spc = SPCFile(spc_filepath)
    spectra_list = []
    
    # Standard SPC files often lose X,Y map coordinates when exported by Renishaw
    for sub in spc:
        # Skip empty padding spectra that are often added during export
        if np.all(sub.y == 0):
            continue
            
        # We try to extract Z and W if they exist, otherwise default to 0.0
        x_pos = sub.subheader.get('z_value', 0.0)
        y_pos = sub.subheader.get('w_value', 0.0)
        
        pos = [x_pos, y_pos]
        spectra_list.append(spectra(pos, sub.x, sub.y))
        
    return spectra_list

def txt_to_spectra(txt_filepath: str) -> list[spectra]:
    """
    Parses a TXT map file and returns a list of spectra objects.
    The text file is expected to have columns: #X, #Y, #Wave, #Intensity
    Each spectra object contains:
      - position_vector: [x, y] stage coordinates
      - wavenumber_vector: 1D array of wavenumbers
      - intensity_vector: 1D array of intensities
    """
    data = np.loadtxt(txt_filepath, skiprows=1)
    
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    wavenumbers_all = data[:, 2]
    intensities_all = data[:, 3]
    
    diffs = (np.diff(x_coords) != 0) | (np.diff(y_coords) != 0)
    split_indices = np.where(diffs)[0] + 1
    
    x_splits = np.split(x_coords, split_indices)
    y_splits = np.split(y_coords, split_indices)
    wavenumber_splits = np.split(wavenumbers_all, split_indices)
    intensity_splits = np.split(intensities_all, split_indices)
    
    spectra_list = []
    for x_arr, y_arr, wave_arr, int_arr in zip(x_splits, y_splits, wavenumber_splits, intensity_splits):
        pos = [x_arr[0], y_arr[0]]
        spectra_list.append(spectra(pos, wave_arr, int_arr))
        
    return spectra_list
