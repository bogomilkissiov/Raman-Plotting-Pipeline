import os
import numpy as np

############################
# SPECTRA CLASS DEFINITION #
############################
class spectra:
    def __init__(self, position_vector, wavenumber_vector, intensity_vector):
        """
        position_vector: (x, y) stage position coordinates for the spectrum within a map.
        wavenumber_vector: 1D numpy array containing wavenumbers (cm^-1)
        intensity_vector: 1D numpy array containing intensities
        """
        self.position_vector = np.array(position_vector)
        self.wavenumber_vector = np.array(wavenumber_vector)
        self.intensity_vector = np.array(intensity_vector)

    @property
    def wavenumbers(self):
        return self.wavenumber_vector

    @wavenumbers.setter
    def wavenumbers(self, value):
        self.wavenumber_vector = np.array(value)

    @property
    def intensities(self):
        return self.intensity_vector

    @intensities.setter
    def intensities(self, value):
        self.intensity_vector = np.array(value)

    @property
    def x(self):
        return self.position_vector[0]

    @property
    def y(self):
        return self.position_vector[1]