import os
import sys
import numpy as np

# Ensure workspace is on sys.path
sys.path.insert(0, "/Users/bogiekissiov/Desktop/raman plotting")

from spectra_class import spectrum, spectra
from file_loaders import txt_to_spectra

def run_tests():
    print("--- Test 1: Creating individual spectrum objects ---")
    s1 = spectrum(wavenumber_vector=[100, 200, 300], intensity_vector=[10, 20, 30], position_vector=[1.0, 2.0])
    s2 = spectrum(wavenumber_vector=[100, 200, 300], intensity_vector=[15, 25, 35], position_vector=[3.0, 4.0])
    assert np.array_equal(s1.wavenumbers, np.array([100, 200, 300]))
    assert s1.x == 1.0 and s1.y == 2.0
    print("✓ spectrum class basic functionality passed.")

    print("\n--- Test 2: Creating spectra from spectrum_list ---")
    sp = spectra([s1, s2])
    assert sp.wavenumber_matrix.shape == (2, 3)
    assert sp.intensity_matrix.shape == (2, 3)
    assert sp.position_matrix.shape == (2, 2)
    assert np.array_equal(sp.x, np.array([1.0, 3.0]))
    assert np.array_equal(sp.y, np.array([2.0, 4.0]))
    print("✓ spectra(list) initialization passed.")

    print("\n--- Test 3: Unpacking spectra back to list ---")
    unpacked = sp.unpack()
    assert len(unpacked) == 2
    assert isinstance(unpacked[0], spectrum)
    assert np.array_equal(unpacked[0].intensities, np.array([10, 20, 30]))
    print("✓ spectra.unpack() passed.")

    print("\n--- Test 4: Creating spectra from_matrices ---")
    w_mat = np.array([[100, 200, 300], [100, 200, 300]])
    i_mat = np.array([[10, 20, 30], [15, 25, 35]])
    p_mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    sp_mat = spectra.from_matrices(w_mat, i_mat, p_mat)
    assert sp_mat.wavenumber_matrix.shape == (2, 3)
    assert np.array_equal(sp_mat.intensities, i_mat)
    print("✓ spectra.from_matrices() passed.")

    print("\n--- Test 5: Testing TXT Loader (synthetic file) ---")
    test_txt_path = "/Users/bogiekissiov/.gemini/antigravity-ide/brain/026f4e15-424a-4493-ac47-e6753d1f0471/scratch/test_data.txt"
    # Format: #X #Y #Wave #Intensity
    txt_content = (
        "#X\t#Y\t#Wave\t#Intensity\n"
        "0.0\t0.0\t100.0\t10.0\n"
        "0.0\t0.0\t200.0\t20.0\n"
        "1.0\t1.0\t100.0\t15.0\n"
        "1.0\t1.0\t200.0\t25.0\n"
    )
    with open(test_txt_path, "w") as f:
        f.write(txt_content)

    # Test pack=False
    list_res = txt_to_spectra(test_txt_path, pack=False)
    assert len(list_res) == 2
    assert isinstance(list_res[0], spectrum)
    assert list_res[0].x == 0.0 and list_res[1].x == 1.0

    # Test pack=True
    packed_res = txt_to_spectra(test_txt_path, pack=True)
    assert isinstance(packed_res, spectra)
    assert packed_res.intensity_matrix.shape == (2, 2)

    # Test spectrum.from_txt
    spec_from_txt = spectrum.from_txt(test_txt_path)
    # 2 spectra in file, should return list
    assert isinstance(spec_from_txt, list)

    # Test spectra.from_txt
    spectra_from_txt = spectra.from_txt(test_txt_path)
    assert isinstance(spectra_from_txt, spectra)

    print("✓ txt_to_spectra, spectrum.from_txt, and spectra.from_txt passed.")
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    run_tests()
