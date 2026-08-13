import numpy as np
from spectra_class import spectrum, spectra
from preprocessing import preprocess_pipeline, despike_spectra, denoise_spectra, remove_baseline, normalize_spectra, shift_to_zero, background_removal_silicon

def test_pipeline_types():
    print("Creating dummy spectra test data...")
    wn = np.linspace(400, 1800, 500)
    int1 = np.sin(wn / 50) + 10 + np.random.normal(0, 0.1, 500)
    int2 = np.cos(wn / 50) + 12 + np.random.normal(0, 0.1, 500)
    
    # Add a dummy spike
    int1[100] += 50
    int2[200] += 50

    spec1 = spectrum(wn, int1, position_vector=[0, 0])
    spec2 = spectrum(wn, int2, position_vector=[1, 1])
    spec_list = [spec1, spec2]
    spectra_obj = spectra(spec_list)

    # 1. Test single spectrum
    print("\n[Test 1] Testing single 'spectrum' object...")
    res_single = preprocess_pipeline(spec1)
    assert isinstance(res_single, spectrum), f"Expected spectrum, got {type(res_single)}"
    print("✓ Passed! Returned single spectrum object.")

    # 2. Test list of spectrum objects
    print("\n[Test 2] Testing 'list[spectrum]'...")
    res_list = preprocess_pipeline(spec_list)
    assert isinstance(res_list, list), f"Expected list, got {type(res_list)}"
    assert isinstance(res_list[0], spectrum), f"Expected list item to be spectrum, got {type(res_list[0])}"
    print("✓ Passed! Returned list of spectrum objects.")

    # 3. Test 2D spectra container
    print("\n[Test 3] Testing 'spectra' 2D container...")
    res_spectra = preprocess_pipeline(spectra_obj)
    assert isinstance(res_spectra, spectra), f"Expected spectra, got {type(res_spectra)}"
    print("✓ Passed! Returned spectra 2D container.")

    # 4. Test standalone silicon background removal
    print("\n[Test 4] Testing standalone 'background_removal_silicon'...")
    res_bg = background_removal_silicon(spectra_obj)
    assert isinstance(res_bg, spectra), f"Expected spectra, got {type(res_bg)}"
    print("✓ Passed! Standalone background removal function works as expected.")

    print("\n==========================================")
    print("ALL TYPE TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    test_pipeline_types()
