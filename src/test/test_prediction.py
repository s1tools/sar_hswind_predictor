import os
from glob import glob
import numpy as np
import json
from inference.run_inference import get_inference
from inference import config


def test_hsWindsea_predictions_from_ocn_file():
    """

    Returns:

    """
    input_test_dataset = glob(os.path.join(
        os.path.dirname(__file__),
        "referencedata",
        '*', 'measurement',
        "*.nc"), recursive=True)

    # TODO to be replaced by the official with a particular version
    input_aux_ml2 = os.path.join(os.path.dirname(__file__),
                                 "referencedata",
                                 "S1A_AUX_ML2_V20140406T133000_G20240429T091835.SAFE")

    path_ref_results = os.path.join(os.path.dirname(__file__),
                                    "reference_results",
                                    "s1_HsWindSea_inference_20241126-223213.json")
    with open(path_ref_results) as ref_json:
        ref_results = json.load(ref_json)

    infer_result = get_inference(list_l2_nc=input_test_dataset,
                                 batch_size=config.INFERENCE.BATCH_SIZE,
                                 outdir=config.filesystem.OUTPUT_FOLDER,
                                 aux_ml2=input_aux_ml2,
                                 verbosity=config.INFERENCE.VERBOSITY,
                                 no_log=True,
                                 no_json=False,
                                 export_result=False)

    for _k, _nc in infer_result['l2_product'].items():

        infer_hsWind_spec = infer_result['pred_hswindsea_from_ww3_spec'][_k]
        infer_psh0 = infer_result['pred_phs0_from_ww3_partitions'][_k]

        for _kk, _nc_ref in ref_results['l2_product'].items():
            if _nc == _nc_ref:
                expected_hsWind_Spec = ref_results['pred_hswindsea_from_ww3_spec'][_kk]
                expected_phs0 = ref_results['pred_phs0_from_ww3_partitions'][_kk]
                break

        # _____ check for hsWindsea from ww3 wave spectum
        assert np.allclose(infer_hsWind_spec, expected_hsWind_Spec,
                           atol=1e-02)

        # _____ check for hsWindSea from ww3 Hs wind partitions
        assert np.allclose(infer_psh0, expected_phs0,
                           atol=1e-02)
    print('--' * 12, ' Tests for Hs Wind Sea inference are OK', '--' * 12)


if __name__ == "__main__":
    test_hsWindsea_predictions_from_ocn_file()
