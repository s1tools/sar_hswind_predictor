import os
from glob import glob
import numpy as np

from hswind_inference.run_inference import get_inference
from hswind_inference import config

# ================= Reference Inference to be checked ==============================
ref_results = {
    "l2_product": {
        "0": "s1a-wv1-ocn-vv-20231122t071316-20231122t071319-051326-063162-031.nc",
        "1": "s1a-wv2-ocn-vv-20231122t065103-20231122t065106-051326-063161-058.nc"
    },
    "pred_hswindsea_from_ww3_spec": {
        "0": 1.936,
        "1": 0.732
    },
    "pred_phs0_from_ww3_partitions": {
        "0": 2.377,
        "1": 1.291
    }
}


# =================================================================================

def test_hsWindsea_predictions_from_ocn_file():
    """

    Returns:

    """
    input_test_dataset = glob(os.path.join(
        os.path.dirname(__file__),
        '**',
        "*.nc"), recursive=True)

    # TODO to be replaced by the official with a particular version
    input_aux_ml2 = '/'.join(os.getcwd().split('/')[:-3])

    infer_result = get_inference(list_l2_nc=input_test_dataset,
                                 batch_size=config.INFERENCE.BATCH_SIZE,
                                 outdir=config.filesystem.OUTPUT_FOLDER,
                                 aux_ml2=input_aux_ml2,
                                 verbosity=config.INFERENCE.VERBOSITY,
                                 log=False,
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
    print('==' * 12, ' Tests for Hs Wind Sea inference are OK', '==' * 12)


if __name__ == "__main__":
    test_hsWindsea_predictions_from_ocn_file()
