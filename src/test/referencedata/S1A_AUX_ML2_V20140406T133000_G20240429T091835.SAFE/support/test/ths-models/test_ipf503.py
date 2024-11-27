import os
import json
import sys
import numpy as np
from s1tools.sarhspredictor.load_quach_2020_keras_model import load_quach2020_model_v2
from s1tools.sarhspredictor.predict_with_quach2020_on_ocn_using_keras import main_level_1
from logbook import Logger, StreamHandler

StreamHandler(sys.stdout).push_application()

log = Logger('test')
aux_json = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "s1a-aux-ml2.json")
with open(aux_json, "r") as fd:
   data = json.load(fd)["Models"]["TotalHS"]
    
ths_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", data["Dir"])


def test_quach2020_model_wv2():
    model_filename = os.path.join(ths_path, data["wv2"]["h5"])
    heteroskedastic_2017 = load_quach2020_model_v2(model_filename)
    ff = 's1a-wv2-ocn-vv-20210201t083307-20210201t083310-036394-044589-004.nc'
    ff = os.path.join(os.path.dirname(__file__), ff)
    output_datatset = main_level_1(ff, heteroskedastic_2017, log)
    assert np.allclose(output_datatset['swh'].values, [5.7149305])
    assert np.allclose(output_datatset['swh_uncertainty'].values, [0.6566842])


def test_quach2020_model_wv1():
    model_filename = os.path.join(ths_path, data["wv1"]["h5"])
    heteroskedastic_2017 = load_quach2020_model_v2(model_filename)
    ff = 's1a-wv2-ocn-vv-20210201t083307-20210201t083310-036394-044589-004.nc'
    ff = os.path.join(os.path.dirname(__file__), ff)
    output_datatset = main_level_1(ff, heteroskedastic_2017, log)
    assert np.allclose(output_datatset['swh'].values, [5.7149305])
    assert np.allclose(output_datatset['swh_uncertainty'].values, [0.6566842])
