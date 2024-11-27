import numpy as np
import os
import xgboost as xgb
import joblib
import json


def test_quality_flag():
    param_qf = np.load(os.path.join(os.path.dirname(__file__), "test_ipf520.npy"))
    mode = "wv2"
    aux_json = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "s1a-aux-ml2.json")
    with open(aux_json, "r") as fd:
        data = json.load(fd)["Models"]["QualityFlag"]
    
    QFpath = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", data["Dir"])

    regressor_Hs = xgb.XGBRegressor(objective='reg:squarederror', silent=True, verbosity=0)
    regressor_Hs.load_model(os.path.join(QFpath, data[mode]["hs"]))

    regressor_Wl = xgb.XGBRegressor(objective='reg:squarederror', silent=True, verbosity=0)
    regressor_Wl.load_model(os.path.join(QFpath, data[mode]["wl"]))

    regressor_Phi = xgb.XGBRegressor(objective='reg:squarederror', silent=True, verbosity=0)
    regressor_Phi.load_model(os.path.join(QFpath, data[mode]["phi"]))

    reg_file = os.path.join(QFpath, data[mode]["thresholds4QF"])
    thresh_single_WV12 = joblib.load(reg_file)

    y_err_pred_Hs = regressor_Hs.predict(param_qf.reshape(1, -1))
    y_err_pred_Wl = regressor_Wl.predict(param_qf.reshape(1, -1))
    y_err_pred_Phi = regressor_Phi.predict(param_qf.reshape(1, -1))

    assert np.allclose(np.array([0.7317485]), y_err_pred_Hs)
    assert np.allclose(np.array([0.17680828]), y_err_pred_Wl)
    assert np.allclose(np.array([28.993273]), y_err_pred_Phi)
    y_err_single_param = y_err_pred_Hs * y_err_pred_Wl * y_err_pred_Phi
    qf = np.clip(np.digitize(y_err_single_param, thresh_single_WV12[1:]), 0, 4)
    assert qf == 3
