import os
from datetime import datetime
import numpy as np
import pandas as pd
import json
from json import loads
import tensorflow as tf

from inference.cmdline import cmdline
from inference import config
from inference.cstm_logger import create_logger
from inference.tools import get_sar_HsWind_featuresEng, apply_MinMax_scaler


def get_inference(list_l2_nc,
                  batch_size=config.INFERENCE.BATCH_SIZE,
                  outdir=config.filesystem.OUTPUT_FOLDER,
                  aux_ml2=config.filesystem.AUX_ML2,
                  verbosity=config.INFERENCE.VERBOSITY,
                  no_log=True,
                  no_json=False,
                  export_result=False) -> None:
    """

    Args:
        list_l2_nc: list of ocn product (netcdf) to Estimate Hs wind sea with DL
        batch_size: the batch size for inference
        outdir: the output workdir for potentilly export and save the resutls if needed
        aux_ml2: the AUX_ML2 data (con be founded in : https://sar-mpc.eu/)
        verbosity: the log verbosity
        no_log: to create a log file or no
        no_json : to return the results in json format (default) or in csv (otherwise)
        export_result : to export or not the result

        For each ocn file, the code will :
            - prepare the features for the model
            - compute the prediction
            - format the result as needed  (json or csv)

        /!\ There is no need to GPU to compute the prediction
    Returns:

    """

    if not os.path.exists(aux_ml2):
        raise IOError(f'no such file or directory : {aux_ml2}')

    mission = os.path.basename(aux_ml2).split('_')[0].lower()

    os.makedirs(outdir, exist_ok=True)

    # ____Process the list of OCN products
    try:
        if not isinstance(list_l2_nc, list):
            with open(list_l2_nc, "r") as file:
                lines = file.readlines()
                listing_l2_nc = [line.rstrip() for line in lines if line.strip() != ""]
        else:
            listing_l2_nc = list_l2_nc

    except Exception as exp:
        raise IOError(f'Error while reading the listing L2 product(s) : {exp}')

    # # ____ Initiate the logger if requested
    log = not no_log
    if log:
        logdir = os.path.join(outdir, 'logs')
        os.makedirs(logdir, exist_ok=True)

        datetime_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        path_to_log = os.path.join(logdir, f'run_inference_{datetime_run}.log')
        logger = create_logger(path_to_log, verbose=verbosity)
    else:
        datetime_run = datetime.now().strftime("%Y%m%d-%H%M%S")
        logger = None

    # ____ Prepare the products listing for inference
    df = pd.DataFrame()
    df['l2_product'] = listing_l2_nc
    df['mode'] = df.apply(lambda x: x['l2_product'].split('/')[-1].split('-')[1], axis=1)
    df['mission'] = df.apply(lambda x: x['l2_product'].split('/')[-1].split('-')[0], axis=1)
    df = df[df['mission'] == mission]
    df.reset_index(drop='index', inplace=True)

    if df.shape[0] > 0:
        if log:
            logger.info(f'Nr of L2 OCN products to be treated : {df.shape[0]}')
    else:
        if log:
            logger.info(
                'List of L2 product(s) to be treated is empty. Check product(s) and the submitted AUX_ML2')
        return None

    # ___Load the models & scalers from the aux_ml2
    models = {}
    scalers = {}

    aux_ml2_data_path: str = os.path.join(aux_ml2, "data")
    try:
        with open(os.path.join(aux_ml2_data_path, f"{mission}-aux-ml2.json")) as fd:
            aux_ml2_data: dict = json.load(fd)["Models"]["HsWindSea"]
        modelScalers_HsWind_path: str = os.path.join(aux_ml2_data_path, aux_ml2_data['Dir'])

        targets_names = list(aux_ml2_data['wv1'].keys())
    except Exception as exp:
        raise IOError(f'Error ({exp}) while reading AUX_ML2 files')

    try:
        for _mode in ['wv1', 'wv2']:
            _dict_mod = {}
            _dict_scl = {}
            for _targ in targets_names:
                _dict_mod[_targ] = tf.keras.models.load_model(
                    os.path.join(modelScalers_HsWind_path, aux_ml2_data[_mode][_targ]['h5']), compile=False)
                with open(os.path.join(modelScalers_HsWind_path, aux_ml2_data[_mode][_targ]['scaler'])) as sf:
                    _dict_scl[_targ]: dict = json.load(sf)
            models.update({_mode: _dict_mod})
            scalers.update({_mode: _dict_scl})
    except Exception as exp:
        raise IOError(f'Error ({exp}) to load models and scalers')

    # ____ For each target, we extract the features names.
    ordered_features = {}
    for _targ in targets_names:
        ordered_features[_targ] = scalers['wv1'][_targ]['features_names_order']

    # ____ Convert the dataframe metadata to dict for faster processing and get the features ready for inference
    features_imacs_names = config.CONSTANTS.IMACS_NAMES
    df2dict = df.to_dict('records')
    procs_dict = []

    for ii in range(len(df2dict)):
        ii_sample = df2dict[ii]
        try:
            featEng_samp = get_sar_HsWind_featuresEng(ii_sample['l2_product'],
                                                      imacs_names=features_imacs_names,
                                                      log=log)
            for col in featEng_samp.columns.tolist():
                ii_sample.update({col: featEng_samp.loc[0, col]})

            procs_dict.append(ii_sample)
        except Exception as exp:
            print(exp)
            pass
    check_proc = False
    try:
        proc_features = pd.DataFrame(procs_dict)
        if proc_features.shape[0] > 0:
            check_proc = True
            if log:
                logger.info('Features engineering was applied correctly')
    except Exception as exp:
        if log:
            logger.error(f'Error ({exp}) to process features for inference')
        return None

    # ____ If all features are ready, we apply the normalization and start serving
    if check_proc:

        df_wv1 = proc_features[proc_features['mode'] == 'wv1'].copy(deep=True)
        df_wv2 = proc_features[proc_features['mode'] == 'wv2'].copy(deep=True)

        df_wv1.reset_index(drop='index', inplace=True)
        df_wv2.reset_index(drop='index', inplace=True)

        for _mode, _data in zip(['wv1', 'wv2'],
                                [df_wv1, df_wv2]):
            if _data.shape[0] > 0:
                for _targ in targets_names:
                    try:
                        _df_tmp = _data[ordered_features[_targ]].copy(deep=True)
                        _df_tmp.reset_index(drop='index', inplace=True)
                        _scaled_df_tmp = _df_tmp.apply(lambda x: apply_MinMax_scaler(x,
                                                                                     np.array(
                                                                                         scalers[_mode][_targ][
                                                                                             'data_min']),
                                                                                     np.array(
                                                                                         scalers[_mode][_targ][
                                                                                             'data_range']),
                                                                                     tuple(scalers[_mode][_targ][
                                                                                               'scaler_range'])
                                                                                     ),
                                                       axis=1)
                        if _scaled_df_tmp.shape[0] == 1:
                            ready_features_arr = np.expand_dims(_scaled_df_tmp.values.squeeze(),
                                                                axis=0)
                        elif _scaled_df_tmp.shape[0] > 1:
                            ready_features_arr = _scaled_df_tmp.values.squeeze()

                        pred = models[_mode][_targ].predict(ready_features_arr,
                                                            verbose=verbosity,
                                                            batch_size=batch_size)

                        proc_pred = np.maximum(0, pred.squeeze())
                        _data.loc[:, f'pred_{_targ}'] = proc_pred
                        if log:
                            logger.info(f'Inference finished for : mode {_mode} - variable {_targ}.')
                    except Exception as exp:
                        if log:
                            logger.error(f' Error ({exp}) while computing inference.')

        infer_csv_format = pd.concat([df_wv1, df_wv2], axis=0)
        infer_csv_format.reset_index(drop='index', inplace=True)
        infer_csv_format['l2_product'] = infer_csv_format.apply(lambda x: os.path.basename(x['l2_product']),
                                                                axis=1)

        # ____ Preparing and formatting the result as requested
        filename_results = f's1_HsWindSea_inference_{datetime_run}'

        if no_json:
            if export_result:
                export_fl = f'{filename_results}.csv'
                infer_csv_format.to_csv(os.path.join(outdir, export_fl),
                                        index=False)
            return infer_csv_format

        else:
            col_pred = [col for col in infer_csv_format.columns.tolist() if 'pred' in col]
            col2extract = ['l2_product'] + col_pred
            extract_df = infer_csv_format[col2extract].copy(deep=True)
            result_json_format = extract_df.to_json(orient='columns')
            parsed_result_json_format = loads(result_json_format)
            if export_result:
                export_fl = f'{filename_results}.json'
                extract_df.to_json(os.path.join(outdir, export_fl),
                                   orient='columns',
                                   double_precision=5,
                                   indent=4,
                                   force_ascii=False)
            return parsed_result_json_format


if __name__ == "__main__":
    args = cmdline().parse_args()
    get_inference(**vars(args))
