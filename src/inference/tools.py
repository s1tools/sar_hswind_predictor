import datetime
import os.path
from datetime import datetime
import numpy as np
import numpy.ma as ma
import netCDF4
import pandas as pd
import xarray as xr


def get_input_cwave_vector_from_ocn(cspc_re, cspc_im, kx, ky, incidenceangle, s0, nv, datedt, lons,
                                    lats,
                                    satellite):
    """
    v2 is copy/paste from v1c, it is a CWAVE algo python that use the polSpec from OCN products,
    it extracts the 20 param and use a CWAVE v2 model tuned on altimeters (only numpy dependancies)
    provided by Yannick Glaser (University of Hawaii)
    date creation: 9 Juillet 2019
    example 1:
       s1a-wv1-ocn-vv-20151130t175854-20151130t175857-008837-00c9eb-087.nc
       hsSM=3.4485

    example 2:
       s1a-wv2-ocn-vv-20151130t201457-20151130t201500-008838-00c9f5-046.nc
       hsSM=1.3282

    :param cspc_re: oswCartSpecRe 128x185 (kx,ky)
    :param cspc_im: oswCartSpecIm 128x185 (kx,ky)
    :parma kx : oswKx/oswGroundRngSize
    :param ky : oswKy/oswAziSize
    :param incidenceangle:
    :param s0: oswNrcs
    :param nv:
    :param datedt:
    :param lons:
    :param lats:
    :param satellite: s1a or...
    :return: cwave params
    """
    # Constants for CWAVE
    # ===================

    kmax = 2 * np.pi / 60  # kmax for empirical Hs
    kmin = 2 * np.pi / 625  # kmin for empirical Hs
    ns = 20  # number of variables in orthogonal decomposition
    S = np.ones((ns, 1)) * np.nan

    # Process KY and KX
    KX, KY = np.meshgrid(kx, ky)
    mask = (abs(KX) < kmin) & (abs(KX) > kmax) & (abs(KY) < kmin) & (abs(KY) > kmax)
    KX_masked = ma.masked_where(mask, KX, copy=True).filled(np.nan)
    KY_masked = ma.masked_where(mask, KY, copy=True).filled(np.nan)

    # DKX = np.ones(KX.shape) * 0.003513732113299
    # DKY = np.ones(KX.shape) * 0.002954987953815

    # Compute dkx and dky
    dkx = np.zeros(kx.size)
    dkx[0] = kx[1] - kx[0]
    dkx[-1] = kx[-1] - kx[-2]
    dkx[1:-1] = (kx[2:] - kx[0:-2]) / 2

    dky = np.zeros(ky.size)
    dky[0] = ky[1] - ky[0]
    dky[-1] = ky[-1] - ky[-2]
    dky[1:-1] = (ky[2:] - ky[0:-2]) / 2
    DKX, DKY = np.meshgrid(dkx, dky)

    flagKcorrupted = False
    for _var in [s0, nv, incidenceangle]:
        if not np.isfinite(_var) or isinstance(_var, np.ma.core.MaskedConstant):  # or s0.mask.all():
            _var = 0
            flagKcorrupted = True

    # Store the obs metadata
    subset_ok = pd.DataFrame()
    subset_ok.loc[0, 'todSAR'] = _conv_time(netCDF4.date2num(datedt, 'hours since 2010-01-01T00:00:00Z UTC'))
    subset_ok.loc[0, 'lonSAR'] = lons
    subset_ok.loc[0, 'latSAR'] = lats
    subset_ok.loc[0, 'incidenceAngle'] = incidenceangle
    subset_ok.loc[0, 'sigma0'] = s0
    subset_ok.loc[0, 'normalizedVariance'] = nv

    # Mission encoding
    if satellite == 's1a':
        subset_ok.loc[0, 'sentinelType'] = 1
    else:
        subset_ok.loc[0, 'sentinelType'] = 0

    # Initialise cwave as DF storing
    for jj in range(ns):
        subset_ok.loc[0, 's' + str(jj)] = np.nan

    # Main cwave computing
    if all([_itm.shape == (128, 185) for _itm in [cspc_re, cspc_im]]) and (cspc_re > 0).any() and not flagKcorrupted:

        cspc = np.sqrt(cspc_re ** 2 + cspc_im ** 2)
        cspc_masked = ma.masked_where(mask.T, cspc, copy=True).filled(np.nan)

        # Compute Orthogonal Moments
        gamma = 2
        a1 = (gamma ** 2 - np.power(gamma, 4)) / (gamma ** 2 * kmin ** 2 - kmax ** 2)
        a2 = (kmax ** 2 - np.power(gamma, 4) * kmin ** 2) / (kmax ** 2 - gamma ** 2 * kmin ** 2)

        # Ellipse
        tmp = a1 * np.power(KX_masked, 4) + a2 * KX_masked ** 2 + KY_masked ** 2

        # eta
        qt = (KX_masked ** 2 + KY_masked ** 2) * tmp * np.log10(kmax / kmin)
        eta = np.sqrt(np.true_divide((2. * tmp), qt))

        alphak = 2. * ((np.log10(np.sqrt(tmp), where=tmp > 0) - np.log10(kmin)) / np.log10(kmax / kmin)) - 1
        alphak[(alphak ** 2) > 1] = 1.

        alphat = np.arctan2(KY_masked, KX_masked)

        # Gegenbauer polynomials
        tmp = abs(np.sqrt(1 - alphak ** 2))  # imaginary???

        g1 = 1 / 2. * np.sqrt(3) * tmp
        g2 = 1 / 2. * np.sqrt(15) * alphak * tmp
        g3 = np.dot((1 / 4.) * np.sqrt(7. / 6.), (15. * np.power(alphak, 2) - 3.)) * tmp  #
        g4 = (1 / 4.) * np.sqrt(9. / 10) * (35. * np.power(alphak, 3) - 15. * alphak ** 2) * tmp

        # Harmonic functions
        f1 = np.sqrt(1 / np.pi) * np.cos(0. * alphat)
        f2 = np.sqrt(2 / np.pi) * np.sin(2. * alphat)
        f3 = np.sqrt(2 / np.pi) * np.cos(2. * alphat)
        f4 = np.sqrt(2 / np.pi) * np.sin(4. * alphat)
        f5 = np.sqrt(2 / np.pi) * np.cos(4. * alphat)

        # Weighting functions
        h = np.ones((KX.shape[0], KX.shape[1], 20))
        h[:, :, 0] = g1 * f1 * eta
        h[:, :, 1] = g1 * f2 * eta
        h[:, :, 2] = g1 * f3 * eta
        h[:, :, 3] = g1 * f4 * eta
        h[:, :, 4] = g1 * f5 * eta
        h[:, :, 5] = g2 * f1 * eta
        h[:, :, 6] = g2 * f2 * eta
        h[:, :, 7] = g2 * f3 * eta
        h[:, :, 8] = g2 * f4 * eta
        h[:, :, 9] = g2 * f5 * eta
        h[:, :, 10] = g3 * f1 * eta
        h[:, :, 11] = g3 * f2 * eta
        h[:, :, 12] = g3 * f3 * eta
        h[:, :, 13] = g3 * f4 * eta
        h[:, :, 14] = g3 * f5 * eta
        h[:, :, 15] = g4 * f1 * eta
        h[:, :, 16] = g4 * f2 * eta
        h[:, :, 17] = g4 * f3 * eta
        h[:, :, 18] = g4 * f4 * eta
        h[:, :, 19] = g4 * f5 * eta
        try:
            P = cspc_masked / (np.nansum(np.nansum(cspc_masked * DKX.T * DKY.T)))  # original

            for jj in range(ns):
                petit_h = h[:, :, jj].squeeze()
                S[jj] = np.nansum(np.nansum(petit_h * P.T * DKX * DKY))
                subset_ok.loc[0, 's' + str(jj)] = S[jj][0]
        except:
            flagKcorrupted = False

    return subset_ok, S, flagKcorrupted


def get_sar_HsWind_featuresEng(l2_nc, imacs_names=list, time_sar=None, log=None):
    """
    Aim : Extract the necessary info from ocn osw porduct to preapre inference
    :param : sar_data : L2 OCN or L2 OSW as xr.dataset

    return : dataframe  with processed SAR features
    """
    _ds = pd.DataFrame()

    if os.path.isfile(l2_nc):
        try:
            l2_data = xr.open_dataset(l2_nc)
        except Exception as exp:
            if log:
                log.error(f'Error ({exp}) to read : {l2_nc}')
            return _ds
    elif isinstance(l2_nc, xr.Dataset):
        l2_data = l2_nc
    else:
        if log:
            log.error('input type of {l_nc} not recognized')
        return _ds

    # __________ADD SAR INFO
    if time_sar is None:
        time_sar = datetime.strptime(l2_data.attrs['lastMeasurementTime'], '%Y-%m-%dT%H:%M:%S.%fZ')

    _ds.loc[0, 'oswWindSpeed'] = np.squeeze(l2_data['oswWindSpeed'].values)
    _ds.loc[0, 'oswWindDirection'] = np.squeeze(l2_data['oswWindDirection'].values)
    _ds.loc[0, 'oswHeading'] = np.squeeze(l2_data['oswHeading'].values)
    _ds.loc[0, 'oswSnr'] = np.squeeze(l2_data['oswSnr'].values)
    _ds.loc[0, 'oswNv'] = np.squeeze(l2_data['oswNv'].values)
    _ds.loc[0, 'oswNrcs'] = np.squeeze(l2_data['oswNrcs'].values)
    _ds['oswSnr_lin'] = _ds.apply(lambda x: 10 ** (x['oswSnr'] / 10), axis=1)
    _ds.loc[0, 'oswAzCutoff'] = np.squeeze(l2_data['oswAzCutoff'].values)
    _ds.loc[0, 'oswSkew'] = np.squeeze(l2_data['oswSkew'].values)
    # _________CONVERT WINDIR CONVENTION
    _ds['windir_sar_capt'] = _ds.apply(lambda x: windir2sensorGeo(x['oswWindDirection'],
                                                                         x['oswHeading']), axis=1)

    # __________ADD IMACS FEATURES
    ky_sar = np.squeeze(l2_data['oswKy'].values)
    kx_sar = np.squeeze(l2_data['oswKx'].values)
    oswGroundRngSize = np.squeeze(l2_data['oswGroundRngSize'].values)
    oswAziSize = np.squeeze(l2_data['oswAziSize'].values)

    imacs = get_imacs(np.squeeze(l2_data['oswCartSpecIm'].values),
                      ky_sar,
                      kx_sar,
                      oswGroundRngSize,
                      oswAziSize)

    for (ind, var_imacs) in enumerate(imacs_names):
        _ds.loc[0, var_imacs] = imacs[ind]

    # _________WIND DIR PROJECTION IN RANGE/AZIMUTH
    _ds.loc[0, 'windir_sar_Ucapt'] = np.cos(np.deg2rad(_ds.loc[0, 'windir_sar_capt']))
    # _ds.loc[0, 'windir_sar_Vcapt'] = abs(np.sin(np.deg2rad(_ds.loc[0, 'windir_sar_capt'])))

    # _________COMPUTE CWAVES
    oswKx_denorm = kx_sar / oswGroundRngSize
    oswKy_denorm = ky_sar / oswAziSize

    _, cwaves_vector, _ = get_input_cwave_vector_from_ocn(
        cspc_re=np.squeeze(l2_data['oswCartSpecRe'][..., 2].T),
        cspc_im=abs(np.squeeze(l2_data['oswCartSpecIm'][..., 2].T)),
        kx=oswKx_denorm,
        ky=oswKy_denorm,
        incidenceangle=np.squeeze(l2_data['oswIncidenceAngle'].values),
        s0=np.squeeze(l2_data['oswNrcs'].values),
        nv=np.squeeze(l2_data['oswNv'].values),
        datedt=time_sar,
        lons=np.squeeze(l2_data['oswLon'].values),
        lats=np.squeeze(l2_data['oswLat'].values),
        satellite=l2_data.attrs['missionName'].lower())
    cwaves_vector = np.squeeze(cwaves_vector)
    for ii in range(len(cwaves_vector)):
        try:
            _ds.loc[0, f'cwave_{ii + 1}'] = cwaves_vector[ii]
        except:
            _ds.loc[0, f'cwave_{ii + 1}'] = np.nan
    return _ds


def _conv_time(in_t):
    """
    Converts data acquisition time

    Args:
        in_t: time of data acquisition in format hours since 2010-01-01T00:00:00Z UTC

    Returns:
        Encoding of time where 00:00 and 24:00 are -1 and 12:00 is 1
    """
    in_t = in_t % 24
    return 2 * np.sin((2 * np.pi * in_t) / 48) - 1


def get_imacs(cartspecim, oswky, oswkx, oswGroundRngSize, oswAziSize):
    """ ESTIMATE THE IMACS VARIABLE from a SAR_file netcdf file
    """

    # data = nc.Dataset(sar_file)
    # cartspecre = data.variables['oswCartSpecRe'][0,0,:,:,:]
    # cartspecim = data.variables['oswCartSpecIm'][0,0,:,:,:]
    ## To normalize by oswGroundRgSize, oswAziSize
    oswky = oswky.squeeze() / oswAziSize
    oswkx = oswkx.squeeze() / oswGroundRngSize
    # oswky à ré-échelonner, tout comme oswkx
    ily1 = np.argmin(np.abs(np.true_divide(-2 * np.pi, oswky) - 600))
    ily2 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswky) - 600))
    ilx1 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 15))
    ilx2 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 20))

    ## np.round(2*np.pi/oswkx.data[0][5::10])
    ## --> array([899., 290., 163., 106.,  74.,  52.,  38.,  27.,  20.,  15.,  11.,
    ##      8.,   6.], dtype=float32)
    ilx3 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 27))
    ilx4 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 38))
    ilx5 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 52))
    ilx6 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 74))
    ilx7 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 106))
    ilx8 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 163))
    ilx9 = np.argmin(np.abs(np.true_divide(2 * np.pi, oswkx) - 290))

    IMACS = np.array([np.mean(cartspecim[ily1:ily2 + 1, ilx2:ilx1 + 1, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx3:ilx2, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx4:ilx3, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx5:ilx4, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx6:ilx5, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx7:ilx6, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx8:ilx7, 1]),
                      np.mean(cartspecim[ily1:ily2 + 1, ilx9:ilx8, 1])])

    return (IMACS)


def windir2sensorGeo(wdir, heading):
    try:
        wind_conv = (wdir - heading + 90) % 360
    except:
        wind_conv = np.nan
    return wind_conv


def apply_MinMax_scaler(X, x_min: np.array, x_range: np.array, scaler_range: tuple):
    X_std = (X - x_min) / x_range
    X_scaled = X_std * (scaler_range[1] - scaler_range[0]) + scaler_range[0]
    return X_scaled
