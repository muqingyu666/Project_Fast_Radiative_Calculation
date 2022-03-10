# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 20:51:45 2021

@author: Muqy o(*￣▽￣*)ブ
"""
import math
from math import *

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from numba import jit
import glob

#################################
############  Pressure dic ######
#################################

pressure = {
    "press43": np.array(
        (
            1.00000e-01,
            2.90000e-01,
            6.90000e-01,
            1.42000e00,
            2.61100e00,
            4.40700e00,
            6.95000e00,
            1.03700e01,
            1.48100e01,
            2.04000e01,
            2.72600e01,
            3.55100e01,
            4.52900e01,
            5.67300e01,
            6.99700e01,
            8.51800e01,
            1.02050e02,
            1.22040e02,
            1.43840e02,
            1.67950e02,
            1.94360e02,
            2.22940e02,
            2.53710e02,
            2.86600e02,
            3.21500e02,
            3.58280e02,
            3.96810e02,
            4.36950e02,
            4.78540e02,
            5.21460e02,
            5.65540e02,
            6.10600e02,
            6.56430e02,
            7.02730e02,
            7.49120e02,
            7.95090e02,
            8.39950e02,
            8.82800e02,
            9.22460e02,
            9.57440e02,
            9.85880e02,
            1.00543e03,
            1.01325e03,
        )
    ),
    "press100": np.array(
        (
            5.00e-03,
            1.60e-02,
            3.8000e-02,
            7.70000e-02,
            1.37000e-01,
            2.24000e-01,
            3.45000e-01,
            5.06000e-01,
            7.14000e-01,
            9.75000e-01,
            1.29700e00,
            1.68700e00,
            2.15300e00,
            2.70100e00,
            3.34000e00,
            4.07700e00,
            4.92000e00,
            5.87800e00,
            6.95700e00,
            8.16500e00,
            9.51200e00,
            1.10040e01,
            1.26490e01,
            1.44560e01,
            1.64320e01,
            1.85850e01,
            2.09220e01,
            2.34530e01,
            2.61830e01,
            2.91210e01,
            3.22740e01,
            3.56510e01,
            3.92570e01,
            4.31000e01,
            4.71880e01,
            5.15280e01,
            5.61260e01,
            6.09890e01,
            6.61250e01,
            7.15400e01,
            7.72400e01,
            8.32310e01,
            8.95200e01,
            9.61140e01,
            1.03017e02,
            1.10237e02,
            1.17778e02,
            1.25646e02,
            1.33846e02,
            1.42385e02,
            1.51266e02,
            1.60496e02,
            1.70078e02,
            1.80018e02,
            1.90320e02,
            2.00989e02,
            2.12028e02,
            2.23442e02,
            2.35234e02,
            2.47408e02,
            2.59969e02,
            2.72919e02,
            2.86262e02,
            3.00000e02,
            3.14137e02,
            3.28675e02,
            3.43618e02,
            3.58966e02,
            3.74724e02,
            3.90893e02,
            4.07474e02,
            4.24470e02,
            4.41882e02,
            4.59712e02,
            4.77961e02,
            4.96630e02,
            5.15720e02,
            5.35232e02,
            5.55167e02,
            5.75525e02,
            5.96306e02,
            6.17511e02,
            6.39140e02,
            6.61192e02,
            6.83667e02,
            7.06565e02,
            7.29886e02,
            7.53628e02,
            7.77790e02,
            8.02371e02,
            8.27371e02,
            8.52788e02,
            8.78620e02,
            9.04866e02,
            9.31524e02,
            9.58591e02,
            9.86067e02,
            1.01395e03,
            1.04223e03,
            1.07092e03,
        )
    ),
}


#########################################################################
######  Calculate fast transmittance from usr specified coef & pred #####
#########################################################################

"""
Auxiliary math function using numba

"""


@jit(nopython=True)
def sum_mutiply(
    channels, profiles, angles, levels, predictor, coef
):
    temp_1 = np.zeros((channels, profiles, angles, levels))
    for k in range(channels):
        for i in range(profiles):
            for j in range(angles):
                temp_1[k, i, j, :] = np.sum(
                    np.multiply(
                        predictor[k, i, j, :, :], coef[k, :, :],
                    ),
                    axis=0,
                )
    return temp_1


@jit(nopython=True)
def sum_temp(channels, profiles, angles, levels, temp_1):
    temp_2 = np.zeros((channels, profiles, angles, levels + 1))
    for k in range(channels):
        for i in range(profiles):
            for j in range(angles):
                for n in range(1, levels + 1):
                    if temp_1[k, i, j, n - 1] < 0:
                        temp_2[k, i, j, n] = (
                            temp_2[k, i, j, n - 1]
                            + temp_1[k, i, j, n - 1]
                        )
                    else:
                        temp_2[k, i, j, n] = temp_2[k, i, j, n - 1]
    return temp_2


@jit(nopython=True)
def calc_trans(channels, profiles, angles, levels, temp_2):
    trans = np.zeros((channels, profiles, angles, levels + 1))
    for i in range(0, levels + 1):
        trans[:, :, :, i] = np.exp(-np.abs(temp_2[:, :, :, i]))
    return trans


#######################################################################

"""
Main function 

"""


class TransmittanceCalculator(object):
    """
    [Main code for transmittance calculation]

    """

    def __init__(
        self,
        coef_specified,
        pred_specified,
        channels,
        angles,
        profiles,
        levels=100,
        coef_dir="/RAID01/data/muqy/Project_Radiation/",
        pred_dir="/RAID01/data/muqy/Project_Radiation/",
    ):

        self.coef_dir = coef_dir
        self.pred_dir = pred_dir
        self.coef_specified = coef_specified
        self.pred_specified = pred_specified
        self.channels = channels
        self.angles = angles
        self.profiles = profiles
        self.levels = levels

    def CalcTrans_CoefPred_oz(self, NX=11):
        """
        Parameters
        ----------
        NX : int, optional
            _description_, by default 11

        Returns trans

        """
        # read coef
        coef_oz = (
            np.array(
                pd.read_csv(
                    self.coef_dir
                    + "coef_pythonformat_oz"
                    + str(self.coef_specified)
                    + ".txt",
                    delim_whitespace=True,
                    header=None,
                )
            )
            .reshape(self.channels, self.levels, NX)
            .transpose((0, 2, 1))
        )

        # read predictors
        predictor_oz = (
            np.array(
                pd.read_csv(
                    self.pred_dir
                    + "predictors_oz"
                    + str(self.pred_specified)
                    + ".txt",
                    delim_whitespace=True,
                    header=None,
                )
            )
            .reshape(
                self.channels,
                self.levels,
                self.angles,
                self.profiles,
                NX,
            )
            .transpose((0, 3, 2, 4, 1))
        )

        temp_oz1 = sum_mutiply(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            predictor_oz,
            coef_oz,
        )
        temp_oz2 = sum_temp(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            temp_oz1,
        )
        trans_oznc = calc_trans(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            temp_oz2,
        )
        trans_oznc[np.isnan(trans_oznc)] = 1
        return trans_oznc, temp_oz1

    def CalcTrans_CoefPred_wv(self, NX=15):
        """
        Parameters
        ----------
        NX : int, optional
            _description_, by default 15

        Returns trans

        """
        # read coef
        coef_wv = (
            np.array(
                pd.read_csv(
                    self.coef_dir
                    + "coef_pythonformat_wv"
                    + str(self.coef_specified)
                    + ".txt",
                    delim_whitespace=True,
                    header=None,
                )
            )
            .reshape(self.channels, self.levels, NX)
            .transpose((0, 2, 1))
        )

        # read predictors
        predictor_wv = (
            np.array(
                pd.read_csv(
                    self.pred_dir
                    + "predictors_wv"
                    + str(self.pred_specified)
                    + ".txt",
                    delim_whitespace=True,
                    header=None,
                )
            )
            .reshape(
                self.channels,
                self.levels,
                self.angles,
                self.profiles,
                NX,
            )
            .transpose((0, 3, 2, 4, 1))
        )

        temp_wv1 = sum_mutiply(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            predictor_wv,
            coef_wv,
        )
        temp_wv2 = sum_temp(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            temp_wv1,
        )
        trans_wvnc = calc_trans(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            temp_wv2,
        )
        trans_wvnc[np.isnan(trans_wvnc)] = 1
        return trans_wvnc, temp_wv1

    def CalcTrans_CoefPred_mg(
        self, NX=11,
    ):
        """
        Parameters
        ----------
        NX : int, optional
            _description_, by default 11

        Returns trans

        """
        # read coef
        coef_mg = (
            np.array(
                pd.read_csv(
                    self.coef_dir
                    + "coef_pythonformat_mg"
                    + str(self.coef_specified)
                    + ".txt",
                    delim_whitespace=True,
                    header=None,
                )
            )
            .reshape(self.channels, self.levels, NX)
            .transpose((0, 2, 1))
        )

        # read predictors
        predictor_mg = (
            np.array(
                pd.read_csv(
                    self.pred_dir
                    + "predictors_mg"
                    + str(self.pred_specified)
                    + ".txt",
                    delim_whitespace=True,
                    header=None,
                )
            )
            .reshape(
                self.channels,
                self.levels,
                self.angles,
                self.profiles,
                NX,
            )
            .transpose((0, 3, 2, 4, 1))
        )

        temp_mg1 = sum_mutiply(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            predictor_mg,
            coef_mg,
        )
        temp_mg2 = sum_temp(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            temp_mg1,
        )
        trans_mgnc = calc_trans(
            self.channels,
            self.profiles,
            self.angles,
            self.levels,
            temp_mg2,
        )
        trans_mgnc[np.isnan(trans_mgnc)] = 1

        return trans_mgnc, temp_mg1

    def RMSE_channels(self, array0, array1):
        return calc_RMSE_channel(
            array0, array1, self.levels, self.channels, self.angles
        )

    def RMSE_profiles(self, array0, array1):
        return calc_RMSE_profile(
            array0, array1, self.levels, self.profiles, self.angles
        )

    def CalcTrans_CoefPred_all(
        self, array_mix, array_wv, array_oz,
    ):
        """
        Calc all transmittance

        Parameters
        ----------
        array_mix : _type_
            _description_
        array_wv : _type_
            _description_
        array_oz : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        temp_all = array_wv + array_oz + array_mix

    def trans_all_calculator(
        self, temp_mg_tigr=0, temp_wv_tigr=0, temp_oz_tigr=0
    ):
        trans_all = np.zeros(
            (
                self.channels,
                self.profiles,
                self.angles,
                self.levels + 1,
            )
        )
        temp_all1 = np.zeros(
            (
                self.channels,
                self.profiles,
                self.angles,
                self.levels + 1,
            )
        )

        temp_all = temp_mg_tigr + temp_wv_tigr + temp_oz_tigr
        for k in range(self.channels):
            for i in range(self.profiles):
                for j in range(self.angles):
                    for n in range(1, self.levels + 1):
                        if temp_all[k, i, j, n - 1] < 0:
                            temp_all1[k, i, j, n] = (
                                temp_all1[k, i, j, n - 1]
                                + temp_all[k, i, j, n - 1]
                            )
                        else:
                            temp_all1[k, i, j, n] = temp_all1[
                                k, i, j, n - 1
                            ]

        for i in range(self.levels + 1):
            trans_all[:, :, :, i] = np.exp(
                -abs(temp_all1[:, :, :, i])
            )
        return trans_all


#########################################################################
############  Calculate transmittance RMSE ##############################
#########################################################################


@jit(nopython=True)
def calc_RMSE_channel(array0, array1, levels, chans, angles):
    RMSE = np.zeros((levels, chans, angles))

    for i in range(levels):
        for j in range(angles):
            for n in range(chans):
                RMSE[i, n, j] = math.sqrt(
                    np.mean(
                        np.square(
                            array0[n, :, j, i] - array1[n, :, j, i]
                        )
                    )
                )
    return RMSE


@jit(nopython=True)
def calc_RMSE_profile(array0, array1, levels, profiles, angles):
    RMSE = np.zeros((levels, profiles, angles))

    for i in range(levels):
        for j in range(angles):
            for n in range(profiles):
                RMSE[i, n, j] = math.sqrt(
                    np.mean(
                        np.square(
                            array0[:, n, j, i] - array1[:, n, j, i]
                        )
                    )
                )
    return RMSE


#########################################################################
############  Read true transmittances from usr specified txt file ######
#########################################################################


class TransmittanceReader(TransmittanceCalculator):
    """
    [Main code for transmittance reading]

    """

    """
    WARNING: INPUT PARAMETERS MUST BE THE SAME AS TRANSMITTANCE CALCULATOR
    """

    def ReadTrueTrans(self, gas_specified="test_mix_layer"):
        # if not specified, read mix gas trans
        """
        [Read transmittance generate by fortran code TRUE VALUE]

        Args:
            gas_specified ([str]): [true trans file name, like "test_mix_laye"]
            channels ([I8]): [channel number]
            angles ([I8]): [angle number, 8 in total]
            profiles ([I8]): [profile number, 8 in total]

        Returns:
            [f8]: [true transmittance in (angles, profiles, channels, 100) dimension]
        """
        data_true = pd.read_csv(
            self.coef_dir + str(gas_specified) + ".txt",
            delim_whitespace=True,
            header=None,
        )

        data_true = np.array(data_true)
        data_true = data_true.reshape(
            self.profiles, self.channels, self.levels, self.angles
        )
        data_true = data_true.transpose(1, 0, 3, 2)

        return data_true


#########################################################################
############  Generate SRF for specified satellite ######################
#########################################################################


@jit(nopython=True, parallel=True)
def Numba_SpectralSinc(spectral_tri_aux):
    # spectral_aux means 779 channels to deal with
    # spectral_main means 777 real channels
    spectral_aux = np.zeros((779, 625))
    spectral_main = np.zeros((779, 625))
    respons_function = np.zeros((779, 486876))
    for i in range(0, 779):
        spectral_aux[i, :] = np.arange(
            649.375 + (i * 0.625),
            649.375 + ((i + 1) * 0.625),
            0.001,
        )
        spectral_main[i, :] = np.arange(
            650 + (i * 0.625), 650 + ((i + 1) * 0.625), 0.001
        )
        respons_function[i, :] = np.sinc(
            2
            * 0.8
            * (spectral_tri_aux - (649.375 + (i * 0.625) + 0.312))
        )
    return spectral_aux, spectral_main, respons_function


@jit(nopython=True)
def Numba_nomalization(array):
    _range = np.max(array) - np.min(array)
    return (array - np.min(array)) / _range


@jit(nopython=True, parallel=True)
def Numba_SRF(spectral_aux, respons_function_all, hamming_win):
    # sig_spectral is composed of every 3 adjacent channels
    # from every 3 channels we form the final srf
    sig_respons_function_normalized = np.zeros((777, 625))
    sig_respons_function = np.zeros((777, 1875))
    sig_spectral = np.zeros((777, 1875))
    for i in range(0, 777):
        sig_spectral[i, :] = np.hstack(
            (
                spectral_aux[i, :],
                spectral_aux[i + 1, :],
                spectral_aux[i + 2, :],
            )
        )
        sig_respons_function[i, :] = (
            respons_function_all[0 + (i * 625) : 1875 + (i * 625)]
            * hamming_win
        )
        sig_respons_function_normalized[i, :] = Numba_nomalization(
            sig_respons_function[i, 625:1250]
        )
    return (
        sig_spectral,
        sig_respons_function,
        sig_respons_function_normalized,
    )


class SRF_generator:
    def __init__(self, flag):
        self.flag = flag
        self.spectral_tri_aux = np.round(
            np.arange(649.375, 1136.251, 0.001), 3
        )
        self.hamming_win = np.hamming(625 * 3)

    def SRF(self):
        if self.flag == 0:
            (
                spectral_aux,
                spectral_main,
                respons_function,
            ) = Numba_SpectralSinc(self.spectral_tri_aux)
            respons_function_all = np.sum(respons_function, axis=0)
            return Numba_SRF(
                spectral_aux, respons_function_all, self.hamming_win
            )
        elif self.flag == 1:
            path0 = glob.glob("E:\\Project\SRF_CH\RESPONSE_CH*")
            SRF0 = np.loadtxt(path0[0], unpack=True)
            SRF1 = np.loadtxt(path0[1], unpack=True)
            SRF2 = np.loadtxt(path0[2], unpack=True)
            SRF3 = np.loadtxt(path0[3], unpack=True)
            SRF4 = np.loadtxt(path0[4], unpack=True)
            SRF5 = np.loadtxt(path0[5], unpack=True)
            SRF6 = np.loadtxt(path0[6], unpack=True)
            SRF7 = np.loadtxt(path0[7], unpack=True)
            return SRF0, SRF1, SRF2, SRF3, SRF4, SRF5, SRF6, SRF7


#########################################################################
############  Using efffective BT method to correct BT ##################
#########################################################################

# BPlunck is the radiance generated from a group of training temperature
@jit(nopython=True, parallel=True)
def Numba_aux_Planck(C1, C2):
    # generate the B(t)
    training_temperature = np.arange(150, 320, 0.1)
    aux_Planck = np.zeros((777, 625, 1700))
    for t in range(1700):
        for i in range(777):
            wavenumber_start = 650 + (i - 1) * 0.625
            for j in range(625):
                wavenumber = wavenumber_start + (j - 1) * 0.001
                aux_Planck[i, j, t] = (C1 * (wavenumber ** 3)) / (
                    np.exp(
                        C2 * wavenumber / training_temperature[t]
                    )
                    - 1
                )
    return aux_Planck


@jit(nopython=True, parallel=True)
def Numba_IntegrateSRF(SRF_normalized, aux_Planck):
    # Integrate radiance to generate the R(t)
    temp_SRF = np.zeros((777))
    temp_radiance = np.zeros((777, 625, 1700))
    for i in range(777):
        temp_SRF[i] = np.sum(SRF_normalized[i, :])
    for t in range(1700):
        for i in range(777):
            for j in range(625):
                temp_radiance[i, j, t] = (
                    SRF_normalized[i, j] * aux_Planck[i, j, t]
                )
    return temp_SRF, temp_radiance


@jit(nopython=True, parallel=True)
def Numba_effective_Planck(temp_SRF, temp_radiance):
    # Integrate radiance to generate the R(t)
    effective_Planck = np.zeros((777, 1700))
    for t in range(1700):
        for i in range(777):
            effective_Planck[i, t] = (
                temp_radiance[i, t] / temp_SRF[i]
            )
    return effective_Planck


@jit(nopython=True, parallel=True)
def Numba_inversPlanck_true(C1, C2, effective_Planck):
    brightness_temperature_true = np.zeros((777, 1700))
    for i in range(777):
        for t in range(1700):
            wavenumber = (i + 1) * 0.625 + 650
            central_wavenumber = wavenumber - (0.625 / 2)
            brightness_temperature_true[i, t] = (
                C2 * central_wavenumber
            ) / np.log(
                C1
                * (central_wavenumber ** 3)
                / effective_Planck[i, t]
                + 1
            )
    return brightness_temperature_true


class effective_BT:
    def __init__(
        self,
        SRF_normalized,
        training_temperature=np.arange(150, 320, 0.1),
        Planck_c1=1.191042972 * 10 ** -5,
        Planck_c2=1.438776878,
    ):
        self.SRF_normalized = SRF_normalized
        self.training_temperature = training_temperature
        # Plunck constant
        self.Planck_c1 = Planck_c1
        self.Planck_c2 = Planck_c2

    def preprocessing_planck_LinerFit(self):
        z1 = np.zeros((777, 2))
        aux_Planck = Numba_aux_Planck(
            self.Planck_c1, self.Planck_c2
        )
        temp_SRF, temp_radiance = Numba_IntegrateSRF(
            self.SRF_normalized, aux_Planck
        )
        temp_radiance = np.nansum(temp_radiance, axis=1)
        effective_Planck = Numba_effective_Planck(
            temp_SRF, temp_radiance
        )
        brightness_temperature_true = Numba_inversPlanck_true(
            self.Planck_c1, self.Planck_c2, effective_Planck,
        )
        for i in range(777):
            z1[i, :] = np.polyfit(
                self.training_temperature[:],
                brightness_temperature_true[i, :],
                1,
            )
        return z1


############################################################################
### Preproceessing the trans data and calc each level planck radiance ######
############################################################################


class preprocess_radiance(TransmittanceCalculator):
    def __init__(
        self,
        transmittance,
        fits_BT,
        coef_specified,
        pred_specified,
        channels,
        angles,
        profiles,
        levels=100,
        flag=0,
        prof_dir="/RAID01/data/muqy/Project_Radiation/",
        coef_dir="/RAID01/data/muqy/Project_Radiation/",
        pred_dir="/RAID01/data/muqy/Project_Radiation/",
        Planck_c1=1.191042972 * 10 ** -5,
        Planck_c2=1.438776878,
    ):
        super().__init__(
            coef_specified,
            pred_specified,
            channels,
            angles,
            profiles,
            levels,
            coef_dir,
            pred_dir,
        )
        self.transmittance = transmittance[:, :, :, ::-1]
        self.flag = flag
        self.fits_BT = fits_BT
        self.Planck_c1 = Planck_c1
        self.Planck_c2 = Planck_c2
        self.prof_dir = prof_dir

    def prof_reader(self):
        """
        read in profile data to calc planck

        Parameters
        ----------
        flag : int, optional
            _description_, by default 0 means calc ec-83
            1 means calc tg43

        Returns
        -------
            profile parameters 
        """
        if self.flag == 0:
            prof_name = "ECMWF_83P_101L_py.dat"
            ec83 = np.loadtxt(
                self.prof_dir + prof_name, unpack=True,
            )
            ec83 = ec83.reshape(8, self.profiles, self.levels + 1)
            ec83_t = ec83.transpose((1, 0, 2))
            ec83_t = np.delete(ec83_t, self.levels, axis=2)
            return ec83_t
        elif self.flag == 1:
            prof_name = "tigr43_43lev_prof.dat"
            tg43 = np.loadtxt(
                self.prof_dir + prof_name, unpack=True,
            )
            tg43 = tg43.reshape(3, 43, self.levels)
            tg43_t = tg43.transpose((1, 0, 2))
            # tg43_t = np.delete(tg43_t, 4, axis=0)
            tg43_t = tg43_t[:, :, ::-1]
            return tg43_t

    def Planck(self):

        prof_t = self.prof_reader()

        if self.flag == 0:
            return _Numba_Planck0(
                prof_t,
                self.fits_BT,
                self.Planck_c1,
                self.Planck_c2,
                self.profiles,
                self.channels,
                self.levels,
            )
        elif self.flag == 1:
            import glob

            path0 = glob.glob("F:\\Project\SRF_CH\RESPONSE_CH*")

            SRF0 = np.loadtxt(path0[0], unpack=True)
            SRF1 = np.loadtxt(path0[1], unpack=True)
            SRF2 = np.loadtxt(path0[2], unpack=True)
            SRF3 = np.loadtxt(path0[3], unpack=True)
            SRF4 = np.loadtxt(path0[4], unpack=True)
            SRF5 = np.loadtxt(path0[5], unpack=True)
            SRF6 = np.loadtxt(path0[6], unpack=True)
            SRF7 = np.loadtxt(path0[7], unpack=True)

            SRF_start_end = np.zeros((2, 8))
            SRF_start_end[1, 0] = SRF0[0, -1]
            SRF_start_end[1, 1] = SRF1[0, -1]
            SRF_start_end[1, 2] = SRF2[0, -1]
            SRF_start_end[1, 3] = SRF3[0, -1]
            SRF_start_end[1, 4] = SRF4[0, -1]
            SRF_start_end[1, 5] = SRF5[0, -1]
            SRF_start_end[1, 6] = SRF6[0, -1]
            SRF_start_end[1, 7] = SRF7[0, -1]

            SRF_start_end[0, 0] = SRF0[0, 0]
            SRF_start_end[0, 1] = SRF1[0, 0]
            SRF_start_end[0, 2] = SRF2[0, 0]
            SRF_start_end[0, 3] = SRF3[0, 0]
            SRF_start_end[0, 4] = SRF4[0, 0]
            SRF_start_end[0, 5] = SRF5[0, 0]
            SRF_start_end[0, 6] = SRF6[0, 0]
            SRF_start_end[0, 7] = SRF7[0, 0]

            return _Numba_Planck1(
                SRF_start_end,
                prof_t,
                self.fits_BT,
                self.Planck_c1,
                self.Planck_c2,
                self.profiles,
                self.channels,
                self.levels,
            )

    def surf_radiance(self):
        Planck = self.Planck()
        return _SurfRadiance(
            Planck,
            self.transmittance,
            self.channels,
            self.profiles,
            self.levels,
            self.angles,
        )

    def atmos_radiance(self):
        Planck = self.Planck()
        return np.nansum(
            _AtmosRadianceEachLevel(
                Planck,
                self.transmittance,
                self.channels,
                self.profiles,
                self.levels,
                self.angles,
            ),
            axis=3,
        )

    def total_radiance(self):
        atmos_radiance = self.atmos_radiance()
        surf_radiance = self.surf_radiance()
        return _TotalRadiance(
            atmos_radiance,
            surf_radiance,
            self.channels,
            self.profiles,
            self.angles,
            surface_emit=0.98,
        )

    def BT_calculator(self):
        total_radiance = self.total_radiance()
        if self.flag == 0:
            BrightnessTemperature = _InversPlanck0(
                total_radiance,
                self.Planck_c1,
                self.Planck_c2,
                self.fits_BT,
                self.channels,
                self.profiles,
                self.angles,
            )
            return BrightnessTemperature
        elif self.flag == 1:

            import glob

            path0 = glob.glob("F:\\Project\SRF_CH\RESPONSE_CH*")

            SRF0 = np.loadtxt(path0[0], unpack=True)
            SRF1 = np.loadtxt(path0[1], unpack=True)
            SRF2 = np.loadtxt(path0[2], unpack=True)
            SRF3 = np.loadtxt(path0[3], unpack=True)
            SRF4 = np.loadtxt(path0[4], unpack=True)
            SRF5 = np.loadtxt(path0[5], unpack=True)
            SRF6 = np.loadtxt(path0[6], unpack=True)
            SRF7 = np.loadtxt(path0[7], unpack=True)

            SRF_start_end = np.zeros((2, 8))
            SRF_start_end[1, 0] = SRF0[0, -1]
            SRF_start_end[1, 1] = SRF1[0, -1]
            SRF_start_end[1, 2] = SRF2[0, -1]
            SRF_start_end[1, 3] = SRF3[0, -1]
            SRF_start_end[1, 4] = SRF4[0, -1]
            SRF_start_end[1, 5] = SRF5[0, -1]
            SRF_start_end[1, 6] = SRF6[0, -1]
            SRF_start_end[1, 7] = SRF7[0, -1]

            SRF_start_end[0, 0] = SRF0[0, 0]
            SRF_start_end[0, 1] = SRF1[0, 0]
            SRF_start_end[0, 2] = SRF2[0, 0]
            SRF_start_end[0, 3] = SRF3[0, 0]
            SRF_start_end[0, 4] = SRF4[0, 0]
            SRF_start_end[0, 5] = SRF5[0, 0]
            SRF_start_end[0, 6] = SRF6[0, 0]
            SRF_start_end[0, 7] = SRF7[0, 0]

            BrightnessTemperature = _InversPlanck1(
                SRF_start_end,
                total_radiance,
                self.Planck_c1,
                self.Planck_c2,
                self.fits_BT,
                self.channels,
                self.profiles,
                self.angles,
            )
            return BrightnessTemperature


@jit(nopython=True, parallel=True)
def _SurfRadiance(
    Planck, transmittance, channels, profiles, levels, angles
):
    surf_radiance = np.zeros((channels, profiles, angles))
    for i in range(channels):
        for j in range(profiles):
            for k in range(angles):
                surf_radiance[i, j, k] = (
                    # transmittance[i, j, k, levels - 1]
                    # * Planck[j, i, levels - 1]
                    transmittance[i, j, k, 0]
                    * Planck[j, i, 0]
                )
    return surf_radiance


@jit(nopython=True, parallel=True)
def _AtmosRadianceEachLevel(
    Planck, transmittance, channels, profiles, levels, angles,
):
    atmos_radiance_lev = np.zeros(
        (channels, profiles, angles, levels - 1)
    )
    for i in range(channels):
        for j in range(profiles):
            for k in range(angles):
                for n in range(levels):
                    # term 1
                    atmos_radiance_lev[i, j, k, n - 1] = Planck[
                        j, i, n - 1
                    ] * (
                        transmittance[i, j, k, n]
                        - transmittance[i, j, k, n - 1]
                    )

    return atmos_radiance_lev


@jit(nopython=True, parallel=True)
def _TotalRadiance(
    atmos_radiance,
    surf_radiance,
    channels,
    profiles,
    angles,
    surface_emit=0.98,
):
    # total radiance = surfemit*surf + atmos
    surface_emit = surface_emit
    total_radiance = np.zeros((channels, profiles, angles))
    for i in range(channels):
        for j in range(profiles):
            for k in range(angles):
                total_radiance[i, j, k] = (
                    surface_emit * surf_radiance[i, j, k]
                    + atmos_radiance[i, j, k]
                )
    return total_radiance


@jit(nopython=True)
def Numba_log(x):
    return np.log(x)


@jit(nopython=True)
def _InversPlanck0(
    total_radiance,
    Planck_c1,
    Planck_c2,
    fits_BT,
    channels,
    profiles,
    angles,
):
    brightness_temperature = np.zeros((channels, profiles, angles))
    for i in range(channels):
        for j in range(profiles):
            for k in range(angles):
                central_wavenumber = (
                    (i + 1) * 0.625 + 650 - (0.625 / 2)
                )
                brightness_temperature[i, j, k] = (
                    (
                        (Planck_c2 * central_wavenumber)
                        / Numba_log(
                            Planck_c1
                            * (central_wavenumber ** 3)
                            / total_radiance[i, j, k]
                            + 1
                        )
                    )
                    - fits_BT[i, 1]
                ) / fits_BT[i, 0]
    return brightness_temperature


@jit(nopython=True)
def _InversPlanck1(
    SRF_start_end,
    total_radiance,
    Planck_c1,
    Planck_c2,
    fits_BT,
    channels,
    profiles,
    angles,
):
    brightness_temperature = np.zeros((channels, profiles, angles))

    for i in range(channels):
        central_wavenumber = (
            SRF_start_end[1, i] - SRF_start_end[0, i]
        ) / 2 + SRF_start_end[0, i]
        for j in range(profiles):
            for k in range(angles):
                brightness_temperature[i, j, k] = (
                    (
                        (Planck_c2 * central_wavenumber)
                        / Numba_log(
                            Planck_c1
                            * (central_wavenumber ** 3)
                            / total_radiance[i, j, k]
                            + 1
                        )
                    )
                    - fits_BT[i, 1]
                ) / fits_BT[i, 0]
    return brightness_temperature


@jit(nopython=True, parallel=True)
def _Numba_Planck0(
    prof_t, fits_BT, C1, C2, profiles, channels, levels
):
    # lev means atmos level, 0 is top
    # inverse the array in order to fit the atmos level
    Planck = np.zeros((profiles, channels, levels))
    prof_t_fit = np.zeros((profiles, channels, levels))
    for i in range(profiles):
        for j in range(channels):
            prof_t_fit[i, j, :] = (
                prof_t[i, 1, :] * fits_BT[j, 0] + fits_BT[j, 1]
            )
            for k in range(levels):
                central_wavenumber = (
                    (j + 1) * 0.625 + 650 - (0.625 / 2)
                )
                Planck[i, j, k] = (
                    C1 * (central_wavenumber ** 3)
                ) / (
                    np.exp(
                        C2
                        * central_wavenumber
                        / prof_t_fit[i, j, k]
                    )
                    - 1
                )
    return Planck


@jit(nopython=True, parallel=True)
def _Numba_Planck1(
    SRF_start_end,
    prof_t,
    fits_BT,
    C1,
    C2,
    profiles,
    channels,
    levels,
):
    # lev means atmos level ,0 is top
    # inverse the array in order to fit the atmos level
    Planck = np.zeros((profiles, channels, levels))
    prof_t_fit = np.zeros((profiles, channels, levels))
    for i in range(profiles):
        for j in range(channels):
            prof_t_fit[i, j, :] = (
                prof_t[i, 1, :] * fits_BT[j, 0] + fits_BT[j, 1]
            )
            central_wavenumber = (
                SRF_start_end[1, j] - SRF_start_end[0, j]
            ) / 2 + SRF_start_end[0, j]
            for k in range(levels):
                Planck[i, j, k] = (
                    C1 * (central_wavenumber ** 3)
                ) / (
                    np.exp(
                        C2
                        * central_wavenumber
                        / prof_t_fit[i, j, k]
                    )
                    - 1
                )
    return Planck


#########################################################################
##########  plot function for RMSE by wavenumbers/profiles        #######
#########################################################################


class RMSEPlot(TransmittanceCalculator):
    def plotRMSE_wave(self, array, spec_angels):

        wave = np.arange(650.625, 1136, 0.625)
        fig = plt.figure(figsize=(10, 5))
        ax1 = plt.subplot(111)
        # if self.levels == 100:
        #     pres = pressure["press100"]
        # elif self.levels == 43 or self.levels == 42:
        #     pres = pressure["press43"]
        a = ax1.imshow(array0[:, :, angel], aspect="auto")
        # plt.title("RMSE for channel"+str(channel+1), fontsize=18)
        ax1.set_xlabel("wavenumber(cm\u207b\u00b9)", fontsize=14)
        ax1.set_ylabel("Pressure(log)", fontsize=14)
        fig.colorbar(a)
        ax1.set_yscale("symlog")
        # ax1.invert_yaxis()
        ax1.set_yticks(
            np.array((1070.92, 600, 400, 200, 100, 0.005))
        )
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%1.2f")
        )

    def plotRMSE_profile(self, array, spec_angels):

        fig = plt.figure(figsize=(15, 9))
        ax1 = plt.subplot(111)
        cmap = plt.get_cmap("binary")
        cmap.set_over("red")
        if self.levels == 100:
            pres = pressure["press100"]
        elif self.levels == 43 or self.levels == 42:
            pres = pressure["press43"]
        a = ax1.pcolormesh(
            np.linspace(1, self.profiles, self.profiles),
            pres[:],
            array[:, :, spec_angels],
            # vmax=vmax,
            cmap=cmap,
        )
        # plt.title("RMSE for channel"+str(channel+1), fontsize=18)
        ax1.set_xlabel("Profile", fontsize=14)
        ax1.set_ylabel("Pressure(log)", fontsize=14)
        fig.colorbar(a)
        # ax1.set_yscale('symlog')
        ax1.invert_yaxis()
        ax1.set_yticks(
            np.array((1070.92, 600, 400, 200, 100, 0.005))
        )
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )
        ax1.yaxis.set_major_formatter(
            matplotlib.ticker.FormatStrFormatter("%1.2f")
        )

