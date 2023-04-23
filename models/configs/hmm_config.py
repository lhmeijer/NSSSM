# -*- coding: utf-8 -*-

from models.configs.config import BaseConfig
import models.hmm as hmm
# from models.hmm import HMM, MultinomialLogisticInputHMM, MultinomialLogisticInputOnlyHMM, \
#     StickBreakingLogisticInputHMM, StickBreakingLogisticInputOnlyHMM, \
#     RegularizedMultinomialLogisticInputHMM, RegularizedMultinomialLogisticInputOnlyHMM
import posixpath
from models.distributions.gaussian import ScalarGaussianNIX, ScalarGaussianFixedMean, GaussianNIW, IsotropicGaussian
from models.distributions.regression import RegressionFixedCoefficients, \
    DiagonalRegressionFixedCoefficients, RegressionNonConj
from preprocessors.reader import read_household_group
import numpy as np
from models.sssm import SwitchingSSM, MultinomialLogisticInputSwitchingSSM
from configuration import Config


class HMMK6Config(BaseConfig):
    
    # MCMC_ITER = 1
    # BURN_IN_ITER = 0
    # SAVE_ITER = 1
    
    MCMC_ITER = 1550
    BURN_IN_ITER = 1300
    SAVE_ITER = 1
    
    # model = hmm.MultinomialLogisticInputHMM
    # model = hmm.RegularizedMultinomialLogisticInputHMM
    model = hmm.MultinomialLogisticInputOnlyHMM
    # model = hmm.RegularizedMultinomialLogisticInputOnlyHMM
    # model = hmm.StickBreakingLogisticInputOnlyHMM
    # model = hmm.RegularizedStickBreakingInputOnlyHMM
    
    num_states = 6 
    
    name = 'HMM_K6'
    
    @property
    def params_input_name(self):
        params_input_name = 'hmm_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    
    
    mu_A = 0.0
    sigma_A = 0.1
    
    @property
    def obs_distns(self):
        obs_distns = [RegressionNonConj(A_0=np.array([[np.log(14948), 0.0]]), sigma_0=self.sigma_0, S_0=self.S_0, nu_0=self.nu_0), 
                      RegressionNonConj(A_0=np.array([[np.log(20758), 0.0]]), sigma_0=self.sigma_0, S_0=self.S_0, nu_0=self.nu_0),
                      RegressionNonConj(A_0=np.array([[np.log(31440), 0.0]]), sigma_0=self.sigma_0, S_0=self.S_0, nu_0=self.nu_0),
                      RegressionNonConj(A_0=np.array([[np.log(44609), 0.0]]), sigma_0=self.sigma_0, S_0=self.S_0, nu_0=self.nu_0),
                      RegressionNonConj(A_0=np.array([[np.log(60830), 0.0]]), sigma_0=self.sigma_0, S_0=self.S_0, nu_0=self.nu_0),
                      RegressionNonConj(A_0=np.array([[np.log(75479), 0.0]]), sigma_0=self.sigma_0, S_0=self.S_0, nu_0=self.nu_0)]
        return obs_distns
    
    sigma_0 = np.array([[1.0, 0.0], [0.0, 0.01]])
    S_0 = np.array([[10]]) # 1.5
    nu_0 = 10
    
    def get_input_data(self, T, N):
        input_data = np.ones((T, N, 2), dtype=np.int16)
        input_data[:, :, 1] = [np.full(N, t) for t in range(T)]
        return input_data
    
    @property
    def init_params(self):
        transition_params = {'mu_A': self.mu_A, 'sigma_A': self.sigma_A}
        init_params = {'obs_distns': self.obs_distns, 'D_in': 25, 'transition_params': transition_params}
        return init_params
    

class HMMK6RegularizedConfig(HMMK6Config):
    
    name = 'HMM_K6_regularized'
    model = hmm.RegularizedMultinomialLogisticInputOnlyHMM
    
    @property
    def params_input_name(self):
        params_input_name = 'hmm_regularized_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name