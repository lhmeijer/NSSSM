# -*- coding: utf-8 -*-
from models.configs.config import BaseConfig
from models.mixture import Mixture
import posixpath
from models.distributions.gaussian import ScalarGaussianNIX, ScalarGaussianFixedMean, GaussianNIW, IsotropicGaussian, DiagonalGaussianNonConjNIG
from models.distributions.regression import RegressionFixedCoefficients, \
    DiagonalRegressionFixedCoefficients, RegressionNonConj
from preprocessors.reader import read_household_group
import numpy as np
from models.sssm import SwitchingSSM, MultinomialLogisticInputSwitchingSSM
from configuration import Config


class BaselineConfig(BaseConfig):

    model = SwitchingSSM

    MCMC_ITER = 200
    BURN_IN_ITER = 20
    SAVE_ITER = 1
    
    # MCMC_ITER = 1
    # BURN_IN_ITER = 1
    # SAVE_ITER = 1
    
    A_emission = np.array([[1.0, 0.0]])
    
    mu_0_init = np.array([np.log(37979), 0])
    
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    
    sigmas_0_init = np.array([1, 0.1])
    
    alpha_0 = 10
    alpha = 10
    beta = np.array([10, 10]) 
    beta_0 = np.array([10, 10])
    beta_emission = np.array([10])
    
    @property
    def init_dynamics_distns(self):
        init_dynamics_distns = [DiagonalGaussianNonConjNIG(
            mu_0=self.mu_0_init, sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0) for _ in range(self.num_states)]
        return init_dynamics_distns
    
    @property
    def init_params(self):
        emission_distns = DiagonalRegressionFixedCoefficients(A=self.A_emission, alpha_0=self.alpha, beta_0=self.beta_emission)
        dynamics_distns = [DiagonalRegressionFixedCoefficients(A=self.A, alpha_0=self.alpha, beta_0=self.beta) for _ in range(self.num_states)]
        init_params = {
            'emission_distns': emission_distns, 
            'dynamics_distns': dynamics_distns, 
            'init_dynamics_distns': self.init_dynamics_distns,
            'alpha': 10,
        }
        return init_params


class BaselineOneClusterConfig(BaselineConfig):
    
    MCMC_ITER = 79
    BURN_IN_ITER = 10
    
    name = 'baseline_1_fixed_cluster'
    
    num_states = 1
    params_input_name = 'baseline_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, num_states)

    def fixed_state_seqs(self, CBS=False, time_idxs=(0, 10), 
                         unit_idxs=(0, 100000), masks=None):
        T = time_idxs[1] - time_idxs[0] 
        N = unit_idxs[1] - unit_idxs[0]
        if masks is None:
            masks = np.ones((T, N), dtype=bool)
        seqs = np.zeros((T, N), dtype=np.int32)
        seqs[~masks] = -1
        return seqs, True
    
    
class BaselineFixedClustersConfig(BaselineConfig):
    
    MCMC_ITER = 135
    BURN_IN_ITER = 85
    
    
    name = 'baseline_6_fixed_clusters'
    num_states = 6
    # ITERATION = 79
    params_input_name = 'baseline_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, num_states)
    
    def fixed_state_seqs(self, CBS=False, time_idxs=(0, 10), 
                         unit_idxs=(0, 100000), masks=None):
        if CBS:
            file_name = Config.household_group_cbs_file
        else:
            file_name = Config.household_group_forecast_file
        seqs = read_household_group(file_name)
        return seqs[time_idxs[0]:time_idxs[1], unit_idxs[0]:unit_idxs[1]], True
    

    
    