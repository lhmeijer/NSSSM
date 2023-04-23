# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:22:42 2023

@author: gst7844lmjr
"""
from models.configs.config import BaseConfig
import models.sssm as sssm
import posixpath
from models.distributions.gaussian import ScalarGaussianNIX, ScalarGaussianFixedMean, GaussianNIW, IsotropicGaussian, DiagonalGaussianNonConjNIG
from models.distributions.regression import RegressionFixedCoefficients, \
    DiagonalRegressionFixedCoefficients, RegressionNonConj
import numpy as np
from configuration import Config


class SSSMConfig(BaseConfig):
    

    model = sssm.RegularizedMultinomialLogisticInputOnlySwitchingSSM

    num_states = 10
    params_input_name = 'sssm_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, num_states)
    
    # MCMC_ITER = 500
    # BURN_IN_ITER = 250
    # SAVE_ITER = 1
    
    MCMC_ITER = 2000
    BURN_IN_ITER = 50
    SAVE_ITER = 1
    
    A_emission = np.array([[1.0, 0.0]])
    mu_0_init = np.array([np.log(37979), 0.0])
    
    A = np.array([[1.0, 1.0], [0.0, 1.0]])

    sigmas_0_init = np.array([1., 0.01])
    
    alpha_0 = 10
    alpha = 10
    beta = np.array([10, 10]) 
    beta_0 = np.array([10, 10])
    beta_emission = np.array([10])
    
    mu_A = 0
    sigma_A = 0.1
    
    
    @property
    def init_dynamics_distns(self):
        init_dynamics_distns = [DiagonalGaussianNonConjNIG(
            mu_0=self.mu_0_init, sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0) for _ in range(self.num_states)]
        return init_dynamics_distns
    
    @property
    def init_params(self):
        emission_distns = DiagonalRegressionFixedCoefficients(A=self.A_emission, alpha_0=self.alpha, beta_0=self.beta_emission)
        dynamics_distns = [DiagonalRegressionFixedCoefficients(A=self.A, alpha_0=self.alpha, beta_0=self.beta) for _ in range(self.num_states)]
        transition_params = {'mu_A': self.mu_A, 'sigma_A': self.sigma_A}
        init_params = {
            'emission_distns': emission_distns, 
            'dynamics_distns': dynamics_distns, 
            'init_dynamics_distns': self.init_dynamics_distns,
            'D_in': 25,
            'transition_params': transition_params
        }
        return init_params
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    
class SSSMK4Config(SSSMConfig):
    
    name = 'SSSM_K4'
    
    model = sssm.MultinomialLogisticInputOnlySwitchingSSM
    num_states = 4
    
    ITERATION = 1949
    
    @property
    def init_dynamics_distns(self):
        init_dynamics_distns = [DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(14948), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha, beta_0=self.beta), 
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(31440), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha, beta_0=self.beta),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(44609), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha, beta_0=self.beta),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(75479), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha, beta_0=self.beta)]
        return init_dynamics_distns
    
class SSSMK4TransitionConfig(SSSMK4Config):
    
    name = 'SSSM_K4_transition'
    
    model = sssm.MultinomialLogisticInputSwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_transition_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name    


class SSSMK4RegularizedConfig(SSSMK4Config):
    
    name = 'SSSM_K4_regularized'
    
    ITERATION = 1949
    
    model = sssm.RegularizedMultinomialLogisticInputOnlySwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    

class SSSMK4TransitionRegularizedConfig(SSSMK4Config):
    
    name = 'SSSM_K4_transition_regularized'
    
    model = sssm.RegularizedMultinomialLogisticInputSwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_transition_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    

class SSSMK6Config(SSSMConfig):
    
    name = 'SSSM_K6'
    
    model = sssm.MultinomialLogisticInputOnlySwitchingSSM
    num_states = 6
    
    ITERATION = 1999
    MCMC_ITER = 500
    BURN_IN_ITER = 250
    
    @property
    def init_dynamics_distns(self):
        init_dynamics_distns = [DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(14948), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0), 
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(20758), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(31440), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(44609), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(60830), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(75479), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0)]
        return init_dynamics_distns
    
    
class SSSMK6TransitionConfig(SSSMK6Config):
    
    name = 'SSSM_K6_transition'
    
    ITERATION = None
    MCMC_ITER = 500
    BURN_IN_ITER = 250
    
    model = sssm.MultinomialLogisticInputSwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_transition_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name    


class SSSMK6RegularizedConfig(SSSMK6Config):
    
    name = 'SSSM_K6_regularized'
    
    model = sssm.RegularizedMultinomialLogisticInputOnlySwitchingSSM
    ITERATION = 1999
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    

class SSSMK6TransitionRegularizedConfig(SSSMK6Config):
    
    name = 'SSSM_K6_transition_regularized'
    
    ITERATION = 949
    
    model = sssm.RegularizedMultinomialLogisticInputSwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_transition_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name

    
class SSSMK8Config(SSSMConfig):
    
    name = 'SSSM_K8'
    
    model = sssm.MultinomialLogisticInputOnlySwitchingSSM
    num_states = 8
    
    ITERATION = 1999
    
    @property
    def init_dynamics_distns(self):
        init_dynamics_distns = [DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(14948), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0), 
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(20758), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(25828), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(31440), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(44609), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(51718), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(60830), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(75479), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0)]
        return init_dynamics_distns
    
    

class SSSMK8RegularizedConfig(SSSMK8Config):
    
    name = 'SSSM_K8_regularized'
    
    model = sssm.RegularizedMultinomialLogisticInputOnlySwitchingSSM
    
    ITERATION = 1999
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    

class SSSMK8TransitionConfig(SSSMK8Config):
    
    name = 'SSSM_K8_transition'
    
    model = sssm.MultinomialLogisticInputSwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_transition_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name  
    
    
class SSSMK8TransitionRegularizedConfig(SSSMK8Config):
    
    name = 'SSSM_K8_transition_regularized'
    
    model = sssm.RegularizedMultinomialLogisticInputSwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_transition_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    
    
class SSSMK10Config(SSSMConfig):
    
    model = sssm.MultinomialLogisticInputOnlySwitchingSSM
    num_states = 10
    
    ITERATION = 1999
    MCMC_ITER = 1550
    BURN_IN_ITER = 1300
    
    name = 'SSSM_K10'
    
    @property
    def init_dynamics_distns(self):
        init_dynamics_distns = [DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(14948), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0), 
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(20758), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(23000), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(25828), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(31440), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(44609), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(51718), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(61027), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(75479), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0),
                                DiagonalGaussianNonConjNIG(mu_0=np.array([np.log(1023402), 0.0]), sigmas_0=self.sigmas_0_init, alpha_0=self.alpha_0, beta_0=self.beta_0)
                                ]
        return init_dynamics_distns
    
    

class SSSMK10RegularizedConfig(SSSMK10Config):
    
    name = 'SSSM_K10_regularized'
    ITERATION = 1234
    
    MCMC_ITER = 1550
    BURN_IN_ITER = 1300
    
    model = sssm.RegularizedMultinomialLogisticInputOnlySwitchingSSM
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    
    
class SSSMTruncatedStickBreakingRegularizedConfig(SSSMConfig):
    
    name = 'SSSM_truncated_stickbreaking_regularized'
    @property
    def params_input_name(self):
        params_input_name = 'sssm_truncated_regularized_stickbreaking_H{0}_S{1}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED)
        return params_input_name
    
    model = sssm.RegularizedTruncatedStickBreakingLogisticInputOnlySwitchingSSM
    
    num_states = 20
    
    
class SSSMTruncatedStickBreakingConfig(SSSMConfig):
    
    ITERATION = 19
    
    name = 'SSSM_truncated_stickbreaking'
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_truncated_stickbreaking_H{0}_S{1}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED)
        return params_input_name
    
    model = sssm.TruncatedStickBreakingLogisticInputOnlySwitchingSSM
    
    num_states = 20
    
class SSSMTruncatedK10StickBreakingConfig(SSSMConfig):
    
    ITERATION = 1499
    
    MCMC_ITER = 150
    BURN_IN_ITER = 100
    
    name = 'SSSM_truncated_K10_stickbreaking'
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_truncated_stickbreaking_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    
    model = sssm.TruncatedStickBreakingLogisticInputOnlySwitchingSSM
    
    num_states = 10
    
class SSSMTruncatedK10StickBreakingRegularizedConfig(SSSMConfig):
    
    name = 'SSSM_truncated_K10_regularized_stickbreaking'
    
    MCMC_ITER = 150
    BURN_IN_ITER = 100
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_truncated_regularized_stickbreaking_H{0}_S{1}_K{2}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED, self.num_states)
        return params_input_name
    
    model = sssm.RegularizedTruncatedStickBreakingLogisticInputOnlySwitchingSSM
    
    num_states = 10
    

class SSSMStickBreakingConfig(SSSMConfig):
    
    name = 'SSSM_stickbreaking'
    MCMC_ITER = 1312
    BURN_IN_ITER = 1062
    
    @property
    def params_input_name(self):
        params_input_name = 'sssm_stickbreaking_H{0}_S{1}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED)
        return params_input_name

    model = sssm.StickBreakingLogisticInputOnlySwitchingSSM
    
    num_states = 20
    

class SSSMStickBreakingRegularizedConfig(SSSMConfig):
    
    name = 'SSSM_stickbreaking_regularized'

    @property
    def params_input_name(self):
        params_input_name = 'sssm_regularized_stickbreaking_H{0}_S{1}'.format(Config.N_HOUSEHOLDS_TRAIN, Config.SEED)
        return params_input_name

    model = sssm.RegularizedStickBreakingLogisticInputOnlySwitchingSSM
    
    num_states = 20
    

    

    