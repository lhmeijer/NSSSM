# -*- coding: utf-8 -*-
import posixpath
import os
import json
import numpy as np
import sys
from models.hmm import MultinomialLogisticInputOnlyHMM, RegularizedMultinomialLogisticInputOnlyHMM
from models.distributions.gaussian import ScalarGaussianNIX
from models.mixture import Mixture
from models.distributions.multinomial import Categorical
from configuration import Config
from models.distributions.gaussian import ScalarGaussianNIX, ScalarGaussianFixedMean, GaussianNIW, IsotropicGaussian, DiagonalGaussianNonConjNIG
from models.distributions.regression import RegressionFixedCoefficients, \
    DiagonalRegressionFixedCoefficients, RegressionNonConj
from models.distributions.polya_gamma import MultinomialLogisticRegression, RegularizedMultinomialLogisticRegression, RegularizedStickBreakingLogisticRegression, StickBreakingLogisticRegression
from models.sssm import MultinomialLogisticInputOnlySwitchingSSM, SwitchingSSM, \
    RegularizedMultinomialLogisticInputOnlySwitchingSSM, MultinomialLogisticInputSwitchingSSM, \
        RegularizedMultinomialLogisticInputSwitchingSSM, RegularizedStickBreakingLogisticInputOnlySwitchingSSM, \
            StickBreakingLogisticInputOnlySwitchingSSM, TruncatedStickBreakingLogisticInputOnlySwitchingSSM, \
                RegularizedTruncatedStickBreakingLogisticInputOnlySwitchingSSM
from models.transitions.hmm_transitions import HMMInputOnlyTransitions, HMMTransitions, RegularizedHMMInputOnlyTransitions, HMMInputTransitions, RegularizedHMMInputTransitions, RegularizedStickBreakingHMMInputOnlyTransitions, StickBreakingHMMInputOnlyTransitions


class BaseConfig(object):
    
    params_input_name = '{0}_{1}'.format(Config.N_HOUSEHOLDS, Config.SEED)
    
    
    MCMC_ITER = 100
    BURN_IN_ITER = 500
    SAVE_ITER = 100
    
    ITERATION = None
    
    
    # MEAN_GROWTH = [1.432, 1.452, 1.431, 1.443, 1.166, 1.223]
    MEAN_GROWTH = [1.0332, 1.0344, 1.0331, 1.0339, 1.0141, 1.0185]
    
    def get_input_data(self, T, N):
        input_data = np.zeros((T, N, 0), dtype=np.int16)
        return input_data

    def fixed_state_seqs(self, CBS=False, time_idxs=(0, 10), 
                         unit_idxs=(0, 100000), masks=None):
        return None, False
    
    def get_params_input_file(self, iteration):
        params_input_directory = posixpath.join('H:', 'Lisa', 'data', 'params2', self.params_input_name, '')
        return '{0}{1}.json'.format(params_input_directory, iteration)
    
    def get_results_file(self, file_name, general=False):
        if general: 
            params_input_directory = posixpath.join('H:', 'Lisa', 'data', 'results', '')
        else:
            params_input_directory = posixpath.join('H:', 'Lisa', 'data', 'results', self.params_input_name, '')
        return '{0}{1}'.format(params_input_directory, file_name)
        
    def extract_model_from_params(self, dict_params):
        class_name = dict_params['name']
        params = {}
        for name, param in dict_params['params'].items():
            if isinstance(param, dict):
                params[name] = self.extract_model_from_params(param)
            elif isinstance(param, list):
                params[name] = np.array([self.extract_model_from_params(elem) if isinstance(elem, dict) else np.array(elem) for elem in param])
            else:
                params[name] = param
        new_class = getattr(sys.modules[__name__], class_name)
        return new_class(**params)
        
    @property
    def fitted_params(self):
        
        # if not os.path.isfile(self.get_params_input_file(0)):
        #     return []
        
        models = []
        for i in range(self.BURN_IN_ITER, self.MCMC_ITER):
        # for i in range(100, 101):
            
            input_file = self.get_params_input_file(i)
            with open(input_file, 'r') as file:
                params = json.load(file)
            
            model = self.extract_model_from_params(params)
            models.append(model)
        return models
            
    