# -*- coding: utf-8 -*-
from models.states.base_states import BaseStates, BaseInputStates
import numpy as np

class Base(object):
    
    _states_class = BaseStates
    
    def __init__(self, obs_distns, trans_distn=None):
        self.obs_distns = obs_distns
        self.trans_distn = trans_distn
        self.states_list = []
        
    @property
    def num_states(self):
        return len(self.obs_distns)
    
    @property
    def num_active_states(self):
        num_active_states = max([s.num_active_states for s in self.states_list])
        return num_active_states
    
    def add_data(self, data, input_data=None, masks=None, state_seqs=None, 
                 fixed_state_seqs=False, covariates=None, initialise_from_prior=False):
        self.states_list.append(self._states_class(model=self, data=data, input_data=input_data, masks=masks,
                                                   state_seqs=state_seqs, fixed_state_seqs=fixed_state_seqs,
                                                   initialise_from_prior=initialise_from_prior))
        return self.states_list[-1]
    
    def generate(self, T, N, input_data=None, keep=True):
        s = self._states_class(model=self, T=T, N=N, initialise_from_prior=True, input_data=input_data)
        state_seqs = s.generate_states()
        data = s.generate_obs()
        if keep:
            self.states_list.append(s)
        return data, state_seqs
    
    def predict(self, seed_data, masks=None, input_data=None, covariates=None, state_seqs=None, fixed_state_seqs=False):
        self.add_data(data=seed_data, masks=masks, input_data=input_data, covariates=covariates,
                      state_seqs=state_seqs, fixed_state_seqs=fixed_state_seqs, initialise_from_prior=False)
        s = self.states_list.pop()
        s.resample()
        # s.generate_states
        return s.generate_obs(), s.state_seqs
    
    def _clear_caches(self):
        for s in self.states_list:
            s.clear_caches()
            
    def resample_model(self):
        self.resample_parameters()
        self.resample_states()
        
    def resample_parameters(self):
        self.resample_obs_distns()
        self.resample_trans_distn()
        
    def resample_obs_distns(self):
        # print("Resample observation distributions.")
        for state, distn in enumerate(self.obs_distns):
            distn.resample([np.dstack((s.input_data, s.data[:, :, None]))[s.state_seqs == state] for s in self.states_list])
        self._clear_caches
        
    def resample_trans_distn(self):
        # print("Resample transition distribution.")
        self.trans_distn.resample([(s.state_seqs, s.masks) for s in self.states_list])
        self._clear_caches
    
    def resample_states(self):
        # print("Resample states.")
        for s in self.states_list:
            s.resample()
            
    def sample_predictions(self, data, covariate_pred, covariates, time_steps, n_units, input_pred=None, 
                           state_seqs=None, state_seqs_pred=None, masks=None, masks_pred=None, input_data=None, fixed_state_seqs=False):
        print("data ", data.shape)
        print("state_seqs ", state_seqs.shape)
        print("covariates ", covariates.shape)
        print("masks ", masks.shape)
        self.add_data(data=data, masks=masks, input_data=input_data, state_seqs=state_seqs, fixed_state_seqs=fixed_state_seqs)
        s = self.states_list.pop()
        return s.sample_predictions(Tpred=time_steps, Npred=n_units, input_pred=input_pred,
                                    masks_pred=masks_pred, state_seqs_pred=state_seqs_pred,
                                    covariate_pred=covariate_pred, states_noise=True, obs_noise=True)

    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {'obs_distns': [distn.params for distn in self.obs_distns],
                             'trans_distn': self.trans_distn.params}}
        return params
    

class InputBase(Base):

    _states_class = BaseInputStates

    def __init__(self, obs_distns, transition_distn=None, D_in=0):
        self.D_in = D_in
        super().__init__(obs_distns, transition_distn)

    def add_data(self, data, covariates=None, initialise_from_prior=False, **kwargs):
        self.states_list.append(
            self._states_class(model=self, data=data, covariates=covariates, 
                               initialise_from_prior=initialise_from_prior, **kwargs))
        return self.states_list[-1]
    
    def sample_predictions(self, data, covariate_pred, covariates, time_steps, n_units, input_pred=None, 
                           state_seqs=None, state_seqs_pred=None, masks=None, masks_pred=None, input_data=None, fixed_state_seqs=False):
        self.add_data(data=data, covariates=covariates, masks=masks, input_data=input_data, 
                      state_seqs=state_seqs, fixed_state_seqs=fixed_state_seqs)
        s = self.states_list.pop()
        return s.sample_predictions(Tpred=time_steps, Npred=n_units, input_pred=input_pred,
                                    masks_pred=masks_pred, state_seqs_pred=state_seqs_pred,
                                    covariate_pred=covariate_pred, states_noise=True, obs_noise=True)

            
    def generate(self, N, T, covariates=None, keep=True):
        if covariates is None:
            covariates = np.zeros((T, N, self.D_in))

        s = self._states_class(model=self, covariates=covariates, N=N, T=T, initialise_from_prior=True)
        s.generate_states()
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return data, s.state_seqs

    def resample_trans_distn(self):
        # print("Resample transition distributions.")
        # self.trans_distn.resample(
        #     state_seqs=[s.state_seqs for s in self.states_list],
        #     cov_seqs=[s.covariates for s in self.states_list],
        #     masks=[s.masks[1:] for s in self.states_list],
        # )
        self.trans_distn.resample(
            state_seqs=[s.state_seqs for s in self.states_list],
            cov_seqs=[s.covariates for s in self.states_list],
            masks=[s.masks for s in self.states_list],
        )
        self._clear_caches()

    @property
    def params(self):
        params = super(InputBase, self).params
        params['params']['D_in'] = self.D_in
        return params
    
    
class NonParametricInputBase(InputBase):
    
    def resample_trans_distn(self):
        # print("Resample transition distributions.")
        self.trans_distn.resample(
            state_seqs=[s.state_seqs for s in self.states_list],
            cov_seqs=[s.covariates for s in self.states_list],
            masks=[s.masks for s in self.states_list],
            active_K=self.num_active_states
        )
        self._clear_caches()
                            
                            
        
    