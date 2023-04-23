# -*- coding: utf-8 -*-
import numpy as np
import itertools


class BaseStates(object):
    
    def __init__(self, model, T=None, N=None, data=None, masks=None, input_data=None,
                 state_seqs=None, fixed_state_seqs=False, initialise_from_prior=True):
        
        self.model = model
        self.T = T if data is None else data.shape[0]
        self.N = N if data is None else data.shape[1]
        self.data = data
        
        self.clear_caches()
        
        if masks is None:
            masks = np.ones((self.T, self.N), dtype=bool)
        self.masks = masks
                
        if input_data is None:
            input_data = np.zeros((self.T, self.N, 0))
        self.input_data = input_data
        
        self.fixed_state_seqs = fixed_state_seqs
        self.state_seqs = state_seqs
        if fixed_state_seqs:
            assert state_seqs is not None, "..."
            
        if initialise_from_prior:
            self.generate_states(state_seqs=state_seqs)
        elif data is not None:
            self.resample()
            
    def generate_states(self, T=None, N=None, state_seqs=None, covariates=None, masks=None):
        self.state_seqs = state_seqs
        return state_seqs
    
    def clear_caches(self):
        self._log_likelihoods = None
        
    def clear_log_likelihoods(self):
        self._log_likelihoods = None

    def generate_obs(self, state_seqs=None, input_data=None, masks=None, obs_noise=True):
        s = self.state_seqs if state_seqs is None else state_seqs
        input_data = self.input_data if input_data is None else input_data
        masks = self.masks if masks is None else masks
        
        T, N = s.shape
        data = np.full((T, N), np.nan)
        for state, distn in enumerate(self.obs_distns):
            for t in range(T):
                indexes = np.arange(N)[s[t] == state]
                if obs_noise:
                    data[t, indexes] = distn.rvs(x=input_data[t, indexes], return_xy=False).ravel()
                else:
                    data[t, indexes] = distn.predict(x=input_data[t, indexes]).ravel()
        return data
    
    @property
    def log_likelihoods(self):
        if self._log_likelihoods is None:
            log_likelihoods = np.zeros((self.T, self.N, self.num_states))
            for state, distn in enumerate(self.obs_distns):
                for t in range(self.T):
                    log_likelihoods[t, :, state] = distn.log_likelihood((self.input_data[t], self.data[t, :, None]))
            log_likelihoods[np.isnan(log_likelihoods)] = 0.
            self._log_likelihoods = log_likelihoods
        return self._log_likelihoods
                    
    def resample(self):
        if not self.fixed_state_seqs:
            self._resample()
        self.clear_log_likelihoods()
        
    def _resample(self):
        pass
    
    def sample_predictions(self, Tpred, Npred, input_pred=None, masks_pred=None, 
                           states_noise=True, obs_noise=True, state_seqs_pred=None, covariate_pred=None):
        state_seqs_pred = self.generate_states(Tpred, Npred, state_seqs_pred, covariate_pred, masks_pred)
        return self.generate_obs(state_seqs_pred, input_pred, masks_pred, obs_noise), state_seqs_pred
    
    @property
    def obs_distns(self):
        return self.model.obs_distns
    
    def trans_matrix(self, x=None):
        return self.model.trans_distn.trans_matrix

    def init_trans_vec(self, x=None):
        return self.model.trans_distn.init_trans_vec
    
    @property
    def num_states(self):
        return self.model.num_states
    
    @property
    def num_to_evaluated_states(self):
        return self.num_states
    
    
class BaseInputStates(BaseStates):

    def __init__(self, covariates, **kwargs):
        self.covariates = covariates
        super(BaseInputStates, self).__init__(**kwargs)

    def init_trans_vec(self, x=None):
        if self._init_trans_vec is None: 
            x = self.covariates[0] if x is None else x
            self._init_trans_vec = self.model.trans_distn.get_init_trans_vec(x)
        return  self._init_trans_vec

    def trans_matrix(self, x=None):
        if self._trans_matrix is None:
            x = self.covariates if x is None else x
            self._trans_matrix = self.model.trans_distn.get_trans_matrix(x)
        return self._trans_matrix
    
    def clear_caches(self):
        self._log_likelihoods = None
        self._init_trans_vec = None
        self._trans_matrix = None
        
        
class TruncatedBaseInputStates(BaseInputStates):
    
    @property
    def num_active_states(self):
        if self._active_states is None:
            if self.state_seqs is None:
                self._active_states = self.model.trans_distn.active_states
            else:
                self._active_states = np.max(self.state_seqs) + 1
        return self._active_states
    
    def clear_caches(self):
        super(TruncatedBaseInputStates, self).clear_caches()
        self._active_states = None
        
    @property
    def num_to_evaluated_states(self):
        return min(self.num_active_states + 1, self.num_states)
        
        
class NonParametricBaseStates(BaseStates):
    
    @property
    def num_active_states(self):
        if self._active_states is None:
            if self.state_seqs is None:
                self._active_states = self.model.trans_distn.active_states
            else:
                self._active_states = np.max(self.state_seqs) + 1
        return self._active_states
    
    @property
    def num_to_evaluated_states(self):
        return min(self.num_active_states + 1, self.num_states)
    
    def clear_caches(self):
        super(NonParametricBaseStates, self).clear_caches()
        self._auxiliary_variables = None
        self._active_states = None
    
    @property
    def auxiliary_variables(self):
        if self._auxiliary_variables is None:
            T, N, K = self.T, self.N, self.num_states 
            s = self.state_seqs if self.state_seqs is not None else np.zeros((T, N), dtype=np.int16)
            u = np.zeros((T, N))
            
            init_trans_vec = self.init_trans_vec()
            trans_matrix = self.trans_matrix()
            
            for s1 in range(K):
                indexes = np.arange(N)[s[0] == s1]
                u[0, indexes] = np.random.uniform(high=init_trans_vec[s1], size=len(indexes))
            
            #TO DO add if s1 does not exist
            for s1, s2 in itertools.product(range(K), range(K)):
                for t in range(1, T):
                    indexes = (s[t-1] == s1) & (s[t] == s2)
                    indexes = np.arange(N)[indexes]
                    u[t, indexes] = np.random.uniform(high=trans_matrix[s1, s2], size=len(indexes))
            u[~self.masks] = 1.
            self._auxiliary_variables = u
        return self._auxiliary_variables
    
        

class NonParametricBaseInputStates(BaseInputStates):
    
    @property
    def num_active_states(self):
        if self._active_states is None:
            if self.state_seqs is None:
                self._active_states = self.model.trans_distn.active_states
            else:
                self._active_states = np.max(self.state_seqs) + 1
        return self._active_states
    
    @property
    def num_to_evaluated_states(self):
        return min(self.num_active_states + 1, self.num_states)
    
    def clear_caches(self):
        super(NonParametricBaseInputStates, self).clear_caches()
        self._auxiliary_variables = None
        self._active_states = None
    
    @property
    def auxiliary_variables(self):
        if self._auxiliary_variables is None:
            init_trans_vec = self.init_trans_vec()
            trans_matrix = self.trans_matrix()
            
            T, N = self.T, self.N
            s = self.state_seqs if self.state_seqs is not None else np.zeros((T, N), dtype=np.int16)
            u = np.zeros((T, N), dtype='double')
            u[0] = [np.random.uniform(high=init_trans_vec[i, s[0, i]]) for i in range(N)]
            for t in range(1, T):
                u[t] =  [
                    np.random.uniform(high=trans_matrix[t, i, s[t-1, i], s[t, i]]) if s[t-1, i] > -1 else
                    np.random.uniform(high=np.max(trans_matrix[t, i, :, s[t, i]])) 
                    for i in range(N)]
            u[~self.masks] = 0.
            self._auxiliary_variables = u
        return self._auxiliary_variables
