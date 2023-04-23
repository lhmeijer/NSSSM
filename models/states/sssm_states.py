# -*- coding: utf-8 -*-
import numpy as np
from models.states.hmm_states import HMMStates, HMMInputStates, NonParametricHMMInputStates, TruncatedHMMInputStates
from models.utils.kalman_filter import KalmanFilter


class SwitchingSSMStates(HMMStates):
    
    def __init__(self, model, gaussian_states=None, T=None, N=None, data=None, **kwargs):
        
        T = T if data is None else data.shape[0]
        N = N if data is None else data.shape[1]
        self.model = model
        
        if gaussian_states is None:
            gaussian_states = np.full((T, T, N, self.D_latent), np.nan)

        self.gaussian_states = gaussian_states
        
        super(SwitchingSSMStates, self).__init__(model=model, T=T, N=N, data=data, **kwargs)
        
    def generate_states(self, T=None, N=None, state_seqs=None, covariates=None, masks=None):
        
        state_seqs = super(SwitchingSSMStates, self).generate_states(
            T=T, N=N, state_seqs=state_seqs, covariates=covariates, masks=masks)
        
        T = self.T if T is None else T
        N = self.N if N is None else N
        n, K = self.D_latent, self.num_states
        
        dss = state_seqs
        full_gss = np.zeros((K, T, N, n), dtype='double')
        gss = np.zeros((T, T, N, n), dtype='double')
        
        for state, distn in enumerate(self.init_dynamics_distns):
            indexes = np.arange(N)[dss[0] == state]
            full_gss[state, 0] = distn.rvs(size=N)
            gss[0, -1, indexes] = full_gss[state, 0, indexes]
            
        for t in range(1, T):
            for state, distn in enumerate(self.dynamics_distns):
                indexes = np.arange(N)[dss[t] == state]
                full_gss[state, t] = distn.rvs(
                    size=N, return_xy=False,
                    x=np.hstack((full_gss[state, t-1], self.input_data[t-1])))
                gss[t, -1, indexes] = full_gss[state, t, indexes]

        self.gaussian_states = gss if self.gaussian_states is None else self.gaussian_states
        return gss, dss
        
    def sample_predictions(self, Tpred, Npred, input_pred=None, masks_pred=None, states_noise=True, obs_noise=True,
                           state_seqs_pred=None, covariate_pred=None):

        state_seqs = super(SwitchingSSMStates, self).generate_states(Tpred, Npred, state_seqs_pred, covariate_pred, masks_pred)
        
        sigma_init_set, mu_init_set = self.sigma_init_set, self.mu_init_set
        A_set, B_set, Q_set = self.A_set, self.B_set, self.Q_set
        C_set, D_set, R_set = self.C_set, self.D_set, self.R_set
        
        input_pred = np.zeros((Tpred, Npred, self.D_input)) if input_pred is None else input_pred
        gaussian_states = np.full((Tpred, Npred, self.D_latent), np.nan)

        all_data = np.full((Tpred, Npred), np.nan)
        all_data = np.concatenate((self.data, all_data), axis=0)
        all_input_data = np.concatenate((self.input_data, input_pred), axis=0)
        all_state_seqs = np.concatenate((self.state_seqs, state_seqs), axis=0)
        
        T, N = self.data.shape

        for state, distn in enumerate(self.dynamics_distns):
            # print("state ", state)
  
            kf = KalmanFilter(
                mu_init=mu_init_set[state], sigma_init=sigma_init_set[state],
                A=A_set[state], B=B_set[state], Q=Q_set[state], C=C_set[state], 
                D=D_set[state], R=R_set[state])
            
            new_masks = (all_state_seqs == state) & (~np.isnan(all_data))
            _, _, init_mu, init_sigma = kf.kalman_filter(all_data, new_masks, all_input_data)
                     
            # print("init_sigma ", init_sigma)
            states = np.zeros((Tpred, Npred, self.D_latent))
            for t in range(T, Tpred+T):
                if states_noise:
                    sigma_chol = np.linalg.cholesky(init_sigma[t])
                    states[t-T] = init_mu[t] + np.random.normal(size=init_mu[t].shape).dot(sigma_chol.T)    
                else:
                    states[t-T] = init_mu[t]
                    
            # if states_noise:
            #     sigma_chol = np.linalg.cholesky(init_sigma)
            #     states[0] = init_mu + np.random.normal(size=init_mu.shape).dot(sigma_chol.T)    
            # else:
            #     states[0] = init_mu


            # for t in range(1, Tpred):
            #     if states_noise:
            #         states[t] = distn.rvs(
            #             return_xy=False, x=np.hstack((states[t-1], input_pred[t])))
            #     else:
            #         states[t] = distn.predict(x=np.hstack((states[t-1], input_pred[t])))
            #     # print('states ', states[t, :10])
     
            indexes = state_seqs == state
            gaussian_states[indexes] = states[indexes]


        obs = self.generate_obs(gaussian_states, state_seqs, masks_pred, 
                                 input_pred, obs_noise) 
        print("obs ", obs[:, :15])
        return obs, state_seqs
        
    def generate_obs(self, gaussian_states=None, state_seqs=None, masks=None, 
                     input_data=None, obs_noise=True):
        
        input_data = self.input_data if input_data is None else input_data
        masks = self.masks if masks is None else masks
        gss = self.gaussian_states[:, -1] if gaussian_states is None else gaussian_states
        dss = self.state_seqs if state_seqs is None else state_seqs
        
        T, N = masks.shape
        data = np.full((T, N), np.nan, dtype='double')
        
        for t in range(T):
            if self.single_emission:
                if obs_noise:
                    data[t] = self.emission_distns[0].rvs(
                        return_xy=False, x=np.hstack((gss[t], input_data[t]))).ravel()
                else:
                    data[t] = self.emission_distns[0].predict(x=np.hstack((gss[t], input_data[t]))).ravel()
            else:
                for state, distn in enumerate(self.emission_distns):
                    indexes =  np.arange(N)[dss[t] == state]
                    if obs_noise:
                        data[t, indexes] = distn.rvs(
                            return_xy=False, x=np.hstack((gss[t, indexes], input_data[t, indexes])))
                    else:
                        data[t, indexes] = distn.predict(
                            x=np.hstack((gss[t, indexes], input_data[t, indexes])))
        data[~masks] = np.nan
        return data
    
    @property
    def single_emission(self):
        return self.model.single_emission
                    
    @property
    def D_latent(self):
        return self.dynamics_distns[0].D_out
    
    @property
    def D_input(self):
        return self.dynamics_distns[0].D_in - self.dynamics_distns[0].D_out
    
    @property
    def D_emission(self):
        return self.emission_distns[0].D_out
    
    @property
    def init_dynamics_distns(self):
        return self.model.init_dynamics_distns
    
    @property
    def dynamics_distns(self):
        return self.model.dynamics_distns
    
    @property
    def emission_distns(self):
        return self.model.emission_distns
    
    @property
    def mu_init_set(self):
        return np.concatenate([distn.mu[None, ...] for distn in self.init_dynamics_distns])

    @property
    def sigma_init_set(self):
        return np.concatenate([distn.sigma[None, ...] for distn in self.init_dynamics_distns])

    @property
    def A_set(self):
        return np.concatenate([distn.A[None, :, :self.D_latent] for distn in self.dynamics_distns])

    @property
    def B_set(self):
        return np.concatenate([distn.A[None, :, self.D_latent:] for distn in self.dynamics_distns])

    @property
    def Q_set(self):
        return np.concatenate([distn.sigma[None, ...] for distn in self.dynamics_distns])

    @property
    def C_set(self):
        return np.concatenate([distn.A[None, :, :self.D_latent] for distn in self.emission_distns])

    @property
    def D_set(self):
        return np.concatenate([distn.A[None, :, self.D_latent:] for distn in self.emission_distns])

    @property
    def R_set(self):
        return np.concatenate([distn.sigma[None, ...] for distn in self.emission_distns])
    
    @property
    def log_likelihoods(self):
        if self._log_likelihoods is None:
            log_likelihoods = self._log_likelihoods = np.zeros((self.T, self.N, self.num_states))
            ids, dds, eds = self.init_dynamics_distns, self.dynamics_distns, self.emission_distns

            for s in range(self.num_states):
                
                for t in range(self.T):
                    idx = -1 - t
                    # Initial state distribution
                    log_likelihoods[t,:,s] = ids[s].log_likelihood(self.gaussian_states[t, idx])
                    
                    if t > 0:
                        xs = np.dstack((self.gaussian_states[t,idx:-1], self.input_data[:t]))
                        log_likelihoods[t, :, s] += np.sum(dds[s].log_likelihood((xs, self.gaussian_states[t,-t:])), axis=0)
            
            # Emissions
            xs = np.dstack((self.gaussian_states[:, -1], self.input_data))
            if self.single_emission:
                log_likelihoods += eds[0].log_likelihood((xs, self.data[..., None]))[..., None]
            else:
                for s in range(self.num_states):
                    log_likelihoods[:, :, s] += eds[s].log_likelihood((xs, self.data[..., None]))
            # print("loglikelihood ", log_likelihoods[:, 3])
            log_likelihoods[np.isnan(log_likelihoods)] = 0.

        return self._log_likelihoods

    def resample(self, n_iter=1):
        for itr in range(n_iter):
            self.resample_discrete_states()
            self.resample_gaussian_states()
        self.clear_caches()

        
    def resample_discrete_states(self):
        super(SwitchingSSMStates, self).resample()

    def resample_gaussian_states(self):
        sigma_init_set, mu_init_set = self.sigma_init_set, self.mu_init_set
        A_set, B_set, Q_set = self.A_set, self.B_set, self.Q_set
        C_set, D_set, R_set = self.C_set, self.D_set, self.R_set
        
        # self.gaussian_states = np.full((self.T, self.N, self.D_latent), np.nan)
        self.gaussian_states = np.full((self.T, self.T, self.N, self.D_latent), np.nan)
        
        for state in range(self.num_to_evaluated_states):
            dynamic_system_model = KalmanFilter(
                mu_init=mu_init_set[state], sigma_init=sigma_init_set[state],
                A=A_set[state], B=B_set[state], Q=Q_set[state], C=C_set[state], 
                D=D_set[state], R=R_set[state])
            
            indexes = self.state_seqs == state
            new_masks = indexes & ~np.isnan(self.data)
            
            states = dynamic_system_model.resample(self.data, new_masks, self.input_data)
            for t in range(self.T):
                idx = self.T - 1 - t
                a = states[:(t+1), indexes[t]]
                self.gaussian_states[t:t+1, idx:, indexes[t]] = np.expand_dims(a, axis=0)
      
  
    
class SwitchingSSMInputStates(SwitchingSSMStates, HMMInputStates):
    pass
      

    
class SwitchingSSMTruncatedInputStates(SwitchingSSMStates, TruncatedHMMInputStates):
    pass
    
    
class SwitchingSSMNonParametricInputStates(SwitchingSSMStates, NonParametricHMMInputStates):
    pass

        
    