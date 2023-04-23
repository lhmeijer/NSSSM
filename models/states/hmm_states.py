# -*- coding: utf-8 -*-
from models.states.base_states import BaseStates, BaseInputStates, NonParametricBaseInputStates, TruncatedBaseInputStates
import numpy as np
from models.utils.stats import sample_discrete_number
from scipy.special import softmax


class HMMStates(BaseStates):
    
    def generate_states(self, T=None, N=None, state_seqs=None, covariates=None, masks=None):

        masks = self.masks if masks is None else masks
        N = self.N if N is None else N
        T = self.T if T is None else T
        
        if state_seqs is None:
            A = self.trans_matrix()
            A_0 = self.init_trans_vec()
            
            state_seqs = np.full((T, N), -1, dtype=np.int32)
            state_seqs[0] = np.random.choice(np.arange(self.num_states), p=A_0, size=N)
            
            for t in range(1, T):
                for k in range(self.num_states):
                    indexes = np.where(state_seqs[t] == k)[0]
                    state_seqs[t, indexes] = np.random.choice(np.arange(self.num_states), p=A[k], size=len(indexes))
            state_seqs[~masks] = -1
        else:
            assert state_seqs.shape == (T, N)
        self.state_seqs = state_seqs if self.state_seqs is None else self.state_seqs
        return state_seqs
        
    def _resample(self):
        state_probs = self.sample_forwards()
        self.sample_backwards(state_probs)
        
    def sample_forwards(self):
        return self._sample_forwards(self.trans_matrix(), self.init_trans_vec(),
                                     self.log_likelihoods)
    
    def _sample_forwards(self, p_transition, p_initial, log_likelihoods):
        T, N, K = log_likelihoods.shape
        state_probs = np.zeros_like(log_likelihoods, dtype='double')
        state_probs[0] = softmax(log_likelihoods[0] + np.log(p_initial), axis=1)
        
        for t in range(1, T):
            p_temp = self._p_forwards(t, state_probs[t-1], p_transition)
            p_temp = np.log(p_temp) + log_likelihoods[t]
            p_temp[~np.isfinite(p_temp)] = -1000
            state_probs[t] = softmax(p_temp, axis=1)
            # print("state_probs[t] ", state_probs[t, 0])
            # state_probs[t, ~np.isfinite(state_probs[t])] = 0.
        
        return state_probs

    @staticmethod
    def _p_forwards(t, state_probs, p_transition):
        p_temp = np.einsum('nk,kl->nl', state_probs, p_transition)
        return p_temp

    def sample_backwards(self, state_probs):
        state_seqs = self._sample_backwards(state_probs, self.trans_matrix())
        state_seqs[~self.masks] = -1
        # print('state seqs ', np.bincount(state_seqs[0][self.masks[0]]))
        # print('state seqs ', np.bincount(state_seqs[3][self.masks[3]])) 
        # print('state seqs ', np.bincount(state_seqs[-1][self.masks[-1]]))  
        self.state_seqs = state_seqs

    def _sample_backwards(self, state_probs, p_transition):
        

        T, N, K = state_probs.shape
        state_seqs = np.full((T, N), -1, dtype=np.int32)
        state_seqs[T-1] = sample_discrete_number(state_probs[T-1], self.num_to_evaluated_states)
        

        for t in range(T-2, -1, -1):
            # print('state_rpobs ', state_probs[t, 0])
            p_temp = self._p_backwards(t, state_seqs[t+1].reshape(-1, 1), state_probs[t], p_transition)
            state_probs[t] = p_temp / np.sum(p_temp, axis=1).reshape(-1, 1)
            # print('state_probs[t] ', state_probs[t, 0])
            state_probs[t, ~np.isfinite(state_probs[t])] = 0.0
            state_seqs[t] = sample_discrete_number(state_probs[t], self.num_to_evaluated_states)

        return state_seqs

    @staticmethod
    def _p_backwards(t, s, state_probs, p_transition):
        p_temp = p_transition[:, s].T * state_probs
        return p_temp[0]


class HMMInputStates(HMMStates, BaseInputStates):

    def generate_states(self, T=None, N=None, state_seqs=None, covariates=None, masks=None):
        
        covariates = self.covariates if covariates is None else covariates
        masks = self.masks if masks is None else masks
        N = self.N if N is None else N
        T = self.T if T is None else T
        K = self.num_states

        if state_seqs is None:
            A = self.trans_matrix(covariates)
            A_0 = self.init_trans_vec(covariates[0])

            state_seqs = np.empty((T, N), dtype=np.int32)
            
            if self.state_seqs is not None:
                ps = np.take_along_axis(A[0], np.expand_dims(self.state_seqs[-1], axis=(1,2)), axis=1)
                state_seqs[0] = np.argmax(ps[:,0], axis=1)
                # state_seqs[0] = sample_discrete_number(ps[:,0])
            else:
                # state_seqs[0] = [np.random.choice(np.arange(K), p=A_0[i]) for i in range(N)]
                state_seqs[0] = np.argmax(A_0, axis=1)


            for t in range(1, T):
                ps = np.take_along_axis(A[t], np.expand_dims(state_seqs[t-1], axis=(1,2)), axis=1)
                # state_seqs[t] = sample_discrete_number(ps[:,0])
                state_seqs[t] = np.argmax(ps[:,0], axis=1)

            state_seqs[~masks] = -1

            self.state_seqs = state_seqs if self.state_seqs is None else self.state_seqs
        else:
            assert state_seqs.shape == (T, N)
            self.state_seqs = state_seqs if self.state_seqs is None else self.state_seqs
 
        return state_seqs

    @staticmethod
    def _p_forwards(t, state_probs, p_transition):
        p_temp = np.einsum('nk,nkl->nl', state_probs, p_transition[t])
        return p_temp
    
    @staticmethod
    def _p_backwards(t, s, state_probs, p_transition):
        p_temp = np.take_along_axis(p_transition[t+1], s[:, None], axis=2)[..., 0] * state_probs
        return p_temp
    
    
class TruncatedHMMInputStates(HMMInputStates, TruncatedBaseInputStates):
    pass
    

class NonParametricHMMInputStates(HMMInputStates, NonParametricBaseInputStates):
        
    def sample_forwards(self):
        return self._sample_forwards(
            self.trans_matrix(), self.init_trans_vec(), self.log_likelihoods, 
            self.auxiliary_variables)
    
    def sample_backwards(self, state_probs):
        state_seqs = self._sample_backwards(
            state_probs, self.trans_matrix(), self.auxiliary_variables, 
            active_K=self.num_active_states)
        
        # print(state_seqs)
        # print(self.state_seqs)      
        # equal_states = state_seqs == self.state_seqs
        # for t in range(self.T):
        #     indexes = np.arange(self.N)[equal_states[t]]
        #     print(len(indexes) / len(np.arange(self.N)[self.masks[t]]))
        
        state_seqs[~self.masks] = -1
        # max_state = np.max(state_seqs)
        # print("max ", max_state)
        # print(self.auxiliary_variables[state_seqs == max_state])
        # print(state_probs[state_seqs == max_state])
        print(np.bincount(state_seqs[0, self.masks[0]]))
        # print(np.bincount(state_seqs[-1, self.masks[-1]]))
        # print(hallo)
        self.state_seqs = state_seqs
    
    
    def _sample_forwards(self, p_transition, p_initial, log_likelihoods, u):
        # print("likelihood ", log_likelihoods.shape)
        # print("p_transition ", p_transition.shape)
        # print("p_initial ", p_initial[0])
        # print("u ", u[0,0])
        T, N, Km = log_likelihoods.shape
        # state_probs = np.zeros_like(log_likelihoods, dtype='double')
        state_probs = np.zeros((T, N, Km), dtype='double')
        p_temp = np.full((T, N, Km), -1000)
        # print("probability ", p_initial[0])
        # print("probability ", p_transition[1, 0])
        # likelihoods = np.exp(log_likelihoods)
        # likelihoods[np.isinf(log_likelihoods)] = 1.
        # print('p_initial ', p_initial)
        # print("p_transition ", p_transition)
        # print('log_likelihoods ', log_likelihoods[:, 0])
        # print("u ", u[3])
        # print('p_initial ', p_initial[0])
        # print("u ", u[0,0])
        satisfy = u[0].reshape(-1, 1) < p_initial
        # print("satisfy ", satisfy[0])
        # print('log_likelihoods[0 ', log_likelihoods[0,0])
        p_temp[0, satisfy] = log_likelihoods[0, satisfy]
        # print("p_temp[0] ", p_temp[0, 0])
        state_probs[0] = softmax(p_temp[0], axis=1)
        # print('state_probs[0] ', state_probs[0, 0])
        # print("finite ", len(np.arange(N)[np.isfinite(state_probs[0, :, 0])]))
        # state_probs[0, ~np.isfinite(state_probs[0])] = 0.

        # state_probs[0, np.isnan(state_probs[0])] = 0.
        # state_probs[0, np.isinf(state_probs[0])] = 0.
        
        # print('state_probs ', state_probs[0,:10])
        # state_probs[0] = likelihoods[0] * p_initial
        # print('state_probs[0] ', state_probs[0,3])
        # print('p_initial ', p_initial[3])
        # print("p_transition ", p_transition[1,3])
        for t in range(1, T):
            # print('u[t] ', u[t, 0])
            # print(p_transition[t, 0, 0])
            satisfy = np.expand_dims(u[t], axis=(1,2)) < p_transition[t]
            # print('state_probs[0] ', state_probs[t-1, 0])
            broadcasted = np.broadcast_to(state_probs[t-1, ..., None], state_probs[t-1].shape + (Km,))
            broadcasted = np.transpose(broadcasted, (0, 2, 1))
            # print("broadcasted ", broadcasted[0])

            p_temp = np.zeros((N, Km, Km), dtype=np.float32)
            p_temp[satisfy] = broadcasted[satisfy]
            # print("p_temp ", p_temp[0])
            p_temp = np.sum(p_temp, axis=1)
            # p_temp = np.einsum('nkl->nl', state_probs[t-1, satisfy])
            # print("p_temp ", p_temp[0])
            # print(p_temp[0])
            # print(np.log(p_temp[0]))
            # print("log_likelihood ",  log_likelihoods[t,0])
            # p_temp += 1e-12
            # print('log_likelihoods[t] ', log_likelihoods[t, 0])
            # print('p_temp ', p_temp[0])
            # print("p_temp ", p_temp[:10])
            # print("log_likelihoods[t] ", log_likelihoods[t, 0])
            p_temp = np.log(p_temp) + log_likelihoods[t]
            # print("p_temp ", p_temp[:10])
            # p_temp[~np.isfinite(p_temp)] = -50
            state_probs[t] = softmax(p_temp, axis=1)    # only zeros return to a array of nans
            # print('state_probs[t] ', state_probs[t, 0])
            state_probs[t, ~np.isfinite(state_probs[t])] = 0.
            # state_probs[t, np.isnan(state_probs[t])] = 0.
            # state_probs[t, np.isinf(state_probs[t])] = 0.
            # state_probs[t, :, K:] = 0
            # print('log_likelihoods[t] ', log_likelihoods[t,0])
            # print("state_probs[t] ", state_probs[t, 0])
            # print("state_probs[t] ", state_probs[t])
            # print(np.arange(N)[np.isnan(log_likelihoods[t, :, 0])])

            # p_temp = likelihoods[t] * p_temp
            # for k in range(K):
                # print(np.arange(N)[np.isinf(p_temp[:, k])])
                # print()
            # state_probs[t] = p_temp / np.sum(p_temp, axis=1).reshape(-1, 1)
        
        return state_probs
    
    def _sample_backwards(self, state_probs, p_transition, u, active_K):

        T, N, K = state_probs.shape
        state_seqs = np.empty((T, N), dtype=np.int32)
        # print(state_probs[T-1, :10])
        state_seqs[T-1] = sample_discrete_number(state_probs[T-1], active_K)
        # print(state_seqs[T-1])
        # print(state_probs.shape)
        # print('u ', u.shape)
        # print("active_K ", active_K)

        for t in range(T-2, -1, -1):
            # print(p_transition[t+1].shape)
            # print(state_seqs[t+1].reshape(-1, 1).shape)
            # print( np.expand_dims(state_seqs[t+1], axis=(1, 2)).shape)
            # print(np.take_along_axis(p_transition[t+1], np.expand_dims(state_seqs[t+1], axis=(1, 2)), axis=2))
            satisfy = np.expand_dims(u[t+1], axis=(1, 2)) < np.take_along_axis(p_transition[t+1], np.expand_dims(state_seqs[t+1], axis=(1, 2)), axis=2)
            p_temp = np.zeros((N, K))
            p_temp[satisfy[..., 0]] = state_probs[t, satisfy[..., 0]]
            state_probs[t] = p_temp / np.sum(p_temp, axis=1).reshape(-1, 1)
            # print('state_probs ', state_probs[t, :10])
            state_probs[t, ~np.isfinite(state_probs[t])] = 0.
            state_seqs[t] = sample_discrete_number(state_probs[t], active_K)
        return state_seqs