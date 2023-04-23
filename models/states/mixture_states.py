# -*- coding: utf-8 -*-
from models.states.base_states import BaseStates
import numpy as np


class MixtureStates(BaseStates):
    
    def _resample(self):
        T, N = self.T, self.N
        log_likelihoods = self.log_likelihoods
        
        scores = log_likelihoods * self.weights
        
        state_seqs = np.full((T, N), -1, dtype=np.int32)
        for t in range(self.T):
            scores = np.exp(log_likelihoods[t] * self.weights)
            for i in range(self.N):
                p = scores[i] / np.sum(scores[i])
                state_seqs[t, i] = np.random.choice(np.arange(self.num_states), p=p)
        self.state_seqs = state_seqs