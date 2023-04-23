# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import special

from models.utils.stats import sample_crp_table_counts

from models.distributions.polya_gamma import MultinomialLogisticRegression

# from polyagamma import random_polyagamma


class Categorical(object):
    
    _name = 'Categorical' 
    
    def __init__(self, weights=None, alpha_0=None, alpha_v_0=None, K=None):
        self.K = K if K is not None else len(weights)
        self.alpha_0 = alpha_0
        self.alpha_v_0 = alpha_v_0
        self.weights = weights
        
        if weights is None:
            self.resample()
            
    @property
    def alpha_0(self):
        return self._alpha_0
    
    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self._alpha_0 = alpha_0
        if not any(_ is None for _ in (self.K, self._alpha_0)):
            self.alpha_v_0 = np.repeat(self._alpha_0 / self.K, self.K)
            
    @property
    def alpha_v_0(self):
        return self._alpha_v_0
    
    @alpha_v_0.setter
    def alpha_v_0(self, alpha_v_0):
        if alpha_v_0 is not None:
            self._alpha_v_0 = alpha_v_0
            self.K = len(alpha_v_0)
            
    def rvs(self, size=None):
        return np.random.choice(np.arange(self.K), p=self.weights, size=size)
    
    def resample(self, data=None, counts=None):
        counts = self._get_statistics(data) if counts is None else counts
        self.weights = np.random.dirichlet(self.alpha_v_0 + counts)
        
    def _get_statistics(self, data, K=None):
        K = K if K is not None else self.K

        if data is None or len(data) == 0:
            counts = np.zeros(K, dtype=np.int32)
        elif isinstance(data, np.ndarray):
            counts = np.bincount(data, minlength=K)
        else:
            counts = sum(np.bincount(d, minlength=K) for d in data)
        return counts
    
    @property
    def params(self):
        params = {'name': self._name, 'params': {'alpha_0': self.alpha_0, 'weights': self.weights.tolist(), 'alpha_v_0': self.alpha_v_0.tolist()}}
        return params
    

class CategoricalAndConcentration(Categorical):
    
    def __init__(self, a_0, b_0, K=None, weights=None, alpha_0=None):
        self.a_0, self.b_0 = a_0, b_0
        self.alpha_0_obj = GammaCompoundDirichlet(a_0=a_0, b_0=b_0, K=K, 
                                                  concentration=alpha_0)
        super(CategoricalAndConcentration, self).__init__(
            alpha_0=self.alpha_0, K=K, weights=weights)
        
    @property
    def alpha_0(self):
        return self.alpha_0_obj.concentration
    
    @alpha_0.setter
    def alpha_0(self, alpha_0):
        self.alpha_0_obj.concentration = alpha_0
        self.alpha_v_0 = np.repeat(alpha_0 / self.K, self.K)
        
    def resample(self, data=None, counts=None):
        counts = self._get_statistics(data) if counts is None else counts
        self.alpha_0_obj.resample(counts=counts)
        self.alpha_0 = self.alpha_0
        super(CategoricalAndConcentration, self).resample(data=data, counts=counts)
        
        
class MultinomialLogistic(object):

    _name = 'MultinomialLogistic'
    _regression_distn = MultinomialLogisticRegression

    def __init__(self, num_states, covariates_dim, **kwargs):
        self.K = num_states
        self.covariates_dim = covariates_dim
        self.regression_distn = self._regression_distn(D_out=num_states, D_in=1+covariates_dim, **kwargs)

    def weights(self, x):
        return self.regression_distn.pi(x)

    def log_likelihood(self, x):
        out = np.zeros_like(x, dtype=np.double)
        nanidx = np.isnan(x)
        err = np.seterr(divide='ignore')
        out[~nanidx] = np.log(self.weights(x))[list(x[~nanidx])]  # log(0) can happen, no warning
        np.seterr(**err)
        return out

    def rvs(self, x):
        weights = self.weights(x)
        T, N = weights.shape
        return np.array([np.array([np.random.choice(np.arange(self.K), p=weights[t, i]) for i in range(N)]) for t in range(T)])

    def resample(self, state_seqs=None, cov_seqs=None, omegas=None):
        def align_lags(state_seqs, cov_seqs):
            prev_state = np.ones_like(state_seqs[1:])
            next_state = np.array(state_seqs[1:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
            return np.dstack((prev_state, cov_seqs[1:])), next_state

        datas = [align_lags(z, x) for z, x in zip(state_seqs, cov_seqs)]

        # Clip the last data column since it is redundant
        # and not expected by the MultinomialRegression
        datas = [(x, y[:, :, :-1]) for x, y in datas]
        self.regression_distn.resample(datas, omega=omegas)

    def params(self):
        params = {
            'name': self._name, 'params': {'num_states': self.K, 'covariates_dim': self.covariates_dim,
                                           **self.regression_distn.params()}}
        return params
        
        
class CRP(object):
    
    def __init__(self, a_0, b_0, concentration):
        self.a_0, self.b_0 = a_0, b_0
        self.concentration = concentration
        
        if concentration is None:
            self.resample(n_iter=1)
    
    def resample(self, counts=None, n_iter=50):
        for itr in range(n_iter):
            a_n, b_n = self._posterior_hypparams(*self._get_statistics(counts))
            self.concentration = np.random.gamma(a_n, scale=1./b_n)
            
    def _posterior_hypparams(self, sample_numbers, total_num_distinct):
        
        a_n = self.a_0 
        b_n = self.b_n
        if total_num_distinct > 0:
            sample_numbers = sample_numbers[sample_numbers > 0]
            
            wvec = np.random.beta(self.concentration + 1, sample_numbers)
            svec = stats.bernoulli.rvs(sample_numbers / (sample_numbers + self.concentration))
            a_n += total_num_distinct - svec.sum()
            b_n -= np.log(wvec).sum()
        return a_n, b_n
        
        
        
class GammaCompoundDirichlet(CRP):
    
    def __init__(self, K, a_0, b_0, concentration=None):
        self.K = K
        super(GammaCompoundDirichlet, self).__init__(
            a_0=a_0, b_0=b_0, concentration=concentration)
        
    def rvs(self, sample_counts=None, size=None):
        pass
    
    def resample(self, data=None, n_iter=50, weighted_cols=None):
        if weighted_cols is None:
            self.weighted_cols = np.ones(self.K)
        else:
            self.weighted_cols = weighted_cols
            
        if isinstance(data, np.ndarray):
            size = data.sum()
        elif isinstance(data, list):
            size = sum(d.sum() for d in data)
        else:
            size = 0
        
        if size == 0:
            n_iter = 1
        
        return super(GammaCompoundDirichlet, self).resample(data, n_iter=n_iter)
    
    def _get_statistics(self, counts):
        
        if counts is None:
            return 0, 0
        
        m = sample_crp_table_counts(self.concentration, counts, self.weighted_cols)
        return counts.sum(1), m.sum()

        