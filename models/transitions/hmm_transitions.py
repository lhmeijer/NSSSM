# -*- coding: utf-8 -*-
from models.distributions.multinomial import Categorical, CategoricalAndConcentration
import numpy as np
from models.distributions.polya_gamma import MultinomialLogisticRegression, StickBreakingLogisticRegression, \
    RegularizedMultinomialLogisticRegression, RegularizedStickBreakingLogisticRegression
from models.utils.stats import count_transitions


class HMMTransitions(object):
    
    _name = 'HMMTransitions'
    
    def __init__(self, num_states, alpha=None, alpha_v=None, trans_matrix=None,
                 init_trans_vec=None):
        self.num_states = num_states
        
        if init_trans_vec is not None:
            self._init_dist = Categorical(alpha_0=alpha, K=num_states, alpha_v_0=alpha_v, weights=init_trans_vec)
        else:
            self._init_dist = Categorical(alpha_0=alpha, K=num_states, alpha_v_0=alpha_v)
        
        if trans_matrix is not None:
            self._distns = [Categorical(alpha_0=alpha, K=num_states, alpha_v_0=alpha_v, weights=row) for row in trans_matrix]
        else:
            self._distns = [Categorical(alpha_0=alpha, K=num_states, alpha_v_0=alpha_v) for _ in range(num_states)]
    
    @property
    def alpha(self):
        return self._distns[0].alpha_0

    @alpha.setter
    def alpha(self, alpha):
        for distn in self._rows_distns:
            distn.alpha_0 = alpha

    @property
    def alpha_v(self):
        return self._distns[0].alpha_v_0

    @alpha_v.setter
    def alpha_v(self, weights):
        for distn in self._distns:
            distn.alpha_v_0 = weights
     
    @property
    def trans_matrix(self):
        return np.array([dist.weights for dist in self._distns])
    
    @property
    def init_trans_vec(self):
        return self._init_dist.weights
    
    def resample(self, state_seqs=None, init_counts=None, trans_counts=None):
        init_counts = self._init_counts(state_seqs) if init_counts is None else init_counts
        self._init_dist.resample(counts=init_counts)
        
        trans_counts = self._count_transitions(state_seqs) if trans_counts is None else trans_counts
        for distn, counts in zip(self._distns, trans_counts):
            distn.resample(counts=counts)
        
    def _count_transitions(self, stateseqs):
        return sum((sum(count_transitions(s, m, num_states=self.num_states) for s, m in zip(stateseq.T, masks.T)) \
                    for stateseq, masks in stateseqs), np.zeros((self.num_states, self.num_states), dtype=np.int32))

    def _init_counts(self, state_seqs):
        return sum((np.bincount(s[0, m[0]], minlength=self.num_states) for s, m in state_seqs), np.zeros(self.num_states, dtype=np.int32))

    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {
                    'num_states': self.num_states,
                    'trans_matrix': self.trans_matrix.tolist(),
                    'init_trans_vec': self.init_trans_vec.tolist()
        }}
        return params
    
    
class HMMTransitionsConc(HMMTransitions):

    def __init__(self, num_states, alpha_a_0, alpha_b_0, **kwargs):
        self.alpha_obj = GammaCompoundDirichlet(num_states, alpha_a_0, alpha_b_0)
        super(HMMTransitionsConc, self).__init__(num_states=num_states, alpha=self.alpha, **kwargs)

    @property
    def alpha(self):
        return self.alpha_obj.concentration

    @alpha.setter
    def alpha(self, alpha):
        if alpha is not None:
            self.alpha_obj.concentration = alpha
            for distn in self._rows_distns:
                distn.alpha_0 = alpha

    def resample(self, state_seqs=None, transition_counts=None):
        transition_counts = self._count_transitions(state_seqs) if transition_counts is None else transition_counts
        self._resample_alpha(transition_counts)
        return super(HMMTransitionsConc, self).resample(
            state_seqs=state_seqs, transition_counts=transition_counts)

    def _resample_alpha(self, transition_counts):
        self.alpha_obj.resample(transition_counts)
        self.alpha = self.alpha_obj.concentration


class WeakLimitHDPHMMTransitions(HMMTransitions):

    _name = 'WeakLimitHDPHMMTransitions'

    def __init__(self, gamma, alpha, num_states, beta=None, **kwargs):
        self.beta_obj = Multinomial(alpha_0=gamma, K=num_states, weights=beta)
        self.alpha = alpha

        super(WeakLimitHDPHMMTransitions, self).__init__(
            num_states=num_states, alpha=alpha, alpha_v=alpha * self.beta, **kwargs)

    @property
    def beta(self):
        return self.beta_obj.weights

    @beta.setter
    def beta(self, weights):
        self.beta_obj.weights = weights
        self.alpha_v = self.alpha * self.beta

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def gamma(self):
        return self.beta_obj.alpha_0

    @gamma.setter
    def gamma(self, gamma):
        self.beta_obj.alpha_0 = gamma

    def resample(self, state_seqs=None, transition_counts=None, ms=None):
        transition_counts = self._count_transitions(state_seqs) if transition_counts is None else transition_counts
        ms = self._get_m(transition_counts) if ms is None else ms

        self._resample_beta(ms)

        return super(WeakLimitHDPHMMTransitions, self).resample(
            state_seqs=state_seqs, transition_counts=transition_counts)

    def _resample_beta(self, ms):
        self.beta_obj.resample(ms)
        self.alpha_v = self.alpha * self.beta

    def _get_m(self, transition_counts):
        if not (0 == transition_counts).all():
            m = sample_crp_tablecounts(float(self.alpha), transition_counts, self.beta)
        else:
            m = np.zeros_like(transition_counts)
        self.m = m
        return m


class WeakLimitHDPHMMTransitionsConc(WeakLimitHDPHMMTransitions):

    _name = 'WeakLimitHDPHMMTransitionsConc'

    def __init__(self, num_states, gamma_a_0, gamma_b_0, alpha_a_0, alpha_b_0,
                 beta=None, **kwargs):
        self.beta_obj = MultinomialAndConcentration(
            a_0=gamma_a_0, b_0=gamma_b_0, K=self.K, weights=beta)
        self.alpha_obj = GammaCompoundDirichlet(self.K, alpha_a_0, alpha_b_0)

        # NOTE: we don't want to call WeakLimitHDPHMMTransitions.__init__
        # because it sets beta_obj in a different way
        HMMTransitions.__init__(self, num_states=num_states, alpha_v=self.alpha * self.beta, **kwargs)

    @property
    def alpha(self):
        return self.alpha_obj.concentration

    @alpha.setter
    def alpha(self, val):
        self.alpha_obj.concentration = val

    def resample(self, state_seqs=None, transition_counts=None, ms=None):
        transition_counts = self._count_transitions(state_seqs) if transition_counts is None else transition_counts
        ms = self._get_m(transition_counts) if ms is None else ms

        self._resample_beta(ms)
        self._resample_alpha(transition_counts)

        return super(WeakLimitHDPHMMTransitionsConc, self).resample(
            state_seqs=state_seqs, transition_counts=transition_counts)

    def _resample_beta(self, ms):
        # NOTE: unlike parent, alphav is updated in _resample_alpha
        self.beta_obj.resample(ms)

    def _resample_alpha(self, transition_counts):
        self.alpha_obj.resample(transition_counts, weighted_cols=self.beta)
        self.alpha_v = self.alpha * self.beta


class WeakLimitStickyHDPHMMTransitions(WeakLimitHDPHMMTransitions):

    _name = 'WeakLimitStickyHDPHMMTransitions'

    def __init__(self, kappa, **kwargs):
        self.kappa = kappa
        super(WeakLimitStickyHDPHMMTransitions, self).__init__(**kwargs)

    @property
    def alpha_v(self):
        return self._rows_distns[0].alpha_v_0

    @alpha_v.setter
    def alpha_v(self, weights):
        self._init_row_distn.alpha_v_0 = weights
        for distn, delta_ij in zip(self._row_distns, np.eye(self.K)):
            distn.alpha_v_0 = weights + self.kappa * delta_ij

    def _get_m(self, transition_counts):
        # NOTE: this thins the m's
        ms = super(WeakLimitStickyHDPHMMTransitions, self)._get_m(transition_counts)
        newms = ms.copy()
        if ms.sum() > 0:
            # np.random.binomial fails when n=0, so pull out nonzero indices
            indices = np.nonzero(newms.flat[::ms.shape[0] + 1])
            newms.flat[::ms.shape[0] + 1][indices] = np.array(np.random.binomial(
                ms.flat[::ms.shape[0] + 1][indices],
                self.beta[indices] * self.alpha / (self.beta[indices] * self.alpha + self.kappa)),
                dtype=np.int32)
        return newms


class WeakLimitStickyHDPHMMTransitionsConc(WeakLimitStickyHDPHMMTransitions, WeakLimitHDPHMMTransitionsConc):
    pass


class HMMInputTransitions(object):

    _regression_distn = MultinomialLogisticRegression

    def __init__(self, num_states, covariates_dim, regression_distn=None, **kwargs):
        self.K = num_states
        self.covariates_dim = covariates_dim
        
        if regression_distn is None:
            regression_distn = self._regression_distn(
                D_out=num_states, D_in=num_states + covariates_dim, input_only=False, **kwargs)
        self.regression_distn = regression_distn

    def get_init_trans_vec(self, x):
        return self.regression_distn.initial_pi(x)

    def get_trans_matrix(self, x):
        return self.regression_distn.pi(x)
    
    def _align_lags(self,state_seqs, cov_seqs):
        prev_state = np.array(state_seqs[:-1, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        prev_state = np.concatenate((np.zeros((1, prev_state.shape[1], self.K)), prev_state), axis=0)

        # next_state = np.array(state_seqs[1:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        next_state = np.array(state_seqs[:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        # return np.dstack((prev_state, cov_seqs[1:])), next_state
        return np.dstack((prev_state, cov_seqs)), next_state

    def resample(self, state_seqs=None, cov_seqs=None, masks=None, omegas=None, active_K=None):
        datas = [self._align_lags(z, x) for z, x in zip(state_seqs, cov_seqs)]

        # Clip the last data column since it is redundant
        # and not expected by the MultinomialRegression
        datas = [(x, y[:, :, :-1]) for x, y in datas]
        self.regression_distn.resample(datas, masks=masks, omegas=omegas, active_K=active_K)

    @property
    def params(self):
        params = {
            'name': self.__class__.__name__, 
            'params': {
                'num_states': self.K, 
                'covariates_dim': self.covariates_dim,
                'regression_distn': self.regression_distn.params}}
        return params
    
class RegularizedHMMInputTransitions(HMMInputTransitions):
    _regression_distn = RegularizedMultinomialLogisticRegression
    
    
class HMMInputOnlyTransitions(HMMInputTransitions):
    
    _regression_distn = MultinomialLogisticRegression
    
    def __init__(self, num_states, covariates_dim, regression_distn=None, **kwargs):
        if regression_distn is None:
            # regression_distn = self._regression_distn(
            #     D_out=num_states, D_in=1+covariates_dim, input_only=True, **kwargs)
            regression_distn = self._regression_distn(
                D_out=num_states, D_in=covariates_dim, input_only=True, **kwargs)
        super(HMMInputOnlyTransitions, self).__init__(
            num_states=num_states, covariates_dim=covariates_dim, regression_distn=regression_distn)
        
    def get_init_trans_vec(self, x):
        # N, b = x.shape
        # x = np.concatenate((np.ones((N, 1)), x), axis=1)
        return self.regression_distn.initial_pi(x)

    def get_trans_matrix(self, x):
        # T, N, b = x.shape
        # x = np.concatenate((np.ones((T, N, 1)), x), axis=2)
        return self.regression_distn.pi(x)
    
    def _align_lags(self, state_seqs, cov_seqs):
        # next_state = np.array(state_seqs[1:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        # return cov_seqs[1:], next_state
        next_state = np.array(state_seqs[:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        return cov_seqs, next_state
        # prev_state = np.ones((state_seqs.shape[0]-1, state_seqs.shape[1], 1), dtype=np.int32)
        # next_state = np.array(state_seqs[1:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        # return np.dstack((prev_state, cov_seqs[1:])), next_state
    
    

class RegularizedHMMInputOnlyTransitions(HMMInputOnlyTransitions):
    _regression_distn = RegularizedMultinomialLogisticRegression


class StickBreakingHMMInputTransitions(HMMInputTransitions):
    
    _regression_distn = StickBreakingLogisticRegression
    
    @property
    def active_states(self):
        return self.regression_distn.active_K
    
    
class StickBreakingHMMInputOnlyTransitions(StickBreakingHMMInputTransitions):
    
    def __init__(self, num_states, covariates_dim, regression_distn=None, **kwargs):
        if regression_distn is None:
            regression_distn = self._regression_distn(
                D_out=num_states, D_in=covariates_dim, input_only=True, **kwargs)
        super(StickBreakingHMMInputOnlyTransitions, self).__init__(
            num_states=num_states, covariates_dim=covariates_dim, regression_distn=regression_distn)
        
    def get_init_trans_vec(self, x):
        return self.regression_distn.initial_pi(x)

    def get_trans_matrix(self, x):
        return self.regression_distn.pi(x)
    
    def _align_lags(self, state_seqs, cov_seqs):
        next_state = np.array(state_seqs[:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        return cov_seqs, next_state

        # next_state = np.array(state_seqs[1:, :, None] == np.arange(self.K)[None, :], dtype=np.int32)
        # return cov_seqs[1:], next_state


class RegularizedStickBreakingHMMInputOnlyTransitions(StickBreakingHMMInputOnlyTransitions):
    _regression_distn = RegularizedStickBreakingLogisticRegression


class RegularizedStickBreakingHMMInputTransitions(StickBreakingHMMInputTransitions):
    _regression_distn = RegularizedStickBreakingLogisticRegression







