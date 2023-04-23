# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:11:30 2022

@author: lisa
"""
import numpy as np
from polyagamma import random_polyagamma
from models.utils.stats import sample_gaussian, logistic, sample_inv_gamma, sample_generalized_inv_gamma
from scipy.special import softmax, expit
import scipy.linalg as linalg
from scipy.stats import geninvgauss
import time


class PGLogisticRegression(object):

    def __init__(self, D_in, D_out, A=None, mu_A=None, sigma_A=None, input_only=True):
        self.D_in, self.D_out = D_in, D_out
        self.mu_0, self.sigma_0 = mu_A, sigma_A
        
        if np.isscalar(mu_A):
            self.mu_0 = mu_A * np.ones((D_out, D_in))
        # else:
        #     assert mu_A.shape == (D_out, D_in)

        if np.isscalar(sigma_A):
            self.sigma_0 = np.array([sigma_A * np.eye(D_in) for _ in range(D_out)])
        # else:
        #     assert sigma_A.shape == (D_out, D_in, D_in)

        if A is not None:
            # assert A.shape == (D_out, D_in)
            self.A = A
        else:
            self.A = np.empty((D_out, D_in))
            for k in range(self.D_out):
                self.A[k] = np.random.multivariate_normal(self.mu_0[k], self.sigma_0[k])


    def a_func(self, data):
        raise NotImplementedError

    def b_func(self, data):
        raise NotImplementedError
    
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {'A': self.A.tolist(), 'D_in': self.D_in, 'D_out': self.D_out,
                             'mu_A':self.mu_0.tolist(), 'sigma_A':self.sigma_0.tolist()}}
        return params

    def kappa_func(self, data, omega, c):
        return self.a_func(data) - self.b_func(data) / 2

    def _resample_auxiliary_variables(self, datas):
        A = self.A  # A -> [[Kx(K+b)]]
        omegas, cs = [], []
        for data in datas:

            if isinstance(data, tuple):
                x, y = data
            else:
                x, y = data[:, :, :self.D_in], data[:, :, self.D_in:]

            b = self.b_func(y)
            psi = np.einsum('tnj,kj->tnk', x, A)

            c = self.compute_cs(psi)
            psi -= c  # psi -> [T x N x (K-1)]
            # print("psi ", psi[0,0])

            b_zero_idxs = b > 0
            omega = np.zeros_like(psi)

            omega[b_zero_idxs] = random_polyagamma(b[b_zero_idxs], psi[b_zero_idxs])    # Nan values will be replaced by 0
            # print("omega ", omega[:, 0])
            omegas.append(omega)
            cs.append(c)

        return omegas, cs

    @staticmethod
    def compute_cs(psi):
        return np.zeros_like(psi)

    def resample(self, data, masks=None, omegas=None, cs=None, active_K=None):
        
        # print("active_K ", active_K)
        active_K = active_K if active_K is not None else self.D_out
        self.active_K = min(active_K, self.D_out)

        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2, \
                "datas must be an (x,y) tuple or a list of such tuples"
            data = [data]

        if masks is None:
            masks = [np.ones((y.shape[0], y.shape[1]), dtype=bool) for x, y in data]

        if omegas is None or cs is None:
            omegas, cs = self._resample_auxiliary_variables(data)
        
        D = self.D_in
        for k in range(self.D_out):

            prior_J = linalg.inv(self.sigma_0[k])
            prior_h = prior_J.dot(self.mu_0[k])

            lkhd_J = np.zeros((D, D))
            lkhd_h = np.zeros(D)
            
            if k < self.active_K:
                for d, m, o, c in zip(data, masks, omegas, cs):
                    if isinstance(d, tuple):
                        x, y = d
                    else:
                        x, y = d[:, :, :D], d[:, :, D:]
                    
                    x[~m] = 0.0
                    h = self.kappa_func(y, o, c)
                    a = x * o[:, :, k, None]
    
                    lkhd_J += np.einsum('tnj,tnk->jk', a, x)
                    lkhd_h += np.einsum('tn,tnr->r', h[:, :, k], x)
            
            # print("lkhd_h ", lkhd_h)
            # print("lkhd_J ", lkhd_J)
            post_h = prior_h + lkhd_h
            post_J = prior_J + lkhd_J

            self.A[k, :] = sample_gaussian(J=post_J, h=post_h)
        # print("self.A in logistic regression ", self.A)

            
class RegularizedPGLogisticRegression(PGLogisticRegression):
    
    def __init__(self, D_in, D_out, A=None, mu_A=None, sigma_A=None, eta=None,
                 xi=None, delta_sq=None, phi_sq=None, input_only=True):
        
        self.input_only = input_only
        self.c = 1
        self.eta = eta if eta is not None else np.array([sample_inv_gamma((1/2), np.full(D_in, self.c)) for _ in range(D_out)])
        self.xi = xi if xi is not None else sample_inv_gamma((1/2), self.c)
        self.delta_sq =  delta_sq if delta_sq is not None else np.array([sample_inv_gamma((1/2), 1.0 / self.eta[k]) for k in range(D_out)])
        self.phi_sq = phi_sq if phi_sq is not None else sample_inv_gamma((1/2), 1.0 / self.xi) 
        
        # sigma_A = np.zeros((D_out, D_in, D_in))
        # for k in range(D_out):
        #     print(self.delta_sq[k] * self.phi_sq)
        #     sigma_A[k] = np.diag(self.delta_sq[k] * self.phi_sq)
        
        PGLogisticRegression.__init__(self, 
            D_in=D_in, D_out=D_out, sigma_A=sigma_A, mu_A=mu_A, A=A)
        
    @property
    def K(self):
        return self.D_out 
        
    @property
    def params(self):
        params = super(RegularizedPGLogisticRegression, self).params
        params['params']['eta'] = self.eta.tolist()
        params['params']['xi'] = self.xi
        params['params']['delta_sq'] = self.delta_sq.tolist()
        params['params']['phi_sq'] = self.phi_sq
        return params
        
    def _resample_delta_sq(self):
        for k in range(self.D_out):
            beta = 1. / self.eta[k] + (1 / (2 * self.phi_sq)) * (self.A[k] ** 2)
            self.delta_sq[k] = sample_inv_gamma(1/2, beta)
        
        if self.input_only:
            pass
            # self.delta_sq[:, 0] = 0.1
        else:
            # 0.0001 werkt niet -> loopt vast! 0.001 werkt
            self.delta_sq[:, :self.K] = 0.1
        
    # def _resample_phi(self):
    #     alpha = self.c0 + (self.D_out * self.D_in) / 2.0
    #     sum_A = np.sum((self.A ** 2) / (4 * self.delta_sq))
    #     self.phi = np.random.gamma(alpha, 1. / (self.c1 + sum_A))
        # print('self.phi ', self.phi)
        
    def _resample_phi_sq(self):
        if self.input_only:
            alpha = ((self.D_in-1) * self.D_out + 1) / 2.0
            sum_A = (1/2) * np.sum(self.A[:, 1:] ** 2 / self.delta_sq[:, 1:])
        else:
            alpha = ((self.D_in-self.K) * self.D_out + 1) / 2.0
            sum_A = (1/2) * np.sum(self.A[:, self.K:] ** 2 / self.delta_sq[:, self.K:])
        beta = 1. / self.xi + sum_A 
        self.phi_sq = sample_inv_gamma(alpha, beta)
        
    def _resample_eta(self):
        for k in range(self.D_out):
            beta = self.c + 1. / self.delta_sq[k]
            self.eta[k] = sample_inv_gamma(1, beta)
    
    def _resample_xi(self):
        beta = self.c + 1. / self.phi_sq
        self.xi = sample_inv_gamma(1, beta)
        
    def resample(self, data, masks=None, omegas=None, cs=None, active_K=None):
        
        active_K = active_K if active_K is not None else self.D_out
        self.active_K = min(active_K, self.D_out)
        # print('active_K ', active_K)
        
        if not isinstance(data, list):
            assert isinstance(data, tuple) and len(data) == 2, \
                "datas must be an (x,y) tuple or a list of such tuples"
            data = [data]

        if masks is None:
            masks = [np.ones((y.shape[0], y.shape[1]), dtype=bool) for x, y in data]

        if omegas is None or cs is None:
            omegas, cs = self._resample_auxiliary_variables(data)
        
        D = self.D_in
        for k in range(self.D_out):
            
            lkhd_J = np.zeros((D, D))
            lkhd_h = np.zeros(D)
            
            if k < self.active_K:
                for d, m, o, c in zip(data, masks, omegas, cs):
                    if isinstance(d, tuple):
                        x, y = d
                    else:
                        x, y = d[:, :, :D], d[:, :, D:]
                    
                    x[~m] = 0.0
                    h = self.kappa_func(y, o, c)
                    a = x * o[:, :, k, None]
    
                    lkhd_J += np.einsum('tnj,tnk->jk', a, x)
                    lkhd_h += np.einsum('tn,tnr->r', h[:, :, k], x)

            delta = 1. / self.delta_sq[k] 
            if self.input_only:
                # delta[1:] *= 1. /self.phi_sq
                delta *= 1. /self.phi_sq
            else:
                delta[self.K:] *= 1. /self.phi_sq
            
            post_h = lkhd_h
            post_J = lkhd_J + np.diag(delta) 
            
            self.A[k, :] = sample_gaussian(J=post_J, h=post_h)
        
        # print("self.A in logistic regression ", self.A)
        self._resample_delta_sq()
        self._resample_phi_sq()
        self._resample_eta()
        self._resample_xi()
        


class StickBreakingLogisticRegression(PGLogisticRegression):

    def __init__(self, D_in, D_out, active_states=1, **kwargs):
        PGLogisticRegression.__init__(self, D_in=D_in, D_out=D_out-1, **kwargs)
        self.active_K = active_states

    def a_func(self, data):
        return data

    def b_func(self, data):
        b = np.ones_like(data) - np.cumsum(data, axis=2) + data
        return b
    
    @property
    def K(self):
        return self.D_out + 1
    
    @property
    def params(self):
        params = super(StickBreakingLogisticRegression, self).params
        params['params']['D_out'] +=  1
        params['params']['active_states'] = int(self.active_K)
        return params

    def initial_pi(self, x):
        N, b = x.shape

        W = self.A
        W_markov = W[:, :-b]
        W_covs = W[:, -b:]

        psi_X = np.einsum('nb,kb->nk', x, W_covs)
        psi_Z = W_markov.T # K x (K-1)
        
        trans_psi = psi_X
        # if psi_Z.shape[0] > 0:
        #     trans_psi += np.diagonal(psi_Z)

        pi = self._psi_to_pi(trans_psi, self.active_K)

        # assert np.allclose(pi.sum(axis=-1), 1.0)
        return pi
    
    @staticmethod
    def _psi_to_pi(psi, active_K):
        
        if psi.ndim == 2:
            N, Km = psi.shape
            K = Km + 1
            pi = np.zeros((N, K))
            stick = np.ones(N)
            for k in range(active_K):
                pi[:, k] = expit(psi[:, k]) * stick
                stick -= pi[:, k]
            pi[:, active_K] = stick
        elif psi.ndim == 3:
            # print("psi ", psi[:, 0])
            T, N, Km = psi.shape
            K = Km + 1
            pi = np.zeros((T, N, K))
            stick = np.ones((T, N))
            for k in range(active_K):
                pi[:, :, k] = expit(psi[:, :, k]) * stick
                stick -= pi[:, :, k]
            pi[:, :, active_K] = stick
        elif psi.ndim == 4:
            T, N, M, Km = psi.shape
            K = Km + 1
            pi = np.zeros((T, N, K, K))
            stick = np.ones((T, N, K))
            for k in range(active_K):
                pi[:, :, :, k] = expit(psi[:, :, :, k]) * stick
                stick -= pi[:, :, :, k]
            pi[:, :, :, active_K] = stick

        else:
            raise ValueError('psi must be 2D or 3D or 4D')

        return pi

    def pi(self, x):
        T, N, b = x.shape

        W = self.A
        # print("W ", W)
        W_markov = W[:, :-b]
        W_covs = W[:, -b:]
        
        psi_X = np.einsum('tnb,kb->tnk', x, W_covs)
        psi_Z = W_markov.T  # [K x (K-1)]
        
        if psi_Z.shape[0] == 0:
            T, N, K = psi_X.shape
            pi = self._psi_to_pi(psi_X, self.active_K)
            pi = np.broadcast_to(pi[..., None], pi.shape + (K+1,))
            pi = np.transpose(pi, (0, 1, 3, 2)).copy()
 
        else:
            trans_psi = psi_X[:, :, None, :] + psi_Z
            pi = self._psi_to_pi(trans_psi, self.active_K)
        pi[:, :, self.active_K+1:] = 0.
        # assert np.allclose(pi.sum(axis=-1), 1.0)
        return pi

    def rvs(self, x=None, size=None, return_xy=False):
        if x is None:
            assert isinstance(size, int)
            x = np.random.randn(size, self.D_in)

        pi = self.pi(x)
        if pi.ndim == 1:
            y = np.random.multinomial(1, pi)
        elif pi.ndim == 2:
            y = np.array([np.random.multinomial(1, pp) for pp in pi])
        else:
            raise NotImplementedError
        return (x, y) if return_xy else y
    

class RegularizedStickBreakingLogisticRegression(
        RegularizedPGLogisticRegression, StickBreakingLogisticRegression):
    
    def __init__(self, D_in, D_out, active_states=1, **kwargs):
        RegularizedPGLogisticRegression.__init__(self, D_in=D_in, D_out=D_out-1, **kwargs)
        self.active_K = active_states


class MultinomialLogisticRegression(PGLogisticRegression):

    def __init__(self, D_in, D_out, **kwargs):
        PGLogisticRegression.__init__(self, D_in=D_in, D_out=D_out-1, **kwargs)
    
    @property
    def params(self):
        params = super(MultinomialLogisticRegression, self).params
        params['params']['D_out'] +=  1
        return params
    
    @property
    def K(self):
        return self.D_out + 1

    @staticmethod
    def compute_cs(psi):
        c = np.zeros_like(psi)
        max_psi = np.max(psi)
        for k in range(psi.shape[2]):
            sum_1 = np.einsum('tnj->tn', np.exp(psi[:, :, :k] - max_psi))
            sum_2 = np.einsum('tnj->tn', np.exp(psi[:, :, k+1:] - max_psi))
            # sum_1 = np.einsum('tnj->tn', np.exp(psi[:, :, :k]))
            # sum_2 = np.einsum('tnj->tn', np.exp(psi[:, :, k+1:]))

            # c[:, :, k] = np.log(sum_1 + sum_2 + 1)
            sum_3 = np.exp(0 - max_psi) # todo moet deze erbij?
            # c[:, :, k] = np.log(sum_1 + sum_2) + max_psi
            c[:, :, k] = np.log(sum_1 + sum_2 + sum_3) + max_psi
        # print('c ', c[0,0])
        return c

    def kappa_func(self, data, omega, c):
        kappa = self.a_func(data) - self.b_func(data) / 2
        # minus or plus? -> plus
        # print("omega ", omega[0,0])
        # print("c ", c)
        # kappa = np.expand_dims(kappa, axis=3) + omega * c
        kappa += omega * c
        # kappa -= omega * c
        # print("kappa ", kappa[0,0])
        return kappa

    def a_func(self, data):
        return data

    def b_func(self, data):
        # assert data.shape[2] == self.D_out - 1
        return np.ones_like(data, dtype=np.float)

    def c_func(self, data):
        return 0

    def rvs(self, x=None, size=None, return_xy=False):
        if x is None:
            assert isinstance(size, int)
            x = np.random.randn(size, self.D_in)
        else:
            assert x.ndim == 2 and x.shape[1] == self.D_in

        N = x.shape[0]
        psi = x.dot(self.A.T)
        p = np.exp(psi)
        p /= np.sum(p, axis=1)[:, None]
        y = np.empty((N, 1))
        for i_y in range(N):
            y[i_y] = np.random.choice(np.arange(self.D_out), p=p[i_y])
        return (x, y) if return_xy else y

    def initial_pi(self, x):
        N, b = x.shape
        # print(x[0])

        W = self.A
        W_markov = W[:, :-b]
        W_covs = W[:, -b:]

        psi_X = x.dot(W_covs.T)  # N x (K-1)
        psi_Z = W_markov.T # K x (K-1)
        
        trans_psi = psi_X
        # if psi_Z.shape[0] > 0:
        #     trans_psi += np.diagonal(psi_Z)
        
        trans_psi = np.concatenate((trans_psi, np.zeros((N, 1))), axis=1)
        
        N, K = trans_psi.shape
        pi = softmax(trans_psi, axis=1)
        # pi[np.isinf(pi)] = 1.0 / K
        # print("pi ", pi[0])
        return pi

    def pi(self, x):

        T, N, b = x.shape
        # print(x[:, 0])
        W = self.A
        
        W_markov = W[:, :-b]
        # print('W_markov ', W_markov.shape)
        W_covs = W[:, -b:]
        # print('W_covs ', W_covs.shape)

        # psi_X = x.dot(W_covs.T)  # [T x N x b] x [b x (K-1)] -> [T x N x (K-1)]
        psi_X = np.einsum('tnb,kb->tnk', x, W_covs)
        psi_Z = W_markov.T  # [K x (K-1)]
        
        if psi_Z.shape[0] == 0:
            T, N, K = psi_X.shape
            # print("K ", K)
            trans_psi = np.concatenate((psi_X, np.zeros((T, N, 1))), axis=2)
            # print(trans_psi[0,0])
            pi = softmax(trans_psi, axis=2)
            # print(pi[0, 0])
            pi = np.broadcast_to(pi[..., None], pi.shape + (K+1,))
            pi = np.transpose(pi, (0, 1, 3, 2))
            # pi = np.full((pi.shape + (K+1,)), pi)
        else:

            trans_psi = psi_X[:, :, None, :] + psi_Z
            T, N, K, _ = trans_psi.shape  # T x N x K x (K-1)
            trans_psi = np.concatenate((trans_psi, np.zeros((T, N, K, 1))), axis=3)
            pi = softmax(trans_psi, axis=3)

        # print(trans_psi[0,0])

        # TODO RuntimeWarning: overflow encountered in exp pi = np.exp(trans_psi)
        # pi[np.isinf(pi)] = 1.0 / K
        # print("pi ", pi[0, 0])
        # print("pi ", pi.shape)
        return pi
    

class RegularizedMultinomialLogisticRegression(
        RegularizedPGLogisticRegression, MultinomialLogisticRegression):
    
    def __init__(self, D_in, D_out, **kwargs):
        RegularizedPGLogisticRegression.__init__(self, D_in=D_in, D_out=D_out-1, **kwargs)
    