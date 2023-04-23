# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:15:07 2022

@author: lisa
"""
import numpy as np
from models.utils.stats import sample_gaussian, sample_inv_wishart, \
    sample_inv_gamma
from scipy.linalg import lapack


class Regression(object):
    
    @property
    def D_out(self):
        return self.A.shape[0]
    
    @property
    def D_in(self):
        return self.A.shape[1]
    
    def _get_statistics(self, data):
        # print("data ", data)
        if isinstance(data, list):
            return sum((self._get_statistics(d) for d in data), self._empty_statistics())
        elif isinstance(data, tuple):
            x, y = data
            # print('data ', data)
            n, D = y.shape
            xxT, yxT, yyT = x.T.dot(x), y.T.dot(x), y.T.dot(y)
            return np.array([yyT, yxT, xxT, n], dtype=np.object)
        else:
            n, D = data.shape[0], self.D_in
            x, y = data[:, :D], data[:, D:]
            # print(x.shape)
            # print(y)
            xxT, yxT, yyT = x.T.dot(x), y.T.dot(x), y.T.dot(y)
            # statmat = data.T.dot(data)
            # print("statmat ", statmat)
            # xxT, yxT, yyT = statmat[:-D, :-D], statmat[-D:, :-D], statmat[-D:, -D:]
            return np.array([yyT, yxT, xxT, n], dtype=np.object)
        
    def _empty_statistics(self):
        D_in, D_out = self.D_in, self.D_out
        return np.array([np.zeros((D_out, D_out)), np.zeros((D_out, D_in)),
                         np.zeros((D_in, D_in)), 0], dtype=np.object)
    
    def predict(self, x):
        # print("x ", x[:5])
        y = x.dot(self.A.T)
        # print('pred ', y[:5])
        return y
    
    def rvs(self, x=None, size=1, return_xy=True):
        # print("sigma ", self.sigma)
        x = np.random.normal(size=(size, self.A.shape[1])) if x is None else x
        y = self.predict(x)
        y += np.random.normal(size=(x.shape[0], self.D_out)).dot(np.linalg.cholesky(self.sigma).T)
        return np.hstack((x, y)) if return_xy else y
    
    def log_likelihood(self, xy):
        # print("A ", self.A)
        A, sigma, D = self.A, self.sigma, self.D_out
        if isinstance(xy, tuple):
            x, y = xy
            dim = x.ndim
        else:
            dim = xy.ndim
            if dim == 2:
                x, y = xy[:, :self.D_in], xy[:, self.D_in:]
            elif dim == 3:
                x, y = xy[:, :, :self.D_in], xy[:, :, self.D_in:]
            
        L = np.linalg.cholesky(sigma)
        sigma_inv = lapack.dpotri(L, lower=True)[0]
        # print("sigma_inv ", sigma_inv)
        
        contract = 'ni,ni->n' if dim == 2 else 'tni,tni->tn' if dim == 3 else 'i,i->' 
        
        A_sigma_inv = A.T.dot(sigma_inv)
        A_sigma_inv_A = A_sigma_inv.dot(A)
        
        out = (-1. / 2) * np.einsum(contract, x.dot(A_sigma_inv_A), x)
        out += (-1. / 2) * np.einsum(contract, y.dot(sigma_inv), y)
        out += np.einsum(contract, x.dot(A_sigma_inv), y)
        
        out -= D/2*np.log(2*np.pi) + np.log(np.diag(L)).sum()
        return out
    
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 'params': {'A': self.A.tolist(), 'sigma': self.sigma.tolist()}}
        return params
    


class RegressionFixedCoefficients(Regression):
    
    def __init__(self, A, sigma=None, S_0=None, nu_0=None):
        self.A, self.sigma = A, sigma
        self.S_0, self.nu_0 = S_0, nu_0
        
        if sigma is None:
            self.resample()
            
    def resample(self, data=None):
        if data is None:
            self.sigma = sample_inv_wishart(self.S_0, self.nu_0)
        else:
            yyT, yxT, xxT, n = self._get_statistics(data)
            self._resample_sigma(yyT, yxT, xxT, n, self.A)
            
    def _resample_sigma(self, yyT, yxT, xxT, n, A):
        # print("xxT ", xxT)
        # print("yxT ", yxT)
        # print("yyT ", yyT)
        # print("n ", n)
        # print("A ", A)
        # print("S_0 ", self.S_0)
        S = self.S_0 + yyT - yxT.dot(A.T) - A.dot(yxT.T) + A.dot(xxT).dot(A.T)
        # print("S ", S)
        nu = self.nu_0 + n
        self.sigma = sample_inv_wishart(S, nu)
        # print("self.sigma in regression", self.sigma)
        
    @property
    def params(self):
        params = super(RegressionFixedCoefficients, self).params
        params['params']['S_0'] = self.S_0.tolist()
        params['params']['nu_0'] = self.nu_0
        return params

    
class DiagonalRegressionFixedCoefficients(RegressionFixedCoefficients):
    
    def __init__(self, A, sigma=None, alpha_0=None, beta_0=None):
        self.A = A
        self.sigma_sqrt_flat = np.diagonal(sigma) if sigma is not None else None
    
        self.alpha_0, self.beta_0 = alpha_0, beta_0
        if self.sigma_sqrt_flat is None:
            self.resample()  # initialise from prior
            
    @property
    def sigma(self):
        return np.diag(self.sigma_sqrt_flat)
    
    def log_likelihood(self, xy, masks=None):
        if isinstance(xy, tuple):
            x, y = xy
            dim = y.ndim
        else:
            dim = xy.ndim
            if dim == 2:
                x, y = xy[:, :self.D_in], xy[:, self.D_in:]
            elif dim == 3:
                x, y = xy[:, :, :self.D_in], xy[:, :, self.D_in:]
        
        contract = 'tnj,ij->tni' if dim == 3 else 'nj,ij->ni'
        Ax = np.einsum(contract, x, self.A)
        
        sq_err = -0.5 * (y - Ax) ** 2 
        ll = np.sum(sq_err / self.sigma_sqrt_flat, axis=dim-1)
        ll += (-0.5 * np.log(2 * np.pi * self.sigma_sqrt_flat)).sum()

        return ll
        
            
    def resample(self, data=None):
        if data is None:
            self.sigma_sqrt_flat = np.reshape(sample_inv_gamma(self.alpha_0, self.beta_0), (self.D_out,))
        else:
            yyT, yxT, xxT, n = self._get_statistics(data)
            self._resample_sigma(xxT, yxT, yyT, n, self.A)
                
    def _resample_sigma(self, xxT, yxT, yyT, n, A):
            
        AAT = np.array([np.outer(a,a) for a in A])
        
        alpha = self.alpha_0 + n / 2.0
        beta = self.beta_0 + 0.5 * np.diagonal(yyT)
        beta += -1.0 * np.sum(yxT * A, axis=1)
        beta += 0.5 * np.sum(AAT * xxT, axis=(1,2))

        self.sigma_sqrt_flat = np.reshape(sample_inv_gamma(alpha, beta), (self.D_out,))
        # print("self.sigma_sqrt_flat  ", self.sigma_sqrt_flat )
        
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {'A': self.A.tolist(), 'sigma': self.sigma.tolist(),
                             'alpha_0': self.alpha_0, 'beta_0':self.beta_0.tolist()}}
        return params
    
        


class RegressionNonConj(Regression):

    def __init__(self, A_0=None, sigma_0=None, nu_0=None, S_0=None, A=None, 
                 sigma=None, n_iter=3):
        self.A, self.sigma = A, sigma

        self.h_0 = np.linalg.solve(sigma_0, A_0.ravel()).reshape(A_0.shape)
        self.J_0 = np.linalg.inv(sigma_0)
        self.A_0, self.sigma_0 = A_0, sigma_0
        self.nu_0, self.S_0 = nu_0, S_0

        self.n_iter = n_iter

        if A is sigma is None:
            self.resample()

    def resample(self, data=None, n_iter=None):
        n_iter = n_iter if n_iter else self.n_iter
        if data is None:
            self.A = np.random.multivariate_normal(self.A_0.ravel(), self.sigma_0).reshape(self.A_0.shape)
            self.sigma = sample_inv_wishart(self.S_0, self.nu_0)
        else:
            yyT, yxT, xxT, n = self._get_statistics(data)
            for itr in range(n_iter):
                self._resample_A(xxT, yxT, self.sigma)
                self._resample_sigma(xxT, yxT, yyT, n, self.A)
        # print("A ", self.A)
        # print("sigma ", self.sigma)

    def _resample_A(self, xxT, yxT, sigma):
        sigma_inv = np.linalg.inv(sigma)
        J = self.J_0 + np.kron(sigma_inv, xxT)
        h = self.h_0 + sigma_inv.dot(yxT)
        self.A = sample_gaussian(J=J, h=h.ravel()).reshape(h.shape)
        
    def _resample_sigma(self, xxT, yxT, yyT, n, A):
        S = self.S_0 + yyT - yxT.dot(A.T) - A.dot(yxT.T) + A.dot(xxT).dot(A.T)
        nu = self.nu_0 + n
        self.sigma = sample_inv_wishart(S, nu)
    
    @property
    def params(self):
        params = super(RegressionNonConj, self).params
        params['params']['nu_0'] = self.nu_0
        params['params']['S_0'] = self.S_0.tolist()
        params['params']['A_0'] = self.A_0.tolist()
        params['params']['sigma_0'] = self.sigma_0.tolist()
        return params
        


class DiagonalRegression(RegressionNonConj):
    
    def __init__(self, D_out, D_in, mu_0=None, Sigma_0=None, alpha_0=None,
                 beta_0=None, A=None, sigmasqrt=None, constant=True, nIter=1):

        self._D_out = D_out     # number of seperate observations
        self._D_in = D_in + 1 if constant else D_in
        self.A = A
        self.sigmasqrt_flat = sigmasqrt
        self.constant = constant # We do not yet support affine
    
        mu_0 = np.zeros(self._D_in) if mu_0 is None else mu_0
        Sigma_0 = np.eye(self._D_in) if Sigma_0 is None else Sigma_0

        assert mu_0.shape == (self._D_in,)
        assert Sigma_0.shape == (self._D_in, self._D_in)
        
        self.h_0 = np.linalg.solve(Sigma_0, mu_0)
        self.J_0 = np.linalg.inv(Sigma_0)
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
    
        self.nIter = nIter
    
        if A is sigmasqrt is None:
            self.A = np.zeros((D_out, self._D_in))
            self.sigmasqrt_flat = np.ones((D_out,))
            self.resample()  # initialise from prior
            
    @property
    def D_out(self):
        return self._D_out
    
    @property
    def D_in(self):
        return self._D_in
    
    @property
    def sigma(self):
        return np.diag(self.sigmasqrt_flat)
    
    def _get_statistics(self, data, D_in=None, D_out=None):

        D_out = self.D_out if D_out is None else D_out
        D_in = self.D_in if D_in is None else D_in
        
        if not isinstance(data, list):
            datas = [data]
        else:
            datas = data
        
        ysqrt = np.zeros(D_out)
        yxT = np.zeros((D_out, D_in))
        xxT = np.zeros((D_out, D_in, D_in))
        n = np.zeros(D_out)
        
        for data in datas:
            
            if self.constant:
                data = np.c_[np.ones(data.shape[0]), data]
            
            if isinstance(data, tuple):
                x, y = data
            else:
                x, y = data[:, :D_in], data[:, D_in:]
                
            assert x.shape[1] == D_in
            assert y.shape[1] == D_out
                
            ysqrt += np.sum(y**2, axis=0)
            yxT += y.T.dot(x)
            xxT += x.T.dot(x)
            n += y.shape[0]
            
        return ysqrt, yxT, xxT, n
    
    def resample(self, data=None, nIter=None):

        nIter = nIter if nIter else self.nIter
        if data is None:
            self.A = np.reshape(sample_gaussian(J=self.J_0, h=self.h_0), (self.D_out, self.D_in))
            # print(self.A)
            self.sigmasqrt_flat = np.reshape(sample_inv_gamma(self.alpha_0, self.beta_0), (self.D_out,))
            # print("self.sigmasqrt_flat ", self.sigmasqrt_flat)
        else:
            yyT, yxT, xxT, n = self._get_statistics(data)
            for itr in range(nIter):
                self._resample_A(xxT, yxT, self.sigmasqrt_flat)
                self._resample_sigma(xxT, yxT, yyT, n, self.A)
                
    def _resample_A(self, xxT, yxT, sigmasqrt_flat):
        for d in range(self.D_out):
            J = self.J_0 + xxT[d] / sigmasqrt_flat[d]
            h = self.h_0 + yxT[d] / sigmasqrt_flat[d]
            self.A[d] = sample_gaussian(J=J, h=h)
            # print("self.A[d] ", self.A[d])
            
    def _resample_sigma(self, xxT, yxT, yyT, n, A):
        
        AAT = np.array([np.outer(a,a) for a in A])
        
        alpha = self.alpha_0 + n / 2.0
        beta = self.beta_0 + 0.5 * yyT
        beta += -1.0 * np.sum(yxT * A, axis=1)
        beta += 0.5 * np.sum(AAT * xxT, axis=(1,2))
    
        self.sigmasqrt_flat = np.reshape(sample_inv_gamma(alpha, beta), (self.D_out,))
        # print("self.sigmasqrt_flat  ", self.sigmasqrt_flat )
        
    def log_likelihood(self, xy):
                
        if isinstance(xy, tuple):
            x, y = xy
            if self.constant:
                x = np.c_[np.ones(x.shape[0]), x]
        else:
            if self.constant:
                xy = np.c_[np.ones(xy.shape[0]), xy]
            x, y = xy[:, :self.D_in], xy[:, self.D_in:]
            
        sqrterr = -0.5 * (y - x.dot(self.A.T)) ** 2
        ll = np.sum(sqrterr / self.sigmasqrt_flat, axis=1)
        ll += -0.5*np.log(2*np.pi*self.sigmasqrt_flat)
        
        return ll