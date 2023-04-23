# -*- coding: utf-8 -*-
import numpy as np
from models.utils.stats import sample_inv_wishart, sample_inv_gamma
from scipy.linalg import lapack


class GaussianNIW(object):
    
    _name = 'GaussianNIW'
    
    def __init__(self, mu=None, sigma=None, mu_0=None, kappa_0=None, 
                 lambda_0=None, nu_0=None):
        self.mu, self.sigma = mu, sigma
        self.mu_0, self.kappa_0 = mu_0, kappa_0
        self.lambda_0, self.nu_0 = lambda_0, nu_0
        
        if None in (mu, sigma):
            self.resample()
        
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        self._sigma = sigma
        self._sigma_chol = None
        
    @property
    def sigma_chol(self):
        if not hasattr(self, '_sigma_chol') or self._sigma_chol is None:
            self._sigma_chol = np.linalg.cholesky(self.sigma)
        return self._sigma_chol
        
    def rvs(self, size=None):
        size = 1 if size is None else size
        size = size + (self.mu.shape[0],) if isinstance(size, tuple) else (size, self.mu.shape[0])
        return self.mu + np.random.normal(size=size).dot(self.sigma_chol.T)
        
    def log_likelihood(self, x):
        mu, sigma, D = self.mu, self.sigma, self.D
        L = np.linalg.cholesky(sigma)
        sigma_inv = lapack.dpotri(L, lower=True)[0]
        
        contract = 'ni,ni->n' if x.ndim == 2 else 'i,i->'
        out = (-1. / 2.) * np.einsum(contract, mu.dot(sigma_inv), mu)
        out += (-1. / 2.) * np.einsum(contract, x.dot(sigma_inv), x)
        out += np.einsum(contract, mu.dot(sigma_inv), x)
        
        out -= D/2.0 * np.log(2*np.pi) + np.log(np.diag(L)).sum()
        return out
    
    @property
    def D(self):
        if self.mu is not None:
            return self.mu.shape[0]
        elif self.mu_0 is not None:
            return self.mu_0.shape[0]
        
    def _get_statistics(self, data, D=None):
    
        if isinstance(data, np.ndarray):
            n = data.shape[0]
            y_bar = data.mean(0)
            centered = data - y_bar
            sum_sqc = centered.dot(centered)
        elif isinstance(data, list):
            n = sum(d.shape[0] for d in data)
            y_bar = sum(d.sum(0) for d in data) / n
            sum_sqc = sum(((d - y_bar).T).dot(d-y_bar) for d in data)
        else:
            n = 0
        
        if n == 0:
            y_bar = np.zeros(self.D)
            sum_sqc = np.zeros((self.D, self.D))
        return n, y_bar, sum_sqc
        
    def resample(self, data=None):
        
        if data is None:
            self.sigma = sample_inv_wishart(self.lambda_0, self.nu_0)
            self.mu = np.random.multivariate_normal(self.mu_0, self.sigma / self.kappa_0)
        else:
            n, y_bar, sum_sqc = self._get_statistics(data)
            self._resample_sigma(n, y_bar, sum_sqc)
            self._resample_mu(n, y_bar)
            
    def _resample_sigma(self, n, y_bar, sum_sqc):
        nu_n = self.nu_0 + n
        cent_mu_0 = (y_bar - self.mu_0).reshape(1, -1)
        lambda_n = self.lambda_0 + sum_sqc +  (self.kappa_0*n / (self.kappa_0 + n)) * cent_mu_0.T.dot(cent_mu_0)
        self.sigma = sample_inv_wishart(lambda_n, nu_n)
        # print("self.sigma ", self.sigma)

    def _resample_mu(self, n, y_bar):
        mu_n = (self.kappa_0 / (self.kappa_0 + n)) * self.mu_0 + (n / (self.kappa_0  + n)) * y_bar
        kappa_n = self.kappa_0 + n
        self.mu = np.random.multivariate_normal(mu_n, self.sigma / kappa_n)
        # print("self.mu ", self.mu)
        
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {'mu': self.mu.tolist(), 'sigma': self.sigma.tolist(),
                             'mu_0': self.mu_0.tolist(), 'kappa_0': self.kappa_0,
                             'nu_0': self.nu_0, 'lambda_0': self.lambda_0.tolist()}}
        return params
    
        
    


class DiagonalGaussianNonConjNIG(object):
    
    def __init__(self, mu=None, sigmas=None, mu_0=None, sigmas_0=None, alpha_0=None,
                 beta_0=None, n_iter=2):
        
        self.mu, self.sigmas = mu, sigmas
        self.mu_0, self.sigmas_0 = mu_0, sigmas_0
        self.alpha_0, self.beta_0 = alpha_0, beta_0
        self.n_iter = n_iter
        
        if mu is sigmas is None:
            self.resample()
            
    @property
    def D(self):
        if self.mu is not None:
            return self.mu.shape[0]
        elif self.mu_0 is not None:
            return self.mu_0.shape[0]
            
    @property
    def sigma(self):
        return np.diag(self.sigmas)

    def rvs(self, size=None):
        size = 1 if size is None else size
        return np.sqrt(self.sigmas) * np.random.normal(size=(size, self.D)) + self.mu
    
    def log_likelihood(self, x):
        Js = -1. / (2 * self.sigmas)
        x = np.reshape(x, (-1, self.D))
        ll = np.einsum('ni,ni,i->n', x, x, Js)
        ll -= np.einsum('ni,i,i->n', x, self.mu * 2, Js)
        ll += ((self.mu ** 2) * Js - (1./2 * np.log(2 * np.pi * self.sigmas))).sum()
        return ll
            
    def resample(self, data=None):
        
        if data is None:
            self.mu = np.sqrt(self.sigmas_0) * np.random.randn(self.D) + self.mu_0
            self.sigmas = 1./np.random.gamma(self.alpha_0, scale=1/self.beta_0)
        else:
            n, y_sum, y_sqrt = self._get_statistics(data)
            for itr in range(self.n_iter):
                self._resample_mu(n, y_sum, self.sigmas)
                self._resample_sigma(n, y_sum, y_sqrt, self.mu)
        # print("mu ", self.mu)
        # print("sigmas ", self.sigmas)
                
    def _resample_mu(self, n, y, sigmas):
        sigmas_n = 1./(1./self.sigmas_0 + n / sigmas)
        mu_n = (self.mu_0 / self.sigmas_0 + y / sigmas) * sigmas_n
        self.mu = np.sqrt(sigmas_n) * np.random.randn(self.D) + mu_n
        
    def _resample_sigma(self, n, y, y_sqrt, mu):
        alpha_n = self.alpha_0 + n / 2.
        beta_n = self.beta_0 + (1./2.) * (y_sqrt + (n * (mu ** 2)) - 2 * mu * y)
        self.sigmas = sample_inv_gamma(alpha_n, beta_n)
        
    def _get_statistics(self, data):
        if isinstance(data, np.ndarray):
            n = data.shape[0]
            y_sum = np.einsum('ni->i', data)
            y_sqrt = np.einsum('ni,ni->i', data, data)
        elif isinstance(data, list):
            n = sum(d.shape[0] for d in data)
            y_sum = sum(d.sum(0) for d in data)
            y_sqrt = sum((d ** 2).sum(0) for d in data)
        else:
            n = 0
            
        if n == 0:
            y_sum = np.zeros(self.D)
            y_sqrt = np.zeros(self.D)
        return n, y_sum, y_sqrt
    
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {'mu': self.mu.tolist(), 'sigmas': self.sigmas.tolist(),
                             'mu_0': self.mu_0.tolist(), 'sigmas_0': self.sigmas_0.tolist(),
                             'alpha_0': self.alpha_0, 'beta_0': self.beta_0.tolist()}}
        return params
    
    
class IsotropicGaussian(DiagonalGaussianNonConjNIG):
    
    _name = 'IsotropicGaussian'
    
    def __init__(self, mu=None, sigmas=None, mu_0=None, nu_0=None, alpha_0=None,
                  beta_0=None):
        
        self.mu, self.sigmas = mu, sigmas
        self.mu_0, self.nu_0 = mu_0, nu_0
        self.alpha_0, self.beta_0 = alpha_0, beta_0
        
        if None in (mu, sigmas):
            self.resample()
    
    def _get_statistics(self, data):
        if isinstance(data, np.ndarray):
            n = data.shape[0]
            y_bar = np.nanmean(data, axis=0)
            centered = data - y_bar
            sum_sqc = np.nansum(centered ** 2, axis=0)
        elif isinstance(data, list):
            n = sum(d.shape[0] for d in data)
            y_bar = sum(np.nansum(d, axis=0) for d in data) / n
            sum_sqc = sum(np.nansum(((d - y_bar) ** 2), axis=0) for d in data)
        else:
            n = 0
        
        if n == 0:
            y_bar = np.zeros(self.D)
            sum_sqc = np.zeros(self.D)
        # print("y_bar ", y_bar)
        print("n ", n)
        return n, y_bar, sum_sqc
    
    def _posterior_hypparams(self, n, y_bar, sum_sq):
        D = self.mu_0.shape[0]
        nu_n = D*n + self.nu_0
        alpha_n = self.alpha_0 + D*n / 2
        beta_n = self.beta_0 + (1/2) * sum_sq + (n * D * self.nu_0) / (n*D + self.nu_0) * (1/2) * ((y_bar - self.mu_0) ** 2).sum()
        mu_n = (n*y_bar + self.nu_0 * self.mu_0) / (n + self.nu_0)
        return mu_n, nu_n, alpha_n, beta_n
    
    def resample(self, data=None):
        if data is None:
            self.sigmas = 1./np.random.gamma(self.alpha_0, scale=1/self.beta_0)
            self.mu = np.sqrt(self.sigmas / self.nu_0) * np.random.randn(self.D) + self.mu_0
        else:
            mu_n, nu_n, alpha_n, beta_n = self._posterior_hypparams(
                *self._get_statistics(data))
            self.sigmas = 1. / np.random.gamma(alpha_n, scale=1./beta_n)
            print("sigams ", self.sigmas)
            self.mu = np.sqrt(self.sigmas / nu_n) * np.random.randn(self.D) + mu_n
            print("mu ", self.mu)
            
    def log_likelihood(self, x, masks=None):
        
        # if masks is None:
            # masks = np.ones_like(x, dype=bool)
        
        sq_err = -0.5 * ((x - self.mu) ** 2)
        ll = np.sum(sq_err / self.sigmas, axis=1)

        ll -= (0.5 * self.D * np.log(2 * np.pi * self.sigmas)).sum()
        return ll
        
    
    

class ScalarGaussianBase(object):
    
    def rvs(self, size=None, x=None, return_xy=False):
        return np.sqrt(self.sigma_sqrt) * np.random.normal(size=size) + self.mu
    
    def log_likelihood(self, x):
        return (-0.5*(x-self.mu)**2 / self.sigma_sqrt - np.log(np.sqrt(2 * np.pi * self.sigma_sqrt)))
        
    def _get_statistics(self, data):
        if isinstance(data, np.ndarray):
            n = data.shape[0]
            y_bar = data.mean()
            centered = data - y_bar
            sum_sqc = centered.dot(centered)
        elif isinstance(data, list):
            n = sum(d.shape[0] for d in data)
            y_bar = sum(d.sum() for d in data) / n
            sum_sqc = sum((d.ravel() - y_bar).dot(d.ravel()-y_bar) for d in data)
        else:
            n = 0
            y_bar = data
            sum_sqc = 0
        return n, y_bar, sum_sqc
    
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 'params': {'mu': self.mu, 'sigma_sqrt': self.sigma_sqrt}}
        return params


class ScalarGaussianNIX(ScalarGaussianBase):
    
    _name = 'ScalarGaussianNIX'
    
    def __init__(self, mu=None, sigma_sqrt=None, mu_0=None, kappa_0=None,
                 sigma_sqrt_0=None, nu_0=None):
        
        self.mu, self.sigma_sqrt = mu, sigma_sqrt
        self.mu_0, self.kappa_0 = mu_0, kappa_0
        self.sigma_sqrt_0, self.nu_0 = sigma_sqrt_0, nu_0
        
        if mu is sigma_sqrt is None:
            self.resample()
    
    def resample(self, data=None):
        mu_n, kappa_n, sigma_sqrt_n, nu_n = self._posterior_hypparams(*self._get_statistics(data))
        # print("mu_n ", mu_n)
        # print("kappa_n ", kappa_n)
        # print('sigma_sqrt_n ', sigma_sqrt_n)
        # print("nu_n ", nu_n)
        self.sigma_sqrt = nu_n * sigma_sqrt_n / np.random.chisquare(nu_n)
        print("sigma_sqrt ", self.sigma_sqrt)
        # self.mu = np.sqrt(self.sigma_sqrt / kappa_n) * np.random.randn() + mu_n
        self.mu = np.sqrt(self.sigma_sqrt / kappa_n) * np.random.standard_t(nu_n) + mu_n
        print("mu ", self.mu)
        
    def _posterior_hypparams(self, n, y_bar, sum_sqc):
        # print("n ", n)
        # print("y_bar ", y_bar)
        # print("sum_sqc ", sum_sqc)
        if n > 0:
            kappa_n = self.kappa_0 + n
            mu_n = (self.kappa_0 * self.mu_0 + n * y_bar) / kappa_n
            nu_n = self.nu_0 + n
            sigma_sqrt_n = (1. / nu_n) * (self.nu_0 * self.sigma_sqrt_0 + sum_sqc + ((self.kappa_0 * n / (n + self.kappa_0)) * (y_bar - self.mu_0) ** 2))
            return mu_n, kappa_n, sigma_sqrt_n, nu_n
        else:
            return self.mu_0, self.kappa_0, self.sigma_sqrt_0, self.nu_0
    
    
def ScalarGaussianNonConjNIX(ScalarGaussianBase):
    
    _name = 'ScalarGaussianNonConjNIX'
    
    def __init__(self, mu=None, sigma_sqrt=None, mu_0=None, tau_sqrt_0=None,
             sigma_sqrt_0=None, nu_0=None, n_iter=1):
    
        self.mu, self.sigma_sqrt = mu, sigma_sqrt
        self.mu_0, self.tau_sqrt_0 = mu_0, tau_sqrt_0
        self.sigma_sqrt_0, self.nu_0 = sigma_sqrt_0, nu_0
        
        self.n_iter = n_iter
        
        if mu is sigma_sqrt is None:
            self.resample()
            
    def resample(self, data=None):
        
        if data is None:
            self.mu = np.sqrt(self.tau_sqrt_0) + np.randon.normal() + self.mu_0
            self.sigma_sqrt = self.sigma_sqrt_0 * self.nu_0 / np.random.chisquare(self.nu_0)
        else:
            ...
            for itr in self.n_iter:
                tau_sqrt_n = 1. / (1. / self.tau_sqrt_0 + n / self.sigma_sqrt)
                mu_n = tau_sqrt_n * ((self.mu_0 / self.tau_sqrt_0) + ... / self.sigma_sqrt)
                self.mu = np.sqrt(tau_sqrt_n) + np.randon.normal() + mu_n
                
                nu_n = self.nu_0 + n
                sigma_sqrt_n = (self.nu_0 * self.sigma_sqrt_0 + ( ... + n * self.mu ** 2 - 2 * ... * self.mu) ) / nu_n
                self.sigma_sqrt = self.sigma_sqrt_n * self.nu_n / np.random.chisquare(self.nu_n)
                
                
class ScalarGaussianFixedMean(ScalarGaussianBase):
    
    _name = 'ScalarGaussianFixedMean'
    
    def __init__(self, mu=None, sigma_sqrt=None, alpha_0=None, beta_0=None):
        print("mu ", mu)
        
        self.mu, self.sigma_sqrt = mu, sigma_sqrt
        self.alpha_0, self.beta_0 = alpha_0, beta_0
        
        if sigma_sqrt is None:
            self.resample()

    def resample(self, data=None):
        
        if data is None:
            self.sigma_sqrt = 1. / np.random.gamma(self.alpha_0, scale=1./self.beta_0)
        else:
            n, sum_sqc = self._get_statistics(data)
            alpha_n = self.alpba_0 + n / 2
            beta_n = self.beta_0 + (1 / 2) * sum_sqc
            self.sigma_sqrt = 1. / np.random.gamma(alpha_n, scale=1./beta_n)
            
    def _get_statistics(self, data):
        if isinstance(data, np.ndarray):
            n = data.shape[0]
            centered = data - self.mu
            sum_sqc = centered.dot(centered)
        elif isinstance(data, list):
            n = sum(d.shape[0] for d in data)
            sum_sqc = sum((d.ravel() - self.mu).dot(d.ravel()-self.mu) for d in data)
        else:
            n = 0
            sum_sqc = 0
        return n, sum_sqc
        
        
