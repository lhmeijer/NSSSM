# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:37:09 2023

@author: lisa
"""

import numpy as np
import scipy.linalg as linalg


class KalmanFilter(object):

    def __init__(self, mu_init, sigma_init, A, B, Q, C, D, R, rts_smoother=False):
        self.mu_init = mu_init
        self.sigma_init = sigma_init

        self.A = A
        self.B = B
        self.Q = Q

        self.C = C
        self.D = D
        self.R = R
        
        self.rts_smoother = rts_smoother
        
    def kalman_filter(self, data, masks=None, inputs_emission=None, inputs_latent=None):
        
        D_latent = self.A.shape[0]
        T, N = data.shape

        if masks is None:
            masks = np.ones_like(data, dtype=bool)
            
        if inputs_latent is None:
            dim = self.B.shape[1]
            inputs_latent = np.zeros((T, N, dim))

        Bu = np.einsum('ij,TNj->TNi', self.B, inputs_latent)

        if inputs_emission is None:
            dim = self.D.shape[1]
            inputs_emission = np.zeros((T, N, dim))

        Du = np.einsum('ij,TNj->TNi', self.D, inputs_emission)
        
        updated_sigma = np.empty((T, D_latent, D_latent), dtype='double')
        updated_mu = np.empty((T, N, D_latent), dtype='double')
        
        pred_sigma = np.empty((T+1, D_latent, D_latent), dtype='double')
        pred_mu = np.empty((T+1, N, D_latent), dtype='double')

        pred_mu[0] = np.repeat([self.mu_init], repeats=N, axis=0)
        pred_sigma[0] = self.sigma_init
        
        for t in range(T):
            updated_mu[t], updated_sigma[t] = self.update_step(
                pred_mu[t], pred_sigma[t], data[t, :, None], masks[t, :, None], 
                self.C, Du[t], self.R)
            pred_mu[t+1], pred_sigma[t+1] = self.prediction_step(
                updated_mu[t], updated_sigma[t], self.A, Bu[t], self.Q)
        
        return updated_mu, updated_sigma, pred_mu, pred_sigma

    def resample(self, data, masks=None, inputs_emission=None, inputs_latent=None):
        
        D_latent = self.A.shape[0]
        T, N = data.shape

        if masks is None:
            masks = np.ones_like(data, dtype=bool)
            
        if inputs_latent is None:
            dim = self.B.shape[1]
            inputs_latent = np.zeros((T, N, dim))

        Bu = np.einsum('ij,TNj->TNi', self.B, inputs_latent)

        updated_mu, updated_sigma, _, _ = self.kalman_filter(
            data, masks, inputs_emission, inputs_latent)

        gaussian_states = np.empty((T, N, D_latent))
        sigma_chol = linalg.cholesky(updated_sigma[T-1])
        gaussian_states[T-1] = self.multivariate_normal(updated_mu[T-1], sigma_chol)
        
        masks = np.ones_like(data, dtype=bool)
        for t in range(T-2, -1, -1):
            if self.rts_smoother:
                updated_mu[t], updated_sigma[t] = self.apply_rts_smoother(
                    updated_mu[t], updated_sigma[t], updated_mu[t+1], 
                    updated_sigma[t+1], masks[t])
                # updated_mu[t], updated_sigma[t] = self.apply_rts_smoother(
                #     updated_mu[t], updated_sigma[t], gaussian_states[t+1], 
                #     updated_sigma[t+1], masks[t])
            else:
                # updated_mu[t], updated_sigma[t] = self.update_step(
                #     updated_mu[t], updated_sigma[t], updated_mu[t+1], 
                #     masks[t], self.A, Bu[t], self.Q)
                updated_mu[t], updated_sigma[t] = self.update_step(
                    updated_mu[t], updated_sigma[t], gaussian_states[t+1], 
                    masks[t], self.A, Bu[t], self.Q)
            
            # print("updated_sigma[t] ", updated_sigma[t])
            sigma_chol = linalg.cholesky(updated_sigma[t])
            gaussian_states[t] = self.multivariate_normal(updated_mu[t], sigma_chol)
        # print("gaussian_states ", gaussian_states)
        return gaussian_states
    
    def apply_rts_smoother(self, mu_x, sigma_x, mu_y, sigma_y, masks):
        pred_mu, pred_sigma = self.prediction_step(mu_x, sigma_x, self.A, self.Q)
        P_inv = linalg.inv(pred_sigma)
        PA = np.einsum('ij,kj->ik', sigma_x, self.A)
        K = np.einsum('ik,ij->kj', PA, P_inv)
        
        mu_minus_mu = mu_y - pred_mu
        # mu_minus_mu[~masks] = 0.0
        
        mu = mu_x + np.einsum('ik,Nk->Ni', K, mu_minus_mu)
        K_P = np.einsum('ij,jk->ik', K, sigma_y - pred_sigma)
        sigma = sigma_x + np.einsum('ij,kj->ik', K_P, K)
        return mu, sigma
    
    @staticmethod
    def multivariate_normal(mu, sigma_chol):
        size = (mu.shape[0], mu.shape[1])
        return mu + np.random.normal(size=size).dot(sigma_chol.T)
        
    @staticmethod
    def prediction_step(x, p, A, Bu, Q):
        predicted_x = np.einsum('ij,Nj->Ni', A, x) + Bu
        predicted_p = np.einsum('ij,jk->ik', A, p)
        predicted_p = np.einsum('ij,kj->ik', predicted_p, A)
        predicted_p += Q
        return predicted_x, predicted_p    

    @staticmethod
    def update_step(x, p, data, masks, C, Du, R):
        Cx = np.einsum('ij,Nj->Ni', C, x)

        err = data - Du - Cx
        err[~masks] = 0.0
        # print("err ", err[masks])

        pC = np.einsum('ij,kj->ik', p, C)
        S = np.einsum('ij,jk->ik', C, pC) + R

        inv_S = linalg.inv(S)
        K = np.einsum('ik,kj->ik', pC, inv_S)
        # print("K ", K)
        updated_x = x + np.einsum('ik,Nk->Ni', K, err)
        # print("updated_x ", updated_x[masks[:, 0]])

        dim = x.shape[1]
        I = np.eye(dim)
        KC = np.einsum('ij,jk->ik', K, C)
        I_KC = I - KC
        I_KC_P = np.einsum('ij,jk->ik', I_KC, p)
        I_KC_P_I_KC = np.einsum('ij,kj->ik', I_KC_P, I_KC)
        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        KR = np.einsum('ij,jk->ik', K, R)
        KRK = np.einsum('ij,kj->ik', KR, K)
        updated_p = I_KC_P_I_KC + KRK
        # print("updated_p ", updated_p)

        return updated_x, updated_p


