# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import linalg


def sample_crp_table_counts(concentration, counts, col_weights):
    
    m = np.zeros_like(counts)
    total = counts.sum()
    rand_seq = np.random.random(total)
    
    starts = np.empty_like(counts)
    starts[0, 0] = 0
    starts.flat[1:] = np.cumsum(np.ravel(counts)[:counts.size - 1])
    
    for (i, j), n in np.ndenumerate(counts):
        w = col_weights[i,j]
        conc = w * concentration
        for k in range(n):
            m[i, j] += rand_seq[starts[i, j] + k] < (conc / (k + conc))
    return m


def count_transitions(stateseq, masks, num_states):
    out = np.zeros((num_states, num_states), dtype=np.int32)
    for idx, (i,j) in enumerate(zip(stateseq[:-1], stateseq[1:])):
        if masks[idx]:
            out[i,j] += 1
    return out



def sample_gaussian(mu=None, Sigma=None, J=None, h=None):
    mean_params = mu is not None and Sigma is not None
    info_params = J is not None and h is not None
    assert mean_params or info_params

    if mu is not None and Sigma is not None:
        return np.random.multivariate_normal(mu, Sigma)
    else:
        from scipy.linalg.lapack import dpotrs
        L = np.linalg.cholesky(J)
        x = np.random.randn(h.shape[0])
        return linalg.solve_triangular(L,x,lower=True,trans='T') \
            + dpotrs(L,h,lower=True)[0]


def sample_inv_wishart(S, nu):
    n = S.shape[0]
    chol = np.linalg.cholesky(S)

    if (nu <= 81+n) and (nu == np.round(nu)):
        x = np.random.randn(int(nu),n)
    else:
        x = np.diag(np.sqrt(np.atleast_1d(stats.chi2.rvs(nu-np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)//2)
    R = np.linalg.qr(x,'r')
    T = linalg.solve_triangular(R.T, chol.T, lower=True).T
    return np.dot(T, T.T)


def sample_inv_gamma(alpha, beta):
    return 1./np.random.gamma(alpha, 1.0 / beta)

def sample_generalized_inv_gamma(alpha, beta, c):
    return 1./stats.gengamma.rvs(alpha, c, scale=1.0 / beta)


def logistic(x):
    return 1./(1+np.exp(-x))


def sample_discrete_number(probs, K=None):
    # print("K ", K)
    N, d = probs.shape
    idx = -1 if K is None else min(K-1, d-1)
    K = d if K is None else min(K, d)
    x = np.random.rand(N)
    # print("x ", x)
    # print("N ", N)
    # print('probs ', probs)
    random_cum = np.cumsum(np.full(d, 1./K))
    # print("random_cum ", random_cum)
    
    cumulative = np.cumsum(probs, axis=1)
    # print('cumulative ', cumulative)
    random_indexes = np.arange(N)[cumulative[:, idx] == 0.0]
    # print('random_indexes ', random_indexes.shape)
    # a = np.arange(N)[cumulative[:, -1] >= 1.0]
    # print('random_indexes ', a.shape)
    cumulative[random_indexes] = random_cum
    # print("cumulative ", cumulative[indices])
    prev_indexes = np.zeros(N, dtype=bool)
    seqs = np.empty(N, dtype=np.int32)
    for idx in range(d):
        indexes = (x < cumulative[:, idx]) & ~prev_indexes
        prev_indexes = indexes | prev_indexes
        # print("indexes ", indexes)
        # if idx > 2:
        #     print(cumulative[indexes])
        #     print(x[indexes])
        seqs[indexes] = idx
    # print("seqs ", np.max(seqs))
    # print("seqs ", np.min(seqs))
    # print('seqs ', seqs)
    return seqs
    
