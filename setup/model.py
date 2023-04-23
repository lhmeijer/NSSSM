# -*- coding: utf-8 -*-
import json
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from models.utils.stats import count_transitions
from configuration import Config
import itertools


class Model(object):
    
    def __init__(self, config):
        self.config = config
        
    def fit(self, y, x, masks, time_idxs=(0, 10), unit_idxs=(0, 100000)):
        
        if os.path.isfile(self.config.get_params_input_file(iteration=self.config.ITERATION)):
            input_file = self.config.get_params_input_file(iteration=self.config.ITERATION)
            with open(input_file, 'r') as file:
                params = json.load(file)
            model = self.config.extract_model_from_params(params)
        else:
            model = self.config.model(**self.config.init_params)
            
        T, N = y.shape
        y = np.log(y)   # take the logarithm of the data

        seqs, fixed_state_seqs = self.config.fixed_state_seqs(
            CBS=True, time_idxs=time_idxs, unit_idxs=unit_idxs, masks=masks)
        input_data = self.config.get_input_data(T, N)
        
        model.add_data(data=y, masks=masks, state_seqs=seqs, fixed_state_seqs=fixed_state_seqs, covariates=x, input_data=input_data)
        
        idx = 0 if self.config.ITERATION is None else self.config.ITERATION
        for itr in range(idx, self.config.MCMC_ITER):
            if itr % 100 == 0:
                print("Iteration ", itr)
            
            model.resample_model()
            if itr >= self.config.BURN_IN_ITER and itr % self.config.SAVE_ITER == 0:
                input_file = self.config.get_params_input_file(idx)
                with open(input_file, 'w') as file:
                    json.dump(model.params, file)
                idx += 1
                
    def forecast(self, y, x, masks, time_steps, time_idxs=(0, 10), unit_idxs=None, CBS=True):
        
        models = self.config.fitted_params
        
        y = np.log(y)
        T, N = y.shape
        
        ts = T 
        te = ts + time_steps
        
        if unit_idxs is None:
            unit_idxs = (0, N)
        
        seqs, fixed_state_seqs = self.config.fixed_state_seqs(
            CBS=CBS, time_idxs=time_idxs, unit_idxs=unit_idxs, masks=masks)
        
        seqs_train, seqs_test = None, None
        if fixed_state_seqs:
            seqs_train = seqs[:ts]
            seqs_test = seqs[ts:te]
        # print("seqs ", seqs.shape)
        input_data = self.config.get_input_data(T + time_steps, N)
        
        n_models = len(models)
        y_preds = np.empty((n_models, time_steps, N), dtype='double')
        state_seqs_preds = np.empty((n_models, time_steps, N), dtype=np.int32)
        
        for idx, model in enumerate(models):
            
            if idx % 10 == 0:
                print("We prediced {0} models already.".format(idx))
            
            y_pred, state_seqs_pred = model.sample_predictions(
                data=y, covariates=x[:ts], covariate_pred=x[ts:te], time_steps=time_steps, 
                n_units=N, masks=masks[:ts], masks_pred = masks[ts:te],
                state_seqs=seqs_train, fixed_state_seqs=fixed_state_seqs, input_data=input_data[:ts],
                state_seqs_pred=seqs_test, input_pred=input_data[ts:te])
    
            # y_preds[idx] = np.exp(y_pred)
            y_preds[idx] = y_pred
            state_seqs_preds[idx] = state_seqs_pred

        return y_preds, state_seqs_preds
        
            
    def predict(self, y, x, masks, time_idxs=(0, 10), unit_idxs=(0, 100000), CBS=True):
        
        models = self.config.fitted_params

        T, N = y.shape
        y = np.log(y)
        print("y ", y.shape)
        
        if unit_idxs is None:
            unit_idxs = (0, N)
        
        seqs, fixed_state_seqs = self.config.fixed_state_seqs(
            CBS=CBS, time_idxs=time_idxs, unit_idxs=unit_idxs, masks=masks)
        
        input_data = self.config.get_input_data(T, N)
        
        n_models = len(models)
        y_preds = np.empty((n_models, T, N), dtype='double')
        state_seqs_preds = np.empty((n_models, T, N), dtype=np.int32)
        
        for idx, model in enumerate(models):
            
            if idx % 10 == 0:
                print("We already prediced {0} models.".format(idx))
            
            y_pred, state_seqs_pred = model.predict(
                seed_data=y, covariates=x, masks=masks,
                state_seqs=seqs, fixed_state_seqs=fixed_state_seqs, input_data=input_data)
    
            y_preds[idx] = y_pred
            state_seqs_preds[idx] = state_seqs_pred
        
        return y_preds, state_seqs_preds
    
    
    @staticmethod
    def root_mean_squared_forecast_error(y_true, y_pred):
        err_sq = (y_true - y_pred) ** 2 
        return np.sqrt(np.mean(err_sq))
    
    @staticmethod
    def root_mean_absolute_forecast_error(y_true, y_pred):
        err_sq = np.abs(y_true - y_pred)
        return np.mean(err_sq)
    
    @staticmethod
    def cross_sectional_coverage_frequency(y_true, y_pred, perc=0.95):
        
        m, N = y_pred.shape
        n_samples = np.floor(perc * m).astype(int)
        
        coverage_frequency = 0.
        average_length = 0.
        
        for i in range(N):
            y_sort = np.sort(y_pred[:, i])
            int_width = y_sort[n_samples:] - y_sort[:m-n_samples]
            min_int = np.argmin(int_width)
            hpd = (y_sort[min_int], y_sort[min_int+n_samples])
            # print("hpd ",  hpd)
            average_length += hpd[1] - hpd[0]
            coverage_frequency += 1 if y_true[i] <= hpd[1] and y_true[i] >= hpd[0] else 0
        
        return coverage_frequency / N, average_length / N
    
    @staticmethod
    def continuous_ranked_probability_score(y_true, y_pred):
        
        m, N = y_pred.shape
        
        crps = 0.
        for i in range(N):
            forecasts = np.sort(y_pred[:, i])
            obs_cdf, forecast_cdf = 0, 0
            prev_forecast = 0
            integral = 0
            for forecast in forecasts:
                if obs_cdf == 0 and y_true[i] < forecast:
                    integral += (y_true[i] - prev_forecast) * forecast_cdf ** 2
                    integral += (forecast - y_true[i]) * (forecast_cdf - 1) ** 2
                    obs_cdf = 1
                else:
                    integral += ((forecast - prev_forecast) * (forecast_cdf - obs_cdf) ** 2)
                
                forecast_cdf += 1 / m
                prev_forecast = forecast
            crps += integral
        return crps / N

        
    def evaluate_parameters(self):
        models = self.config.fitted_params
        M = len(models)
        
        init_mus = np.zeros((M, self.config.num_states, 2))
        init_sigmas = np.zeros((M, self.config.num_states, 2))
        
        dynamic_sigmas = np.zeros((M, self.config.num_states, 2))
        emission_sigma = np.zeros((M, 1, 1))
        
        As = np.zeros((M, 9, 25))
        for m_idx, model in enumerate(models):
            mus = np.array([distn.mu for distn in model.init_dynamics_distns])
            idx_sorted_mus = np.argsort(mus[:, 0])
            init_mus[m_idx] = mus[idx_sorted_mus]
            init_sigmas[m_idx] = np.array([distn.sigmas for distn in model.init_dynamics_distns])[idx_sorted_mus]
            
            dynamic_sigmas[m_idx] = np.array([distn.sigma_sqrt_flat for distn in model.dynamics_distns])[idx_sorted_mus]
            emission_sigma[m_idx] = np.array([model.emission_distns[0].sigma_sqrt_flat])
            As[m_idx] = np.array([model.trans_distn.regression_distn.A])
            
        perctiles_init_mus = np.percentile(init_mus, axis=0, q=[5, 50, 95])
        perctiles_init_sigmas = np.percentile(init_sigmas, axis=0, q=[5, 50, 95])
        perctiles_dynamic_sigmas = np.percentile(dynamic_sigmas, axis=0, q=[5, 50, 95])
        perctiles_emission_sigma = np.percentile(emission_sigma, axis=0, q=[5, 50, 95])
        perctiles_As = np.percentile(As, axis=0, q=[5, 50, 95])
        
        result = ''
        
        for idx, perc in enumerate([5, 50, 95]):
            result += '{1}%_init_mu_mean,{0}\n'.format(','.join(str(np.round(mu[0], 4)) for mu in perctiles_init_mus[idx]), perc)
            result += '{1}%_init_mu_change,{0}\n'.format(','.join(str(np.round(mu[1], 4)) for mu in perctiles_init_mus[idx]), perc)
            result += '{1}%_init_sigma_mean,{0}\n'.format(','.join(str(np.round(sigma[0], 4)) for sigma in perctiles_init_sigmas[idx]), perc) 
            result += '{1}%_init_sigma_change,{0}\n'.format(','.join(str(np.round(sigma[1], 4)) for sigma in perctiles_init_sigmas[idx]), perc) 
            result += '{1}%_sigma_mean,{0}\n'.format(','.join(str(np.round(sigma[0], 4)) for sigma in perctiles_dynamic_sigmas[idx]), perc) 
            result += '{1}%_sigma_change,{0}\n'.format(','.join(str(np.round(sigma[1], 4)) for sigma in perctiles_dynamic_sigmas[idx]), perc) 
            result += '{1}%_sigma_emission,{0}\n'.format(str(np.round(perctiles_emission_sigma[idx, 0, 0], 4)), perc) 
            result += '\n'.join('{2}%_A_K{1},{0}'.format(','.join(str(np.round(row, 4)) for row in A), k, perc) for k, A in enumerate(perctiles_As[idx]))   
            result += '\n'
        
        file_name = 'parameters.csv'
        with open(self.config.get_results_file(file_name), 'w') as file:
            file.write(result)

        
        
    def evaluate_state_sequences(self, s_pred, y_pred, y_data, data, masks):
        models = self.config.fitted_params
        
        y_min = 9.0 
        y_max = 12.5
        step = 0.02
        
        bins = np.arange(y_min, y_max + step, step)
        n_bins = len(bins) + 1
        
        M, T, N = s_pred.shape
        K = self.config.num_states
        
        counts = np.zeros((M, T, K, K), dtype=np.int32)
        counts_per_category = {}
        
        mean_income = np.full((M, T, K), np.nan)
        quintile_income = np.full((M, T, K, 3), np.nan)
        
        mean_pred_income = np.full((M, T, K), np.nan)
        quintile_pred_income = np.full((M, T, K, 3), np.nan)
        
        pred_ecdfs = np.zeros((M, T, 100))
        
        for category, v in Config.CATEGORIES.items():
            counts_per_category[category] = np.zeros((M, v[0], K))
        
        for m_idx, model in enumerate(models):
            mus = [distn.mu[0] for distn in model.init_dynamics_distns]
            arg_mus = np.argsort(mus)
            
            state_seqs_pred = np.full((T, N), -1, dtype=np.int32)
            for idx, mu_idx in enumerate(arg_mus):
                indexes = s_pred[m_idx] == mu_idx
                
                for category, v in Config.CATEGORIES.items():
                    column = data[:, :, v[1]].copy()
                    column[~indexes] = -2
                    a = np.bincount(column[column > -1].flatten(), minlength=v[0])
                    counts_per_category[category][m_idx, :, idx] = a
                    
                state_seqs_pred[indexes] = idx
                for t in range(T):
                    if np.sum(indexes[t]) > 0:
                        mean_income[m_idx, t, idx] = np.mean(y_data[t, indexes[t]])
                        mean_pred_income[m_idx, t, idx] = np.mean(y_pred[m_idx, t, indexes[t]])
                        quintile_income[m_idx, t, idx] = np.percentile(y_data[t, indexes[t]], q=[25, 50, 75])
                        quintile_pred_income[m_idx, t, idx] = np.percentile(y_pred[m_idx, t, indexes[t]], q=[25, 50, 75])
            
            for t in range(T):
                if t == 0:
                    counts[m_idx, t, 0] = np.bincount(state_seqs_pred[0, masks[0]], minlength=K)
                else:
                    for i, j in list(itertools.product(*[range(K), range(K)])):
                        indexes = (state_seqs_pred[t-1] == i) & (state_seqs_pred[t] == j)
                        counts[m_idx, t, i, j] = np.sum(indexes)
                
                pred_ecdfs[m_idx, t], _ = np.histogram(np.exp(y_pred[m_idx, t]), bins=np.arange(0, 202000, 2000), density=True)
                
        percentiles_perc_per_category = {}
        for category, v in Config.CATEGORIES.items():
            percentiles_perc_per_category[category] = np.percentile(counts_per_category[category] , axis=0, q=[5, 50, 95])
            
        results = ''
        for category, v in Config.CATEGORIES.items():            
            for idx, perc in enumerate([5, 50, 95]):
                results += '\n'.join('{0}_{1}%_{2},{3}'.format(category, perc, i, ','.join(str(np.round(perc_c, 4)) for perc_c in perc_category)) for i, perc_category in enumerate(percentiles_perc_per_category[category][idx]))
                results += '\n'
                
        perctiles_counts = np.percentile(counts, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            for t in range(T):
                results += '\n'.join('counts_{0}%_T{1}_K{2},{3}'.format(perc, t, i, ','.join(str(np.round(c, 4)) for c in counts)) for i, counts in enumerate(perctiles_counts[idx, t]))
                results += '\n'
        
        perctiles_mean_income = np.nanpercentile(mean_income, axis=0, q=[5, 50, 95])
        
        for idx, perc in enumerate([5, 50, 95]):
            results += '\n'.join('mean_income_{0}%_T{1},{2}'.format(perc, i, ','.join(str(np.round(mu, 4)) for mu in mean_income)) for i, mean_income in enumerate(perctiles_mean_income[idx]))
            results += '\n'
            
        perctiles_quintile_income = np.nanpercentile(quintile_income, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            for t in range(T):
                results += '\n'.join('quartile_income_{0}%_T{1}_K{2},{3}'.format(perc, t, i, ','.join(str(np.round(quintile, 4)) for quintile in quintile_income)) for i, quintile_income in enumerate(perctiles_quintile_income[idx, t]))
                results += '\n'
                
        perctiles_mean_pred_income = np.nanpercentile(mean_pred_income, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            results += '\n'.join('mean_pred_income_{0}%_T{1},{2}'.format(perc, i, ','.join(str(np.round(mu, 4)) for mu in mean_income)) for i, mean_income in enumerate(perctiles_mean_pred_income[idx]))
            results += '\n'

        perctiles_quintile_pred_income = np.nanpercentile(quintile_pred_income, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            for t in range(T):
                results += '\n'.join('quantile_pred_income_{0}%_T{1}_K{2},{3}'.format(perc, t, i, ','.join(str(np.round(quintile, 4)) for quintile in quintile_income)) for i, quintile_income in enumerate(perctiles_quintile_pred_income[idx, t]))
                results += '\n'
                
        pred_ecdf = np.percentile(pred_ecdfs, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            results += '\n'.join('histogram_{0}%_T{1},{2}'.format(perc, t, ','.join(str(np.round(mu, 8)) for mu in ecdf)) for t, ecdf in enumerate(pred_ecdf[idx]))
            results += '\n'
                
        file_name = 'cluster_insight.csv'
        with open(self.config.get_results_file(file_name), 'w') as file:
            file.write(results)
    
    def evaluate(self, y_true, y_pred, known_y=True):
        
        y_min = 9.0 
        y_max = 12.5
        step = 0.02
        
        bins = np.arange(y_min + step, y_max, step)
        n_bins = len(bins) + 1
        
        y_true = np.log(y_true)
        
        M, N = y_pred.shape
        
        true_groups = np.digitize(y_true, bins, right=False)
        bin_count_true = np.bincount(true_groups, minlength=n_bins)
        true_ecdf = np.cumsum(bin_count_true / N)
        
        RMSFEs = [self.root_mean_squared_forecast_error(y_true, y_pred[i]) for i in range(M)]
        RMSFE = np.percentile(RMSFEs, q=[5, 50, 95])

        crps = self.continuous_ranked_probability_score(y_true, y_pred)
        
        mean_pred = np.percentile(np.mean(y_pred, axis=1), axis=0, q=[5, 50, 95])
        perc_pred = np.percentile(np.percentile(y_pred, axis=1, q=[10, 20, 30, 40, 50, 60, 70, 80, 90]), axis=1, q=[5, 50, 95])

        pred_ecdfs = np.zeros((M, n_bins))
        for i in range(M):
            pred_groups = np.digitize(y_pred[i], bins, right=False)
            bin_count = np.bincount(pred_groups, minlength=n_bins)
            pred_ecdfs[i] = np.cumsum(bin_count / N)
        
        pred_ecdf = np.percentile(pred_ecdfs, axis=0, q=[5, 50, 95])
        

        cramervonmisess = [stats.cramervonmises_2samp(y_pred[i], y_true).statistic for i in range(M)]
        cramervonmises = np.percentile(cramervonmisess, q=[5, 50, 95])
        andersons = [stats.anderson_ksamp([y_pred[i], y_true]).statistic for i in range(M)]
        anderson = np.percentile(andersons, q=[5, 50, 95])

        plt.plot(range(n_bins), pred_ecdf[0])
        plt.plot(range(n_bins), pred_ecdf[2])
        plt.plot(range(n_bins), pred_ecdf[1])
        plt.plot(range(n_bins), true_ecdf)
        plt.show()
        
        string_of_results = ''
        for i, p in enumerate([5, 50, 95]):
        
            results = [RMSFE[i], crps, cramervonmises[i], anderson[i], mean_pred[i]]
            results.extend(perc_pred[i].flatten().tolist())
            results.extend(pred_ecdf[i].flatten().tolist())
            
            string_of_results += '{1}_{2}_M{3}_{4}%_100_150,{0}\n'.format(
                ','.join(str(np.round(result, 4)) for result in results), self.config.name,
                'knownY' if known_y else 'unknownY', M, p) 
        
        file_name = 'forecasting_output_250_500.csv'
        with open(self.config.get_results_file(file_name, general=True), 'a') as file:
            file.write(string_of_results)
        
        
        
        
                