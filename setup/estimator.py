# -*- coding: utf-8 -*-
from preprocessors.reader import read_preprocessed_cbs_data, read_primos_data, \
    read_indices_per_primos_group, read_preprocessed_input_data, read_cbs_data
import numpy as np
import hera.utils as help_functions
import matplotlib.pyplot as plt
import scipy.stats as sts
from scipy.interpolate import make_interp_spline, splrep, BSpline, splev

class Estimator(object):
    
    
    def __init__(self, config, model):
        self.model = model
        self.config = config
        
    @property
    def num_states(self):
        return self.model.config.num_states

    def fit(self, start_year=None, end_year=None):
        
        start_year = self.config.BEGIN_YEAR if start_year is None else start_year
        ts = start_year - self.config.BEGIN_YEAR
        end_year = self.config.BASE_YEAR if end_year is None else end_year
        te = end_year - self.config.BEGIN_YEAR + 1

        T = self.config.BASE_YEAR - self.config.BEGIN_YEAR + 1
        N = self.config.N_HOUSEHOLDS_TRAIN
        print(ts, te, T)
        
        data, masks = read_preprocessed_cbs_data(self.config)
        y = data[:, :N, 0]
        x = data[ts:te, :N, 1:]
        
        for t in range(ts, te):
            y[t] = y[t] * np.prod(self.config.INFLATION[t+1:T])
        
        self.model.fit(y=y[ts:te], x=x, masks=masks[ts:te, :N],
                       time_idxs=(ts, te), unit_idxs=(0, N))
        
    def evaluate_predictions(self, start_year=None, end_year=None):
        
        start_year = self.config.BEGIN_YEAR if start_year is None else start_year
        ts = start_year - self.config.BEGIN_YEAR
        end_year = self.config.BASE_YEAR if end_year is None else end_year
        te = end_year - self.config.BEGIN_YEAR + 1
        
        T = self.config.BASE_YEAR - self.config.BEGIN_YEAR + 1
        ne = self.config.N_HOUSEHOLDS_TRAIN
        ns = 0
        
        data, masks = read_preprocessed_cbs_data(self.config)
        cbs_data, cbs_masks = read_cbs_data(self.config)
        y_cbs = cbs_data[:, ns:ne, 0]
        
        y = data[:, ns:ne, 0]
        x = data[ts:te, ns:ne, 1:]
        
        for t in range(ts, te):
            y_cbs[t] = y_cbs[t] * np.prod(self.config.INFLATION[t+1:T])
        y_cbs = np.log(y_cbs)
    
        y[ts:te] = -2.
        y_preds, seqs_preds = self.model.predict(y=y[ts:te], x=x, masks=masks[ts:te, ns:ne], 
                                                  time_idxs=(ts, te), unit_idxs=(ns, ne))
        
        self.model.evaluate_parameters()
        self.model.evaluate_state_sequences(seqs_preds, y_preds, y_cbs[ts:te], cbs_data[ts:te, ns:ne], masks=masks[ts:te, ns:ne])
        
        
    def evaluate_one_step_ahead_forecast(self, start_year=None, base_year=None,
                                         end_year=None, prev_y_known=True):
        
        start_year = self.config.BEGIN_YEAR if start_year is None else start_year
        ts = start_year - self.config.BEGIN_YEAR
        end_year = self.config.BASE_YEAR if end_year is None else end_year
        te = end_year - self.config.BEGIN_YEAR + 1
        print(ts, te)

        T = self.config.BASE_YEAR - self.config.BEGIN_YEAR + 1
        ns = self.config.N_HOUSEHOLDS_TRAIN
        ne = self.config.N_HOUSEHOLDS
        
        data, masks = read_preprocessed_cbs_data(self.config)
        cbs_data, cbs_masks = read_cbs_data(self.config)
        y = data[:T, :, 0]
        x = data[ts:T, ns:ne, 1:]
        
        for t in range(ts, te):
            y[t] = y[t] * np.prod(self.config.INFLATION[t+1:T])

        if not prev_y_known:
            y[ts:te, ns:ne] = -2.
            
        y_preds, seqs_preds = self.model.forecast(y=y[ts:te, ns:ne], x=x, masks=masks[ts:T, ns:ne], 
                                                 time_idxs=(ts, T), unit_idxs=(ns, ne), time_steps=1)
        
        self.model.evaluate(
            y[-1, ns:ne][masks[-1, ns:ne]], y_preds[:, -1][:, masks[-1, ns:ne]], prev_y_known)
        
        
    def forecast(self, start_year=None, base_year=None, end_year=None):
        
        start_year = self.config.BEGIN_YEAR if start_year is None else start_year
        base_year = self.config.BASE_YEAR if base_year is None else base_year
        end_year = self.config.END_YEAR if end_year is None else end_year
        
        data, masks = read_preprocessed_input_data(self.config)
        primos_data = read_primos_data(self.config)
        groups = read_indices_per_primos_group(self.config)
        
        ts = start_year - self.config.BEGIN_YEAR
        print("ts ", ts)
        te = end_year - self.config.BEGIN_YEAR + 1
        print("te ", te)
        tb = base_year - self.config.BEGIN_YEAR + 1
        print("tb ", tb)
        
        y, x = data[:, :, 0], data[ts:te, :, 1:]
        
        T_Base = self.config.BASE_YEAR - self.config.BEGIN_YEAR + 1
        for t in range(ts, te):
            y[t] = y[t] * np.prod(self.config.INFLATION[t+1:T_Base])
        
        # y[ts:tb] = -2.
        
        time_steps = te - tb
        y_preds, seqs_preds = self.model.forecast(
            y=y[ts:tb], x=x, masks=masks[ts:te], time_idxs=(ts, te), time_steps=time_steps, CBS=False)
        

        # y_preds = np.exp(y_preds)
        
        M, T, N = y_preds.shape  # Number of models, time, number of households

        # make it from 0 to euro 350.000
        bins = np.arange(self.config.INCOME_STEPS + self.config.MINIMUM_INCOME, self.config.MAXIMUM_INCOME, self.config.INCOME_STEPS)
        G = len(bins) + 1

        final_estimates = np.empty((M, te-tb, self.num_states + 1, G))
        mean_estimates = np.empty((M, te-tb, self.num_states + 1))
        mean_data = np.empty((M, te-tb, self.num_states + 1))

        deciles = np.empty((M, te-tb, self.num_states + 1, 10))
        deciles_data = np.empty((M, te-tb, self.num_states + 1, 9))

        for m in range(M):
            for t in range(te-tb):
                year = t + base_year
                income = y_preds[m, t]
                income[income >= 13.5] = np.nan
                income[income <= 7.0] = np.nan
                
                masks = ~np.isnan(income)
                
                for k in range(self.num_states):
                    indexes = seqs_preds[m, t] == k
                    mean_data[m, t, k] = np.nanmean(income[indexes])
                    deciles_data[m, t, k] = np.nanpercentile(income[indexes], q=[10, 20, 30, 40, 50, 60, 70, 80, 90])
                mean_data[m, t, -1] = np.nanmean(income)
                deciles_data[m, t, -1] = np.nanpercentile(income, q=[10, 20, 30, 40, 50, 60, 70, 80, 90])
                
                income_groups = np.full(N, -1, dtype=np.int32)
                income_groups[masks] = np.digitize(income[masks], bins, right=False)

                # income_groups[group_to_large] = -1

                # Compute the estimates
                t_primos = year - self.config.PRIMOS_BEGIN_YEAR

                estimates = self.get_estimates(income_groups, seqs_preds[m, t], groups[year], primos_data[t_primos], G)
                mean_estimates[m, t] = [help_functions.mean_estimates(estimates[k], self.config.INCOME_STEPS) for k in range(self.num_states+1)]
                
                final_estimates[m, t] = estimates
                results = [help_functions.deciles(estimates[k]) for k in range(self.num_states+1)]
                deciles[m, t] = [result[0] for result in results]
        
        
        perc_estimates = np.percentile(final_estimates, axis=0, q=[5, 50, 95])
        results = ''
        for idx, perc in enumerate([5, 50, 95]):
            for t in range(T):
                results += '\n'.join('{0}%_T{1}_K{2},{3}'.format(perc, t, i, ','.join(str(np.round(mu, 4)) for mu in estimates)) for i, estimates in enumerate(perc_estimates[idx, t]))
                results += '\n'
        
        file_name = self.model.config.get_results_file('forecasts.csv')
        with open(file_name, 'w') as file:
            file.write(results)
        
        perc_mean_estimates = np.percentile(mean_estimates, axis=0, q=[5, 50, 95])
        results = ''
        for idx, perc in enumerate([5, 50, 95]):
            results += '\n'.join('estimates_mean_income_{0}%_T{1},{2}'.format(perc, i, ','.join(str(np.round(mu, 4)) for mu in mu_income)) for i, mu_income in enumerate(perc_mean_estimates[idx]))
            results += '\n'
          
        perc_mean_estimates = np.percentile(mean_data, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            results += '\n'.join('data_mean_income_{0}%_T{1},{2}'.format(perc, i, ','.join(str(np.round(mu, 4)) for mu in mu_income)) for i, mu_income in enumerate(perc_mean_estimates[idx]))
            results += '\n'
            
        perc_deciles_estimates = np.percentile(deciles, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            for t in range(te-tb):
                results += '\n'.join('estimates_percentiles_income_{0}%_T{1}_K{2},{3}'.format(perc, t, i, ','.join(str(np.round(mu, 4)) for mu in mu_income)) for i, mu_income in enumerate(perc_deciles_estimates[idx, t]))
                results += '\n'
          
        perc_deciles_estimates = np.percentile(deciles_data, axis=0, q=[5, 50, 95])
        for idx, perc in enumerate([5, 50, 95]):
            for t in range(te-tb):
                results += '\n'.join('data_percentiles_income_{0}%_T{1}_K{2},{3}'.format(perc, t, i, ','.join(str(np.round(mu, 4)) for mu in mu_income)) for i, mu_income in enumerate(perc_deciles_estimates[idx, t]))
                results += '\n'
            
        file_name = self.model.config.get_results_file('forecast_statistics.csv')
        with open(file_name, 'w') as file:
            file.write(results)

    def get_estimates(self, income_groups, state_sequences, primos_groups, primos_data, n_income_groups):
        N = state_sequences.shape[0]
        x = np.arange(self.config.MINIMUM_INCOME, self.config.MAXIMUM_INCOME, self.config.INCOME_STEPS)
self.config.MAXIMUM_INCOME  + self.config.INCOME_STEPS, self.config.INCOME_STEPS)
        
        estimates = np.zeros((self.num_states + 1, n_income_groups))
        for k in range(self.num_states):
            model_indices = np.arange(N)[state_sequences == k]
            for group, indices in primos_groups.items():
                if len(indices) == 0:
                    continue
          
                i_s = tuple([int(e) for e in group.split('-')])
                all_indices = np.intersect1d(indices, model_indices)
                selected_groups = income_groups[all_indices]
                c = np.bincount(selected_groups[selected_groups > -1], minlength=n_income_groups)
                c = c / len(indices)
                c *= primos_data[i_s]

                estimates[k] += c
                estimates[-1] += c

            spline = splrep(x, estimates[k], s=0.01, t=np.linspace(8, 12, 20))
            spline_estimates = BSpline(*spline)(x)
            estimates[k] = spline_estimates / np.sum(spline_estimates)
         
        spline = splrep(x, estimates[-1], s=0.01, t=np.linspace(8, 12, 20))
        spline_estimates = BSpline(*spline)(x)
        estimates[-1] = spline_estimates / np.sum(spline_estimates)

        return estimates




