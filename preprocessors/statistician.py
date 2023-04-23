# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import numpy as np
from preprocessors.config import PreprocessorConfig
import itertools
from collections.abc import Iterable
from preprocessors.reader import read_household_group, read_cbs_data, read_raw_cbs_data, read_input_data
import matplotlib.pyplot as plt



def write_to_csv_file(dictionary, file_name):
    with open(file_name, 'w') as file:
        
        for k1, v1 in dictionary.items():
            for k2, v2 in v1.items():
                if isinstance(v2, Iterable):
                    if isinstance(v2[0], Iterable):
                        str_object = ','.join(['-'.join([str(y) for y in x]) for x in v2])
                    else:
                        str_object = ','.join([str(round(x, 2)) for x in v2])
                else:
                    str_object = str(round(v2, 2))
    
                row = ",".join((k1, k2, str_object)) + '\n'
                file.write(row)


def age_transitions(age, income, results):
    t_s = 0
    t_e = age.shape[0] - 1
    results['counts'] = np.zeros(5, dtype=np.int32)
    for i in range(5):
        indices = np.intersect1d(np.arange(N)[age[0] == i], np.arange(N)[age[-1] == i+1])
        results['counts'][i] = len(indices)
        if len(indices) > 0:
            name = 't{2}-{3}_{0}_{1}'.format(i, i+1, t_s, t_e)
            delta_income = income[-1, indices] - income[0, indices]
            results['{}_mean'.format(name)] = np.mean(delta_income)
            results['{}_percentiles'.format(name)] = np.percentile(a=delta_income, q=np.arange(5, 100, 5)).tolist()
    results['counts'] = results['counts'].tolist()
   
         
def transitions(column, income, c, results):
    T, N = column.shape
    for t in [1, 5, T-1]:
        results['t{0}-{1}_counts'.format(t-1, t)] = np.zeros((c, c), dtype=np.int32)
        for i, j in itertools.permutations(range(c), 2):
            indices = np.intersect1d(np.arange(N)[column[t-1] == i], np.arange(N)[column[t] == j])
            results['t{0}-{1}_counts'.format(t-1, t)][i, j] = int(len(indices))
            if len(indices) > 0:
                name = 't{2}-{3}_{0}_{1}'.format(i, j, t-1, t)
                delta_income =  income[t, indices] - income[t-1, indices]
                results['{}_mean'.format(name)] = np.mean(delta_income)
                results['{}_percentiles'.format(name)]= np.percentile(a=delta_income, q=np.arange(5, 100, 5)).tolist()

        results['t{0}-{1}_counts'.format(t-1, t)] = results['t{0}-{1}_counts'.format(t-1, t)].tolist()

        
def multiple(column1, column2, c1, c2, income, results):
    
    T, N = column1.shape
    for t in [0, 5, T-1]:
        results['t{}_counts'.format(t)] = np.zeros((c1, c2), dtype=np.int32)
        for i in range(c1):
            for j in range(c2):
                indices = np.intersect1d(np.arange(N)[column1[t] == i], np.arange(N)[column2[t] == j])
                results['t{}_counts'.format(t)][i, j] = len(indices)
                if len(indices) > 0:
                    name = 't{2}_{0}_{1}'.format(str(i), str(j), str(t))
                    results['{}_mean'.format(name)] =  np.mean(income[t, indices])
                    results['{}_percentiles'.format(name)] = np.percentile(a=income[t, indices], q=np.arange(5, 100, 5)).tolist()
        results['t{}_counts'.format(t)] = results['t{}_counts'.format(t)].tolist()
                

def single(column, c, income, results, time_steps=[1, 5, 10]):
    T, N = column.shape
    for t in time_steps:
        counts = np.bincount(column[t, column[t] > -1], minlength=c).tolist()
        results['t{}_counts'.format(t)] = counts
            
        for i in range(c):
            indices = np.arange(N)[column[t] == i]
            if len(indices) > 0:
                results['t{0}_{1}_mean'.format(t, i)] = np.mean(income[t, indices])
                results['t{0}_{1}_percentiles'.format(t, i)]= np.percentile(a=income[t, indices], q=[10, 20, 30, 40, 50, 60, 70, 80, 90])
                

def income_source(data, age, generation, education_level, composition, income_source):
    perc = np.zeros((composition, generation, education_level, age, income_source), dtype=np.dtype('U15'))
    for c in range(composition):
        N = data.shape[0]
        indices_composition = np.arange(N)[data[:, 3] == c]
        data_composition = data[indices_composition]
        for g in range(generation):
            N = data_composition.shape[0]
            indices_generation = np.arange(N)[data_composition[:, 5] == g]
            data_generation = data_composition[indices_generation]
            for e in range(education_level):
                N = data_generation.shape[0]
                indices_education_level = np.arange(N)[data_generation[:, 2] == e]
                data_education = data_generation[indices_education_level]
                for a in range(age):
                    N = data_education.shape[0]
                    indices_age = np.arange(N)[data_education[:, 1] == a]
                    data_age = data_education[indices_age]
                    for i in range(income_source):
                        N = data_age.shape[0]
                        indices_income_source = np.arange(N)[data_age[:, 4] == i]
                        if len(indices_income_source) < 10:
                            perc[c, g, e, a, i] = "0"
                        else:
                            perc[c, g, e, a, i] = str(len(indices_income_source))
                         
    return perc


def compute_income_trend(config):
    T = config.N_YEARS
    N = 5000000
    # data, masks = read_cbs_data(config, N=5000000)
    data, masks = read_cbs_data(config, N=N)
    groups = read_household_group(config.household_group_file)
    
    mean_2010 = np.mean(data[0, :, 0])
    mean_2020 = np.mean(data[-1, :, 0])
    print("mean 2010 ", mean_2010)
    print("mean 2020 ", mean_2020)
    print("growth ", (mean_2020 / mean_2010))
    
    median_2010 = np.median(data[0, :, 0])
    median_2020 = np.median(data[-1, :, 0])
    print("mean median_2010 ", median_2010)
    print("mean median_2020 ", median_2020)
    print("growth ", (median_2020 / median_2010))
    
    # for t in range(0, T-1):
    #     print("t ", t)
    #     indices = (masks[t]) & (masks[t+1])
    #     x = data[t, indices, 0]
    #     y = data[t+1, indices, 0]
    #     model = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), y)
    #     print("coefficient ", model.coef_)
        
    # indices = (masks[0]) & (masks[-1])
    # x = data[0, indices, 0]
    # y = data[-1, indices, 0]
    # model = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), y)
    # print("coefficient 2010-2020", model.coef_)
        
    
    n_groups = int(np.max(groups)) + 1
    for g in range(n_groups):
        print("group ", g)
        # indices = ((g == groups[0]) | (g == groups[-1])) & (masks[0]) & (masks[-1])
        x = data[0, g == groups[0], 0]
        y = data[-1, g == groups[-1], 0]
        # model = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), y)
        # print("coefficient ", model.coef_)
        mean_2010 = np.mean(x)
        mean_2020 = np.mean(y)
        print("mean 2010 ", mean_2010)
        print("mean 2020 ", mean_2020)
        print("growth ", (mean_2020 / mean_2010))
        
        median_2010 = np.median(x)
        median_2020 = np.median(y)
        print("mean median_2010 ", median_2010)
        print("mean median_2020 ", median_2020)
        print("growth ", (median_2020 / median_2010))
    

if __name__ == "__main__":
    config = PreprocessorConfig()
    # data, masks = read_raw_cbs_data(config, N=5000000)
    data, masks = read_input_data(config)
    # data, masks = read_cbs_data(config)
    # data, masks = data[:, :config.N_HOUSEHOLDS_TRAIN], masks[:, :config.N_HOUSEHOLDS_TRAIN]
    # compute_income_trend(config)
    T = data.shape[0]
    for t in range(T):
        data[t, :, 0] = data[t, :, 0] * np.prod(config.INFLATION[t+1:T])
        

    
    # income_source_data = income_source(data[-1], 6, 3, 3, 5, 3)
    # print(income_source_data)
    # source_file_name = config.get_household_income_source_output_file()
    # str_income_source = ','.join(income_source_data.flatten())
    # with open(source_file_name, 'w') as file:
    #     file.write(str_income_source)
    
    masks = masks & data[:, :, 0] > 0
    single_results = {}
    for category, v in config.CATEGORIES.items():
        single_results[category] = {}
        single(data[:, :, v[1]], v[0], data[:, :, 0], single_results[category])

        
    single_results['general'] = {}
    for t in range(config.N_YEARS):
        single_results['general']['t{0}_mean'.format(t)] = np.mean(data[t, masks[t], 0])
        single_results['general']['t{0}_percentiles'.format(t)] = np.percentile(a=data[t, masks[t], 0], q=[10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    histograms = np.zeros((config.N_YEARS, 100))
    for t in range(0, config.N_YEARS):
        histograms[t], _ = np.histogram(data[t, :, 0], bins=np.arange(0, 202000, 2000), density=True)

        plt.hist(data[t, :, 0], np.arange(0, 202000, 2000), density=True)
        plt.show()

    
    file_name = config.single_statistics_file
    write_to_csv_file(single_results, file_name)
    
    with open(file_name, 'a') as file:
        text = '\n'.join(','.join(str(np.round(v, decimals=10)) for v in histogram) for histogram in histograms)
        file.write(text)

    
                

        
    

    