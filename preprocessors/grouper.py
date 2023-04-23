# -*- coding: utf-8 -*-
from preprocessors.config import PreprocessorConfig
from preprocessors.reader import read_input_data, read_cbs_data, read_raw_cbs_data
import numpy as np
import itertools


def select_cbs_dataset(config, N=5000000):
    data, masks = read_raw_cbs_data(config, N=N, age_to_categories=False)
    T, N, d = data.shape
    
    mass = config.MASS_IN_PERCENTILES
    perc = np.array(config.PERCENTILES)
    T = config.N_YEARS
    
    selected_data = np.zeros((T, 0, d), dtype=int)
    selected_indexes = np.array([], dtype=int)
    for t in range(T):
        inflation = np.prod(config.INFLATION[t+1:T])
        corr_perc = perc[t] * inflation
        y = data[t, :, 0] * inflation
        income_groups = np.digitize(y, corr_perc, right=True)
        
        if t > 0:
            counts = np.bincount(income_groups[selected_indexes], minlength=len(mass))
        else:
            counts = np.zeros(len(mass), dtype=int)
        print("counts ", counts)
        value_to_add = 50000 if t == 0 else 2000
        n_selected_indexes = len(selected_indexes) + value_to_add
        for idx, m in enumerate(mass):
            n_newly_selected = max(int(n_selected_indexes * m - counts[idx]), 0)
            print("n_newly_selected ", n_newly_selected)
            if n_newly_selected > 0:
                in_dataset = np.all(masks[t:], axis=0)
                indexes = np.arange(N)[(income_groups == idx) & in_dataset]
                if t > 0:
                    indexes = np.setdiff1d(indexes, selected_indexes)
                chosen_indexes = np.random.choice(indexes, size=n_newly_selected)
                selected_indexes = np.concatenate((selected_indexes, chosen_indexes))
                selected_data = np.concatenate((selected_data, data[:, chosen_indexes]), axis=1)
        print(selected_data.shape)
        
    test_indexes = np.setdiff1d(np.arange(N), selected_indexes)
    chosen_test_indexes =  np.random.choice(test_indexes, size=config.N_HOUSEHOLDS_TEST)
    selected_data = np.concatenate((selected_data, data[:, chosen_test_indexes]), axis=1)
    
    T, N, d = selected_data.shape
    selected_data = selected_data.reshape((N * T, d))
    file_name = config.raw_selected_input_file
    np.savetxt(file_name, selected_data, fmt='%i', delimiter=',')

def to_one_hot_encoding(config, CBS=False):
    
    if CBS:
        data, masks = read_cbs_data(config)
        file_name = config.processed_input_file
    else:
        data, masks = read_input_data(config)
        file_name = config.processed_exogenous_file
    print("data ", data[:, :10])
        
    stack = data[:, :, 0]
    for _, category in config.CATEGORIES.items():
        idx = category[1]
        print('idx ', idx)
        n = category[0]
        one_hot_encoding = np.array(data[:, :, idx, None] == np.arange(n), dtype=np.int32)
        # one_hot_encoding = np.delete(one_hot_encoding, category[2], 2)
        stack = np.dstack((stack, one_hot_encoding))
        
    T, N, d = stack.shape
    stack = stack.reshape((N * T, d))
    print("stack ", stack)
    # print(hallo)
    np.savetxt(file_name, stack.astype(int), fmt='%i', delimiter=',')
    
    
def modify_primos_raw_input_file(config):
    x = np.genfromtxt(config.raw_primos_input_file, delimiter=',').astype(int)
    x = np.reshape(x, (32, 5, 3, 3, 6))

    y = np.genfromtxt(config.income_source_file, delimiter=',').astype(int)
    y = np.reshape(y, (5, 3, 3, 6, 3))

    new_x = np.empty((32, 5, 3, 3, 6, 3))
    list_of_groups = [range(c[0]) for _, c in config.CATEGORIES.items()]
    groups = list(itertools.product(*list_of_groups[:-1]))
    
    for group in groups:
        p = y[group] / sum(y[group])
        v = np.round(x[:, group[0], group[1], group[2], group[3], None] * p)
        new_x[:, group[0], group[1], group[2], group[3]] = v
        
    np.savetxt(config.primos_input_file, new_x.flatten().astype(int), delimiter=',', fmt='%i')
    
    
def to_indices_per_primos_group(config, CBS=False):
    
    if CBS:
        data, masks = read_cbs_data(config)
        file_name = config.indices_primos_groups_cbs_file
    else:
        data, masks = read_input_data(config)
        file_name = config.indices_primos_groups_forecast_file
    
    print("data ", data)
    T, N, _ = data.shape
    
    list_of_groups = [range(c[0]) for _, c in config.CATEGORIES.items()]
    idxs = [c[1] for _, c in config.CATEGORIES.items()]
    groups = list(itertools.product(*list_of_groups))
    
    with open(file_name, 'w') as file:
        for t in range(T):
            year = config.BEGIN_YEAR + t
            print("year ", year)
            for i, group in enumerate(groups):
                
                indexes = np.ones(N, dtype=bool)
                for g, idx in enumerate(idxs):
                    indexes = indexes & (data[t, :, idx] == group[g])
                
                indices = np.arange(N)[indexes]
                str_group = '-'.join([str(idx) for idx in group])
    
                file.write('{0},{1},{2}\n'.format(str(year), str_group, ','.join(indices.astype(str))))
                    

def to_household_group(config, CBS=False):
    
    # 
    if CBS:
        data, masks = read_cbs_data(config, age_to_categories=False)
        file_name = config.household_group_cbs_file
    else:
        data, masks = read_input_data(config, age_to_categories=False)
        file_name = config.household_group_forecast_file
    
    print("data ", data)
    print(data.shape)
    idx_age = config.CATEGORIES['age'][1]
    idx_com = config.CATEGORIES['composition'][1]
    
    T, N, _ = data.shape
    sequences = np.full((T, N), -1, dtype=int)
    for t in range(T):
        indices = (data[t, :, idx_age] >= 20) & (data[t, :, idx_age] <= 65) & (data[t, :, idx_com] == 0)
        sequences[t, indices] = 0
        
        indices = (data[t, :, idx_age] >= 20) & (data[t, :, idx_age] <= 65) & (data[t, :, idx_com] == 1)
        sequences[t, indices] = 1
        
        indices = (data[t, :, idx_age] >= 20) & (data[t, :, idx_age] <= 65) & (data[t, :, idx_com] == 2)
        sequences[t, indices] = 2
        
        indices = (data[t, :, idx_age] >= 20) & (data[t, :, idx_age] <= 65) & (data[t, :, idx_com] == 3)
        sequences[t, indices] = 3
            
        indices = (data[t, :, idx_age] > 65) & ((data[t, :, idx_com] == 0) | (data[t, :, idx_com] == 1))
        sequences[t, indices] = 4
        
        indices = (data[t, :, idx_age] > 65) & ((data[t, :, idx_com] == 2) | (data[t, :, idx_com] == 3))
        sequences[t, indices] = 5
        
        indices = data[t, :, idx_com] == 4
        sequences[t, indices] = 0
    
    sequences[~masks] = -1
    print("sequences ", sequences)
    np.savetxt(file_name, sequences, fmt='%i', delimiter=',')
    


# [7.07816 8.30259 1.56648 6.35208 3.85812 8.2098  2.83517 3.56858 5.8499
#  4.41325 4.99066]
# [8.92365 8.805   1.64545 6.74879 4.13024 8.64061 2.95541 3.67998 6.13734
#  4.76123 5.35503]
# [10.54714  9.34942  1.71287  7.22999  4.32536  9.17781  3.17046  4.12377
#   6.54348  5.23743  5.81471]
if __name__ == "__main__":
    config = PreprocessorConfig()
    # select_cbs_dataset(config)
    
    # to_household_group(config, CBS=False)
    to_indices_per_primos_group(config, CBS=False)
    # to_one_hot_encoding(config, CBS=False)
    
    # modify_primos_raw_input_file(config)
    
    