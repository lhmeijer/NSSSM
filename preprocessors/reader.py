# -*- coding: utf-8 -*-

import numpy as np


def _age_to_categories(config, data, masks):
    idx_age = config.CATEGORIES['age'][1]
    bins = config.AGE_BINS
    data[masks, idx_age] = np.digitize(data[masks, idx_age], bins, right=False)


def read_raw_cbs_data(config, N=None, age_to_categories=True):
    file_name = config.raw_input_file
    T = config.N_YEARS
    N = config.N_HOUSEHOLDS if N is None else N
        
    data = np.genfromtxt(file_name, delimiter=',', max_rows=T*N, dtype=np.int32)
    NT, d = data.shape
    
    data = data.reshape((N, T, d))
    data = np.transpose(data, (1, 0, 2))
    data = data[:, :, 2:]
    
    masks = (data[:, :, 0] != -1) & (data[:, :, 0] != 0)
    
    if age_to_categories:
        _age_to_categories(config, data, masks)
    
    return data, masks


def _read_data(file_name, T, config, age_to_categories):
    data = np.genfromtxt(file_name, delimiter=',').astype(int)
    NT, d = data.shape
    N = int(NT / T)
    
    data = data.reshape((T, N, d))
    masks = (data[:, :, 0] != -1) & (data[:, :, 0] != 0)
    
    if age_to_categories:
        _age_to_categories(config, data, masks)
        
    return data, masks

def read_cbs_data(config, age_to_categories=True):
    file_name = config.raw_selected_input_file
    T = config.N_YEARS
    return _read_data(file_name, T, config, age_to_categories)

def read_preprocessed_cbs_data(config):
    file_name = config.processed_input_file
    T = config.N_YEARS
    return _read_data(file_name, T, config, age_to_categories=False)


def read_input_data(config, age_to_categories=True):
    file_name = config.raw_exogenous_file
    T = config.N_YEARS + config.N_FUTURE_YEARS
    return _read_data(file_name, T, config, age_to_categories)


def read_preprocessed_input_data(config):
    file_name = config.processed_exogenous_file
    T = config.N_YEARS + config.N_FUTURE_YEARS
    return _read_data(file_name, T, config, age_to_categories=False)


def read_household_group(file_name):
    data = np.genfromtxt(file_name, delimiter=',').astype(int)
    return data


def read_primos_data(config):
    file_name = config.primos_input_file
    data = np.genfromtxt(file_name, delimiter=',').astype(int)
    data = np.reshape(data, (32, 5, 3, 3, 6, 3))
    return data
    

def read_indices_per_primos_group(config, CBS=False):
    if CBS:
        file_name = config.indices_primos_groups_cbs_file
    else:
        file_name = config.indices_primos_groups_forecast_file
    indices = {}
    
    with open(file_name, 'r') as file:
        for line in file:
            line = line.replace('\n', '')
            split_line = line.split(',')
            year = int(split_line[0])
            if year not in indices:
                indices[year] = {}
            if split_line[2] == '':
                indices[year][split_line[1]] = np.array([])
            else:
                indices[year][split_line[1]] = np.array(split_line[2::]).astype(np.int32)

                
    return indices