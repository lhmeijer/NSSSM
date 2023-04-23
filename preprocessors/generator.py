# -*- coding: utf-8 -*-
from preprocessors.reader import read_cbs_data, read_raw_cbs_data
import itertools
from preprocessors.config import PreprocessorConfig
import numpy as np


def generate(config):

    # cbs_data, masks = read_cbs_data(config, age_to_categories=False)
    # T_CBS, N_CBS, d = cbs_data.shape
    
    cbs_data, masks = read_raw_cbs_data(config, N=1000000, age_to_categories=False)
    T_CBS, N_CBS, d = cbs_data.shape
    
    list_of_groups = [range(20, 65), range(3), range(5), range(3), range(3)]
    groups = list(itertools.product(*list_of_groups))
    # list_of_accretion_groups = [range(3), range(5), range(3), range(3)]
    # accretion_groups = list(itertools.product(*list_of_accretion_groups))
    
    print("N_CBS ", N_CBS)
    T = config.N_YEARS + config.N_FUTURE_YEARS
    # N = len(groups) * config.N_PER_GROUP + (len(accretion_groups) * config.N_PER_GROUP * (T - T_CBS)) + N_CBS
    N = len(groups) * config.N_PER_GROUP * (T - T_CBS + 1) + N_CBS

    print("N ", N)
    
    dataset = np.full((T, N, d), -1, dtype=np.int32)
    dataset[:T_CBS, :N_CBS] = cbs_data
    
    extra_dataset = np.repeat(groups, config.N_PER_GROUP, axis=0)
    hh_alone = (extra_dataset[:, 1] == 0) | (extra_dataset[:, 1] == 1)
    extra_dataset[hh_alone, 4] = 0
    print("cbs_data ",  cbs_data.shape)
    print('prev_dataset ', extra_dataset.shape)
    
    # add a column with with unknown income
    extra_dataset = np.hstack((np.full((extra_dataset.shape[0], 1), -2), extra_dataset))
    
    print('prev_dataset ', extra_dataset.shape)
    
    prev_dataset = np.vstack((cbs_data[-1], extra_dataset))
    print("prev_dataset ", prev_dataset.shape)
    
    end_idx = prev_dataset.shape[0]
    dataset[T_CBS-1, :end_idx] = prev_dataset
    
    # accr_dataset = np.repeat(accretion_groups, config.N_PER_GROUP, axis=0)
    # hh_alone = (accr_dataset[:, 1] == 0) | (accr_dataset[:, 1] == 1)
    # accr_dataset[hh_alone, 3] = 0
    # accr_dataset = np.hstack((np.full((accr_dataset.shape[0], 1), 20), accr_dataset))
    # accr_dataset = np.hstack((np.full((accr_dataset.shape[0], 1), -2), accr_dataset))
    # print(accr_dataset.shape)
    
    idx_age = config.CATEGORIES['age'][1]
    idx_com = config.CATEGORIES['composition'][1]
    idx_ins = config.CATEGORIES['incomeSource'][1]
    
    for t in range(T_CBS, T):
        print("t ", t)
        new_dataset = prev_dataset.copy()
        masks = new_dataset[:, 0] == -1
        new_dataset[~masks, 0] = -2
        size = new_dataset.shape[0]

        new_dataset[~masks, idx_age] += 1
        age_above_85 = prev_dataset[:, idx_age] >= 85
        new_dataset[age_above_85, :] = -1
        print(new_dataset[0])
        
        # From together to alone
        hh_together = (prev_dataset[:, idx_com] == 2) | (prev_dataset[:, idx_com] == 3)
        hh_together = np.arange(size)[hh_together]
        n = int(len(hh_together) * 0.05)
        to_change = np.random.choice(hh_together, n, replace=False)
        new_dataset[to_change, idx_com] = [0 if prev_dataset[i, idx_com] == 2 else 1 for i in to_change]
        new_dataset[to_change, idx_ins] = 0
        
        # From alone to together
        hh_alone = (prev_dataset[:, idx_com] == 0) | (prev_dataset[:, idx_com] == 1)
        hh_alone = np.arange(size)[hh_alone]
        n = int(len(hh_alone) * 0.05)
        to_change = np.random.choice(hh_alone, n, replace=False)
        new_dataset[to_change, idx_com] = [2 if prev_dataset[i, idx_com] == 0 else 3 for i in to_change]
        np.random.shuffle(to_change)
        n = int(len(to_change) * 0.6)
        new_dataset[to_change[:n], idx_ins] = 1
        new_dataset[to_change[n:], idx_ins] = 0
        
        # Households without children to children
        hh_without = ((prev_dataset[:, idx_com] == 0) | (prev_dataset[:, idx_com] == 2)) & (prev_dataset[:, idx_age] < 45 )
        hh_without = np.arange(size)[hh_without]
        n = int(len(hh_without) * 0.05)
        to_change = np.random.choice(hh_without, n, replace=False)
        new_dataset[to_change, idx_com] = [1 if prev_dataset[i, idx_com] == 0 else 3 for i in to_change]
        
        # Households with chilren to without children
        hh_with = ((prev_dataset[:, idx_com] == 1) | (prev_dataset[:, idx_com] == 3)) & ((prev_dataset[:, 0] >= 40 ) | (prev_dataset[:, 0] <= 65 ))
        hh_with = np.arange(size)[hh_with]
        n = int(len(hh_with) * 0.05)
        to_change = np.random.choice(hh_with, n, replace=False)
        new_dataset[to_change, idx_com] = [0 if prev_dataset[i, idx_com] == 1 else 2 for i in to_change]
        
        # Households with benefits to wage
        hh_benefit = (prev_dataset[:, idx_ins] == 2) & (prev_dataset[:, idx_age] <= 65)
        hh_benefit = np.arange(size)[hh_benefit]
        n = int(len(hh_benefit) * 0.05)
        to_change = np.random.choice(hh_benefit, n, replace=False)
        np.random.shuffle(to_change)
        n = int(len(to_change) * 0.05)
        new_dataset[to_change[:n], idx_ins] = 0
        new_dataset[to_change[n:], idx_ins] = 1
        
        hh_alone = ((new_dataset[:, idx_com] == 0) | (new_dataset[:, idx_com] == 1)) & (new_dataset[:, idx_ins] == 1)
        new_dataset[hh_alone, idx_ins] = 0

        # Household with wage to benefits 65plus
        hh_wage_65plus = ((prev_dataset[:, idx_ins] == 0) | (prev_dataset[:, idx_ins] == 1)) & (prev_dataset[:, idx_age] > 65 ) 
        hh_wage_65plus = np.arange(size)[hh_wage_65plus]
        n = int(len(hh_wage_65plus) * 0.7)
        to_change = np.random.choice(hh_wage_65plus, n, replace=False)
        new_dataset[to_change, idx_ins] = 2 
        
        # Household with wage to benefits 65min
        hh_wage_65min = ((prev_dataset[:, idx_ins] == 0) | (prev_dataset[:, idx_ins] == 1)) & (prev_dataset[:, idx_age] <= 65 ) 
        hh_wage_65min = np.arange(size)[hh_wage_65min]
        n = int(len(hh_wage_65min) * 0.05)
        to_change = np.random.choice(hh_wage_65min, n, replace=False)
        new_dataset[to_change, idx_ins] = 2 
        
        # Household with together wage to alone wage
        hh_together_wage = (prev_dataset[:, idx_ins] == 1) & (prev_dataset[:, idx_age] <= 65 ) 
        hh_together_wage = np.arange(size)[hh_together_wage]
        n = int(len(hh_together_wage) * 0.05)
        to_change = np.random.choice(hh_together_wage, n, replace=False)
        new_dataset[to_change, idx_ins] = 0 
        
        # Household with alone wage to together wage
        hh_alone_wage = (prev_dataset[:, idx_ins] == 0) & ((prev_dataset[:, idx_com] == 2) | (prev_dataset[:, idx_com] == 3))
        hh_alone_wage = np.arange(size)[hh_alone_wage]
        n = int(len(hh_alone_wage) * 0.05)
        to_change = np.random.choice(hh_alone_wage, n, replace=False)
        new_dataset[to_change, idx_ins] = 1 
        
        prev_dataset = np.vstack((new_dataset, extra_dataset))
        end_idx = prev_dataset.shape[0]
        dataset[t, :end_idx] = prev_dataset
        
        
        # print(hallo)

    print("dataset ", dataset.shape)
    dataset = dataset.reshape(N * T, d)
    np.savetxt(config.raw_exogenous_file, dataset.astype(int), fmt='%i', delimiter=',')
        

if __name__ == "__main__":
    config = PreprocessorConfig()
    generate(config)
        
        
        