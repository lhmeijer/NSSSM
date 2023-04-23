# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:41:50 2022

@author: lisa
"""
import numpy as np
from hera.config import HeraConfig


def mean_estimates(prev_estimates, step_size):
    """

    """
    s = step_size / 2.
    bins = np.arange(s + HeraConfig.MINIMUM_INCOME, HeraConfig.MAXIMUM_INCOME, step_size)
    mean_estimaes = np.sum(bins * prev_estimates)
    return mean_estimaes
    # n = prev_estimates.shape[0]
    # estimates = np.zeros(n + 2)
    # estimates[1:n + 1] = prev_estimates

    # level = np.zeros(n)
    # for j in range(n):
    #     idx = j + 1
    #     if estimates[idx] == estimates[idx - 1]:
    #         level[j] = (idx + idx - 1) / 2
    #     elif estimates[idx] == 0 and estimates[idx - 1] > 0:
    #         level[j] = (idx - 1 + 0.25) / 2
    #     else:

    #         a = estimates[idx] - estimates[idx - 1]
    #         b = estimates[idx] - estimates[idx + 1]
    #         c = estimates[idx + 1] - estimates[idx - 1]

    #         if a > 0 and b > 0:
    #             p = b / (a + b)
    #         elif a < 0 and b < 0:
    #             p = (-1 * b) / ((-1 * a) + (-1 * b))
    #         else:
    #             p = a / c

    #         p_l = p / (p + 0.5)
    #         p_r = (1.0 - p) / ((1.0 - p) + 0.5)

    #         level_l = estimates[idx - 1] + p_l * a
    #         level_r = estimates[idx] + p_r * (-1 * b)

    #         threshold = int((1.0 - p) * step_size)
    #         opp_l = ((level_l / estimates[idx]) + 1) * threshold / (2*step_size)
    #         opp_r = ((level_r / estimates[idx]) + 1) * (step_size - threshold) / (2*step_size)

    #         if opp_l < 0.5:
    #             corr = (threshold + (opp_r - 0.5) / opp_r * (step_size - threshold)) / step_size
    #         else:
    #             corr = 0.5 / opp_l * threshold / step_size
    #         level[j] = idx - 1 + corr

    # return np.sum(prev_estimates * level) / np.sum(prev_estimates)


def total_estimates(estimates):
    total_income = 0
    for i in range(estimates.shape[0]):
        total_income += estimates[i] * (i + 1)

    return total_income


def deciles(estimates):
    part_borders, part_values = parts(estimates, 10)
    # Compute the 50/50 ratio and 20/80 ratio
    ratio_80_20 = np.sum(part_values[8:]) / np.sum(part_values[:2])
    ratio_50_50 = np.sum(part_values[5:]) / np.sum(part_values[:5])

    return part_borders, ratio_50_50, ratio_80_20


def parts(estimates, n_parts):
    part_borders = np.zeros(n_parts)
    part_values = np.zeros(n_parts)

    N = estimates.shape[0]
    cum = np.zeros(N + 1)
    sum_estimates = np.sum(estimates)
    i_thres = 0
    for i in range(N):
        cum[i + 1] += cum[i] + estimates[i]
        border = sum_estimates * ((i_thres + 1) / n_parts)
        if cum[i + 1] < border:
            part_values[min(i_thres, n_parts - 1)] += ((i + 1) - 0.5) * estimates[i]
        while cum[i + 1] >= border:
            # print(cum[i + 1])
            # print("border ", border)
            perc = (border - cum[i]) / (cum[i + 1] - cum[i])
            part_borders[min(i_thres, n_parts - 1)] = (i + perc) * HeraConfig.INCOME_STEPS + HeraConfig.MINIMUM_INCOME

            part_value_left = ((i + 1) * perc) * estimates[i]
            # TODO why -0.5?
            part_value_right = (1 - perc) * ((i + 1) - 0.5) * estimates[i]

            part_values[min(i_thres, n_parts - 1)] += part_value_left
            part_values[min(i_thres + 1, n_parts - 1)] += part_value_right

            i_thres += 1
            border = sum_estimates * ((i_thres + 1) / n_parts)

    part_borders[-1] = HeraConfig.MAXIMUM_INCOME

    return part_borders, part_values


def in_income_groups(estimates, group_boundaries):
    n_in = np.zeros_like(group_boundaries)
    perc_in = np.zeros_like(group_boundaries)
    total = np.sum(estimates)
    for idx, g in enumerate(group_boundaries):
        n_in[idx] = estimates[:g]
        perc_in[idx] = n_in = total
    return n_in, perc_in


