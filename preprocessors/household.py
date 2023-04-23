# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from .configuration import TOTAL_YEARS, REFERENCE_YEAR
import numpy as np
from datetime import datetime


class Household(object):
    
    __slots__ = ('_rin_number', '_start_index', '_composition', '_education_level',
                 '_change_in_composition', '_age', '_generation', '_income',
                 '_income_source', '_change_in_income_source')

    def __init__(self, rin_number, start_index):
        self._rin_number = rin_number
        self._start_index = start_index
        
        self._composition = np.full(TOTAL_YEARS, None)
        self._change_in_composition = np.full(TOTAL_YEARS, None)
        
        self._age = np.full(TOTAL_YEARS, None)
        self._education_level = np.full(TOTAL_YEARS, None)
        self._generation = np.full(TOTAL_YEARS, None)
        self._income = np.full(TOTAL_YEARS, None)
        
        self._income_source = np.full(TOTAL_YEARS, None)
        self._change_in_income_source = np.full(TOTAL_YEARS, None)
        
    @property
    def rin_number(self):
        return self._rin_number
    
    @property
    def start_index(self):
        return self._start_index
    
    def set_generation(self, generation):
        if generation != '-':
            self._generation[self.start_index:] = int(generation)
             
    def get_generation(self, index=None):
        if index is None:
            return self._generation
        return self._generation[index]
    
    def set_age(self, year):
        age = REFERENCE_YEAR.year - datetime.strptime(year, '%Y').year
        age += self.start_index
        for t in range(self.start_index, TOTAL_YEARS):
            if age >= 20:
                self._age[t] = int(age)
            age += 1
            
    def get_age(self, index=None):
        if index is None:
            return self._age
        return self._age[index]
    
    def set_education_level(self, education_level):
        if '1' == str(education_level)[0]:
            self._education_level[self.start_index:] = 0   # Low education level
        elif '2' == str(education_level)[0]:
            self._education_level[self.start_index:] = 1   # Middle education level
        elif '3' == str(education_level)[0]:
            self._education_level[self.start_index:] = 2   # High education level
            
    def get_education_level(self, index=None):
        if index is None:
            return self._education_level
        return self._education_level[index]

    def set_income(self, income, index):
        if income != 9999999999.0 and income >= 0:
            self._income[index] = int(income)
    
    def get_income(self, index=None):
        if index is None:
            return self._income
        return self._income[index]

    def set_composition(self, composition, index):
        if composition in ['11', '12', '13', '14']:  # Alone
            self._composition[index] = 0
        elif composition in ['41', '42', '43', '55', '56', '57']:     # 1 parent family 
            self._composition[index] = 1
        elif composition in ['21', '22', '51', '58']:     # together without children
            self._composition[index] = 2 
        elif composition in ['31', '32', '33', '52', '53', '54']:     # together with children
            self._composition[index] = 3
        else :
            self._composition[index] = 4
            
        if index > 0:
            self.set_change_in_composition(index)
            
    def get_composition(self, index=None):
        if index is None:
            return self._composition
        return self._composition[index]
    
    def set_change_in_composition(self, index):
        c = self._composition[index]
        p = self._composition[index-1]
        if c is not None and p is not None:
            if c == p:
                self._change_in_composition[index] = 0
            else:
                number = c + 1 if p < c else c
                self._change_in_composition[index] = (5 * p) + number
                
    def get_change_in_composition(self, index=None):
        if index is None:
            return self._change_in_composition
        return self._change_in_composition[index]
    
    def set_income_source(self, income_source, n_with_income, index):
        if income_source in ['11', '12', '13', '14']:
            if n_with_income > 1:
                self._income_source[index] = 1
            else:
                self._income_source[index] = 0
        elif income_source in ['21', '22', '23', '24', '25', '26']:
            self._income_source[index] = 2
        
        if index > 0:
            self.set_change_in_income_source(index)
    
    def get_income_source(self, index=None):
        if index is None:
            return self._income_source
        return self._income_source[index]
                
    def set_change_in_income_source(self, index):
        c = self._income_source[index]
        p = self._income_source[index-1]
        if c is not None and p is not None:
            if c == p:
                self._change_in_income_source[index] = 0
            else:
                number = c + 1 if p < c else c
                self._change_in_income_source[index] = (3 * p) + number
                
    def get_change_in_income_source(self, index=None):
        if index is None:
            return self._change_in_income_source
        return self._change_in_income_source[index]
    
    def headers(self):
        return ["RINPERSOON_HKW", "YEAR", "INCOME", "AGE", "EDUCATION_LEVEL", "COMPOSITION", 
                "INCOME_SOURCE", "GENERATION"]

    def to_list(self, index):    
        year = int(REFERENCE_YEAR.year + index)
        to_list = [self._rin_number, year, self._income[index], self._age[index],
                   self._education_level[index], self._composition[index],
                   self._income_source[index], self._generation[index]]

        if None in to_list:
            to_list = [-1 for _ in range(len(to_list))]
            to_list[0] = self._rin_number
            to_list[1] = year

        return to_list