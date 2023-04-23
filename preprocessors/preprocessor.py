# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
from src.main.dim.preprocessing2.configuration import TOTAL_YEARS, REFERENCE_YEAR, Configuration
from src.main.dim.preprocessing2.Household import Household
import numpy as np


class Preprocessor(object):

    def __init__(self, config):
        self._config = config
        self._households = {}       # household_number: Household
        
    def _collect_individual_data(self, personal_record):
        """
        Function that returns the household corresponding to RINGNUMMBERHKW
        :param personal_record: RINGNUMMBERHKW
        :return: class object of household if it exists. Otherwise return None
        """
        rin_number_hkw = int(personal_record['rinpersoon'])
        if rin_number_hkw in self._households:
            household = self._households[rin_number_hkw]
            household.set_age(personal_record['gbageboortejaar'])
            household.set_generation(personal_record['gbageneratie'])
        
    def _collect_household_income_data(self, household_record, index):
        """
        Function that collect the household income and assign it to the 
        corresponding household.
        :param householdRecord: RINGNUMMBERHKW
        :param t: year index
        :return: class object of household if it exists. Otherwise return None
        """
        rin_number_hkw = int(household_record['RINPERSOONHKW'])
        if rin_number_hkw in self._households:
            household = self._households[rin_number_hkw]
        else:
            household = Household(rin_number_hkw, index)
            self._households[rin_number_hkw] = household

        household.set_income(household_record['INHBESTINKH'], index)
        household.set_income_source(household_record['INHBBIHJ'], household_record['INHAHLMI'], index)
        household.set_composition(household_record['INHSAMHH'], index)
            
    def _collect_education_data(self, personal_record):
        """
        Function that collects the education level of each individual and 
        set the household education level to the highest level.
        :param personal_record: row of the data
        :return: -
        """
        rin_number_hkw = int(personal_record['RINPERSOON'])
        if rin_number_hkw in self._households:
            household = self._households[rin_number_hkw]
            household.set_education_level(personal_record['OPLNIVSOI2021AGG4HBmetNIRWO']) 
            
    def collect_data(self):
        
        for t in range(TOTAL_YEARS):
            year = REFERENCE_YEAR.year + t
             
            print("Get all household income from the INHATAB.")
            data_reader = self._config.get_INHATAB(year=str(year+1))
            nRows = 0
            for data, _ in data_reader:
                nRows += data.shape[0]
                if nRows % 1000000 == 0:
                    print("nRows ", nRows)
                data.apply(lambda record: 
                            self._collect_household_income_data(record, t), axis=1)            
        
        print("Get all individuals from the GBAPERSOONTAB.")
        data_reader = self._config.get_GBAPERSOONTAB("2021")
        nRows = 0
        for data, _ in data_reader:
            nRows += data.shape[0]
            if nRows % 1000000 == 0:
                print("nRows ", nRows)
            data.apply(lambda personal_record: 
                        self._collect_individual_data(personal_record), axis=1)
            
        print("Get all education levels from the HOOGSTEOPLTAB.")
        data_reader = self._config.get_HOOGSTEOPLTAB("2021")
        nRows = 0
        for data, _ in data_reader:
            nRows += data.shape[0]
            if nRows % 1000000 == 0:
                print("nRows ", nRows)
            data.apply(lambda personal_record: 
                       self._collect_education_data(personal_record), axis=1)
        
        file_name = self._config.get_household_input_file()
        self._data_to_csv_file(file_name, self._households)
                
                  
    def _data_to_csv_file(self, file_name, data):
        with open(file_name, 'a') as f:
            for _, record in data.items():
                record_to_list = np.array([np.array(record.to_list(t)) for t in range(TOTAL_YEARS)])
                sum_income = np.sum(record_to_list[:, 2])
                if sum_income > (-1.0 * TOTAL_YEARS):
                    np.savetxt(f, record_to_list.astype(int), delimiter=',', fmt='%i')
                
        
        
if __name__ == "__main__":
    preprocessor = Preprocessor(config=Configuration())
    preprocessor.collect_data()     
        