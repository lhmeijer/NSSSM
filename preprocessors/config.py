# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
from datetime import datetime
import posixpath
# import pyreadstat
# import pandas as pd
from configuration import Config


REFERENCE_YEAR = datetime.strptime('2010', '%Y')
TOTAL_YEARS = 11

fileConfigurations = {'INHATAB': {'variables': ['RINPERSOONHKW', 'INHBESTINKH', 'INHAHLMI', 'INHBBIHJ', 'INHSAMHH'],
                                  'versions': {'2011': '2', '2012': '2', '2013': '2', '2014': '1', '2015': '1',
                                               '2016': '2', '2017': '2', '2018': '2', '2019': '2', '2020': '2', '2021': '1'}},
                      'GBAPERSOONTAB': {'variables': ['rinpersoon', 'gbageboortejaar', 'gbageneratie'],
                                        'versions': {'2020': '3', '2021': '1'}},
                      'HOOGSTEOPLTAB': {'variables': ['RINPERSOON', 'OPLNIVSOI2021AGG4HBmetNIRWO'],
                                        'versions': {'2020': '2', '2021': '1'}}
                      }

class PreprocessorConfig(Config):
    
    MASS_IN_PERCENTILES = [0.0, 0.0, 0.12, 0.11, 0.10, 0.10, 0.1, 0.1, 0.1, 0.1, 0.09, 0.08, 0.0]
    
    # Variables to generate new data
    N_PER_GROUP = 10
    
    
    income_source_file = posixpath.join('H:', 'Lisa', 'data', 'input', 'income_source.txt')
    raw_primos_input_file = posixpath.join('H:', 'Lisa', 'data', 'input', 'HhoNatio_P21.csv')
    
    # statistics input file
    single_statistics_file = posixpath.join('H:', 'Lisa', 'data', 'statistics', 'single_statistics_H106149.csv')
    multiple_statistics_file = posixpath.join('H:', 'Lisa', 'data', 'statistics', 'multiple_statistics.csv')
    transition_statistics_file = posixpath.join('H:', 'Lisa', 'data', 'statistics', 'transitions_statistics.csv')
    
    @staticmethod
    def _get_data_csv(file_name):
        reader = pd.read_csv(file_name, chunksize=1000000)
        return reader
    
    @staticmethod
    def _get_data_spss(file_name, variables):
        reader = pyreadstat.read_file_in_chunks(pyreadstat.read_sav, file_name, chunksize=100000, usecols=variables, formats_as_category=False, disable_datetime_conversion=True)
        return reader
    
    @staticmethod
    def _get_data_stata(file_name, variables):
        reader = pyreadstat.read_file_in_chunks(pyreadstat.read_dta, file_name, chunksize=100000, usecols=variables, formats_as_category=False, disable_datetime_conversion=True)
        return reader

    @staticmethod
    def get_INHATAB(year):
        fileName = 'INHA{0}TABV{1}.sav'.format(year, fileConfigurations['INHATAB']['versions'][year])
        return PreprocessorConfig._get_data_spss(os.path.join('G:', 'InkomenBestedingen', 'INHATAB', fileName),
                              fileConfigurations['INHATAB']['variables'])
    
    @staticmethod
    def get_GBAPERSOONTAB(year):
        fileName = 'GBAPERSOON{0}TABV{1}.dta'.format(year, fileConfigurations['GBAPERSOONTAB']['versions'][year])
        return PreprocessorConfig._get_data_stata(os.path.join('G:', 'Bevolking', 'GBAPERSOONTAB', year, 'geconverteerde data', fileName),
                              fileConfigurations['GBAPERSOONTAB']['variables'])
    
    @staticmethod
    def get_HOOGSTEOPLTAB(year):
        fileName = 'HOOGSTEOPL{0}TABV{1}.sav'.format(year, fileConfigurations['HOOGSTEOPLTAB']['versions'][year])
        return PreprocessorConfig._get_data_spss(os.path.join('G:', 'Onderwijs', 'HOOGSTEOPLTAB', year, fileName),
                              fileConfigurations['HOOGSTEOPLTAB']['variables'])

