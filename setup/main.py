# -*- coding: utf-8 -*-
from hera.config import HeraConfig
from hera.estimator import Estimator
from hera.model import Model
import models.configs.baseline_config as baseline
import models.configs.hmm_config as hmm
import models.configs.sssm_config as sssm
import numpy as np
from datetime import datetime


if __name__ == "__main__":
    
    np.set_printoptions(precision=5)
    end_year = datetime.strptime('2019', '%Y').year
    start_year = datetime.strptime('2011', '%Y').year
    
    # model_configs = [sssm.SSSMTruncatedK10StickBreakingConfig()]
    #                  
    #                  sssm.SSSMStickBreakingConfig()]
    # model_configs = [baseline.BaselineFixedClustersConfig()]
    # model_configs = [hmm.HMMK6RegularizedConfig()]
    # model_configs = [sssm.SSSMTruncatedK10StickBreakingRegularizedConfig(), sssm.SSSMStickBreakingConfig()] # nog berekenen voor andere mcmc waardes
    
        
    # model_configs = [sssm.SSSMK10Config(), sssm.SSSMK10RegularizedConfig()]
    # model_configs = [sssm.SSSMTruncatedK10StickBreakingConfig()]
    model_configs = [sssm.SSSMK10Config(), sssm.SSSMK10RegularizedConfig(), sssm.SSSMTruncatedK10StickBreakingConfig()]
    for model_config in model_configs:
        model = Model(config=model_config)
        config = HeraConfig()
        estimator = Estimator(config=config, model=model)
        # estimator.forecast(start_year=start_year, end_year=datetime.strptime('2030', '%Y').year)
        # estimator.fit(start_year=start_year, end_year=end_year)
        # estimator.evaluate_one_step_ahead_forecast(start_year=start_year, end_year=end_year,
        #                                             prev_y_known=False)
        # estimator.evaluate_one_step_ahead_forecast(start_year=start_year, end_year=end_year,
        #                                             prev_y_known=True)
        estimator.evaluate_predictions(start_year=start_year, end_year=end_year)
        


        
