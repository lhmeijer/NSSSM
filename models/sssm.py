import numpy as np
import models.hmm as hmm
import models.states.sssm_states as states


class _SwitchingSSM(object):
    
    _states_class = states.SwitchingSSMStates
    
    def __init__(self, dynamics_distns, emission_distns, init_dynamics_distns, **kwargs):        
        self.init_dynamics_distns = init_dynamics_distns
        self.dynamics_distns = dynamics_distns
        
        if not isinstance(emission_distns, list):
            self.single_emission = True
            self._emission_distn = emission_distns
            self.emission_distns = [emission_distns] * len(dynamics_distns)
        else:
            assert len(emission_distns) == len(dynamics_distns)
            self.single_emission = False
            self.emission_distns = emission_distns
            
        super(_SwitchingSSM, self).__init__(obs_distns=self.dynamics_distns, **kwargs)

    def resample_parameters(self):
        self.resample_ssm_parameters()
        self.resample_hmm_parameters()
        
    def resample_hmm_parameters(self):
        super(_SwitchingSSM, self).resample_parameters()
    
    def resample_ssm_parameters(self):
        self.resample_init_dynamics_distns()
        self.resample_dynamics_distns()
        self.resample_emission_distns()
        
    def resample_init_dynamics_distns(self):
        # print("Resample initial dynamics distributions")
        for state, distn in enumerate(self.init_dynamics_distns):
            distn.resample([s.gaussian_states[0, -1, s.state_seqs[0]==state] for s in self.states_list])
            # distn.resample([s.gaussian_states[0, s.state_seqs[0]==state] for s in self.states_list])
        self._clear_caches()
            
    def resample_dynamics_distns(self):
        y_differences = [np.diff(s.state_seqs+1, axis=0, prepend=10) for s in self.states_list]
        x_differences = [np.roll(y_difference, -1, axis=0)[:-1] for y_difference in y_differences]
        for state, distn in enumerate(self.dynamics_distns):
            indexes = [s.state_seqs == state for s in self.states_list]
            x_indexes = [idxs[:-1] & (x_difference == 0) for idxs, x_difference in zip(indexes, x_differences)]
            y_indexes = [idxs[1:] & (y_difference[1:] == 0) for idxs, y_difference in zip(indexes, y_differences)]
            ys = [s.gaussian_states[1:,-1][y_idxs] for y_idxs, s in zip(y_indexes, self.states_list)]
            xs = [np.hstack((s.gaussian_states[:-1, -1][x_idxs], s.input_data[:-1][x_idxs])) for x_idxs, s in zip(x_indexes, self.states_list)]
            distn.resample([(x, y) for x, y in zip(xs, ys)])

        self._clear_caches()
        
    def resample_emission_distns(self):
        # print("Resample emission distributions")
        if self.single_emission:
            # data = [(np.hstack((s.gaussian_states[s.masks], 
            #                     s.input_data[s.masks])), 
            #          s.data[s.masks, None]) for s in self.states_list]
            data = [(np.hstack((s.gaussian_states[:, -1][s.masks], 
                                s.input_data[s.masks])), 
                     s.data[s.masks, None]) for s in self.states_list]
            self._emission_distn.resample(data=data)
        else:
            for state, distn in enumerate(self.emission_distns):
                # data = [(np.dstack((s.gaussian_states, s.input_data))[s.state_seqs == state],
                #          s.data[s.state_seqs == state, None])
                #         for s in self.states_list]
                data = [(np.dstack((s.gaussian_states[:, -1], s.input_data))[s.state_seqs == state],
                         s.data[s.state_seqs == state, None])
                        for s in self.states_list]
                distn.resample(data=data)
        self._clear_caches()
       
    def resample_obs_distns(self):
        pass
    
    @property
    def params(self):
        params = {'name': self.__class__.__name__, 
                  'params': {
                      'init_dynamics_distns': [distn.params for distn in self.init_dynamics_distns],
                      'dynamics_distns': [distn.params for distn in self.dynamics_distns],
                      'emission_distns': self._emission_distn.params if self.single_emission else [distn.params for distn in self.emission_distns],
                      'trans_distn': self.trans_distn.params
                      }
        }
        return params
    
    
################################
# Switching State Space models #
################################

class SwitchingSSM(_SwitchingSSM, hmm.HMM):
    pass


class WeakLimitHDPHMMSwitchingSSM(_SwitchingSSM, hmm.WeakLimitHDPHMM):
    pass


class WeakLimitStickyHDPHMMSwitchingSSM(_SwitchingSSM, hmm.WeakLimitStickyHDPHMM):
    pass


class MultinomialLogisticInputSwitchingSSM(_SwitchingSSM, hmm.MultinomialLogisticInputHMM):
    _states_class = states.SwitchingSSMInputStates
    
    
class RegularizedMultinomialLogisticInputSwitchingSSM(_SwitchingSSM, hmm.RegularizedMultinomialLogisticInputHMM):
    _states_class = states.SwitchingSSMInputStates
    

class MultinomialLogisticInputOnlySwitchingSSM(_SwitchingSSM, hmm.MultinomialLogisticInputOnlyHMM):
    _states_class = states.SwitchingSSMInputStates


class RegularizedMultinomialLogisticInputOnlySwitchingSSM(_SwitchingSSM, hmm.RegularizedMultinomialLogisticInputOnlyHMM):
    _states_class = states.SwitchingSSMInputStates


class StickBreakingLogisticInputSwitchingSSM(_SwitchingSSM, hmm.StickBreakingLogisticInputHMM):
    _states_class = states.SwitchingSSMNonParametricInputStates
    

class StickBreakingLogisticInputOnlySwitchingSSM(_SwitchingSSM, hmm.StickBreakingLogisticInputOnlyHMM):
    _states_class = states.SwitchingSSMNonParametricInputStates
    
    
class TruncatedStickBreakingLogisticInputOnlySwitchingSSM(_SwitchingSSM, hmm.StickBreakingLogisticInputOnlyHMM):
    _states_class = states.SwitchingSSMTruncatedInputStates
   
    
class RegularizedTruncatedStickBreakingLogisticInputOnlySwitchingSSM(_SwitchingSSM, hmm.RegularizedStickBreakingInputOnlyHMM):
    _states_class = states.SwitchingSSMTruncatedInputStates
   
    
class RegularizedStickBreakingLogisticInputSwitchingSSM(_SwitchingSSM, hmm.RegularizedStickBreakingInputHMM):
    _states_class = states.SwitchingSSMNonParametricInputStates    


class RegularizedStickBreakingLogisticInputOnlySwitchingSSM(_SwitchingSSM, hmm.RegularizedStickBreakingInputOnlyHMM):
    _states_class = states.SwitchingSSMNonParametricInputStates
    

                    
            