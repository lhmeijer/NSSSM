# -*- coding: utf-8 -*-
from models.base import Base, InputBase, NonParametricInputBase
import models.states.hmm_states as hmm_states
import models.transitions.hmm_transitions as transitions


class HMM(Base):
    
    _states_class = hmm_states.HMMStates
    _trans_class = transitions.HMMTransitions
    _trans_conc_class = transitions.HMMTransitionsConc
    
    def __init__(self, obs_distns, trans_distn=None, alpha=None, alpha_a_0=None,
                 alpha_b_0=None, trans_matrix=None, init_trans_vec=None):
        
        if trans_distn is not None:
            trans_distn = trans_distn
        elif None not in (alpha_a_0, alpha_b_0):
            trans_distn = self._trans_conc_class(
                num_states=len(obs_distns), alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0, 
                trans_matrix=trans_matrix, init_trans_vec=init_trans_vec)
        else:
            trans_distn = self._trans_class(
                num_states=len(obs_distns), alpha=alpha, 
                trans_matrix=trans_matrix, init_trans_vec=init_trans_vec)
        super(HMM, self).__init__(obs_distns=obs_distns, trans_distn=trans_distn)
        
        
class WeakLimitHDPHMM(Base):

    _name = 'WeakLimitHDPHMM'

    _states_class = hmm_states.HMMStates
    _trans_class = transitions.WeakLimitHDPHMMTransitions
    _trans_conc_class = transitions.WeakLimitHDPHMMTransitionsConc

    def __init__(self, obs_distns, transition_distn=None, alpha=None, alpha_a_0=None,
                 alpha_b_0=None, gamma=None, gamma_a_0=None, gamma_b_0=None,
                 transition_matrix=None, init_transition_vector=None):

        if transition_distn is not None:
            transition_distn = transition_distn
        elif not None in (alpha_a_0, alpha_b_0):
            transition_distn = self._trans_conc_class(
                num_states=len(obs_distns), alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0, gamma_a_0=gamma_a_0,
                gamma_b_0=gamma_b_0, transition_matrix=transition_matrix, init_transition_vector=init_transition_vector)
        else:
            transition_distn = self._trans_class(
                num_states=len(obs_distns), alpha=alpha, gamma=gamma,
                transition_matrix=transition_matrix, init_transition_vector=init_transition_vector)

        super(WeakLimitHDPHMM, self).__init__(obs_distns=obs_distns, transition_distn=transition_distn)


class WeakLimitStickyHDPHMM(object):

    _name = 'WeakLimitStickyHDPHMM'

    _states_class = hmm_states.HMMStates
    _trans_class = transitions.WeakLimitStickyHDPHMMTransitions
    _trans_conc_class = transitions.WeakLimitStickyHDPHMMTransitionsConc

    def __init__(self, obs_distns, trans_distn, kappa=None, alpha=None,
                 gamma=None, transition_matrix=None, init_transition_vector=None,
                 alpha_a_0=None, alpha_b_0=None, gamma_a_0=None, gamma_b_0=None):

        if trans_distn is not None:
            trans_distn = trans_distn
        elif None not in (alpha, gamma):
            trans_distn = self._trans_class(
                num_states=len(obs_distns), kappa=kappa, alpha=alpha,
                gamma=gamma, transition_matrix=transition_matrix, init_transition_vector=init_transition_vector)
        else:
            trans_distn = self._trans_conc_class(
                num_states=len(obs_distns), kappa=kappa, alpha_a_0=alpha_a_0,
                alpha_b_0=alpha_b_0, gamma_a_0=gamma_a_0, gamma_b_0=gamma_b_0,
                transition_matrix=transition_matrix, init_transition_vector=init_transition_vector)

        super(WeakLimitStickyHDPHMM, self).__init__(obs_distns=obs_distns, trans_distn=trans_distn)


class MultinomialLogisticInputHMM(InputBase):

    _trans_class = transitions.HMMInputTransitions
    _states_class = hmm_states.HMMInputStates

    def __init__(self, obs_distns, trans_distn=None, transition_params=None, D_in=0):

        if trans_distn is not None:
            trans_distn = trans_distn
        else:
            trans_distn = self._trans_class(num_states=len(obs_distns), covariates_dim=D_in, **transition_params)

        super(MultinomialLogisticInputHMM, self).__init__(obs_distns, trans_distn, D_in)
        
class RegularizedMultinomialLogisticInputHMM(MultinomialLogisticInputHMM):
    _trans_class = transitions.RegularizedHMMInputTransitions
        

class MultinomialLogisticInputOnlyHMM(MultinomialLogisticInputHMM):
    _trans_class = transitions.HMMInputOnlyTransitions
    

class RegularizedMultinomialLogisticInputOnlyHMM(MultinomialLogisticInputHMM):
    _trans_class = transitions.RegularizedHMMInputOnlyTransitions


class StickBreakingLogisticInputHMM(NonParametricInputBase):
    
    _trans_class = transitions.StickBreakingHMMInputTransitions
    _states_class = hmm_states.NonParametricHMMInputStates
    
    def __init__(self, obs_distns, trans_distn=None, transition_params=None, D_in=0):

        if trans_distn is not None:
            trans_distn = trans_distn
        else:
            trans_distn = self._trans_class(num_states=len(obs_distns), covariates_dim=D_in, 
                                            **transition_params)

        super(StickBreakingLogisticInputHMM, self).__init__(obs_distns, trans_distn, D_in)


class StickBreakingLogisticInputOnlyHMM(StickBreakingLogisticInputHMM): 
    _trans_class = transitions.StickBreakingHMMInputOnlyTransitions
    # _states_class = hmm_states.NonParametricHMMInputStates


class RegularizedStickBreakingInputHMM(StickBreakingLogisticInputHMM):
    _trans_class = transitions.RegularizedStickBreakingHMMInputTransitions


class RegularizedStickBreakingInputOnlyHMM(StickBreakingLogisticInputHMM):
    _trans_class = transitions.RegularizedStickBreakingHMMInputOnlyTransitions



