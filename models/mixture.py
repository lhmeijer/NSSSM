# -*- coding: utf-8 -*-

from models.base import Base
from models.distributions.multinomial import CategoricalAndConcentration, Categorical
from models.states.mixture_states import MixtureStates


class Mixture(Base):
    
    _name = 'Mixture'
    
    _states_class = MixtureStates
    _trans_conc_class = CategoricalAndConcentration
    _trans_class = Categorical
    
    
    def __init__(self, obs_distns, trans_distn=None, alpha_0=None, a_0=None, b_0=None, weights=None):
        
        if trans_distn is not None:
            trans_distn = trans_distn
        elif None not in (a_0, b_0):
            trans_distn = self._trans_conc_class(K=len(obs_distns), a_0=a_0, b_0=b_0, weights=weights)
        else:
            trans_distn = self._trans_class(K=len(obs_distns), alpha_0=alpha_0, weights=weights)
            
        super(Mixture, self).__init__(obs_distns=obs_distns, trans_distn=trans_distn)
        
    def resample_trans_distn(self):
        self.trans_distn.resample([s.state_seqs[s.masks].flatten() for s in self.states_list])
        self._clear_caches