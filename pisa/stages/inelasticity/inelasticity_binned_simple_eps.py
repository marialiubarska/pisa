"""
A stage to change inelasticity distribution in multiple [energy] bins.
Intended for the inelasticity measurement.

This particular stage implements simple inelasticity reweighting sheme
previusely used in high-energy inelasticity measurement.
Note that this parameterization is not really phenomenologically 
motivated at energies below ~1TeV (because of relatively big 
contribution from events with high Bjorken x)!

Maria Liubarska
"""

import numpy as np
import pickle
from numba import guvectorize

from pisa.core.stage import Stage
from pisa.utils.resources import open_resource
from pisa.utils import vectorizer
from pisa.utils.fileio import from_file
from pisa.utils.log import logging
from pisa import FTYPE, TARGET
from pisa.utils.numba_tools import WHERE

class inelasticity_binned_simple_eps(Stage):
    """
    blah

    Parameters
    ----------
    inelasticity_ana_binning : MultiDimBinning
        Analysis binning
    
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params .. ::
        
            epsilon : quantity (dimensionless)
                param

            lambda : quantity (dimensionless)
                Another parameter (see 
                https://icecube.wisc.edu/~gabinder/inel/docs/build/html/inel.html#parametrization )

    """
    def __init__(
            self,
            inelasticity_ana_binning=None,
            **std_kwargs,
    ):
        
        
        self.num_bins = inelasticity_ana_binning.reco_energy.num_bins
        self.bin_edges = np.array(inelasticity_ana_binning.reco_energy.bin_edges)
        
        logging.debug(
                'Initializing inelasticity_binned_simple stage with %d bins.' % self.num_bins)

        # modify expected_params to have as many sets of parameters 
        # as there are energy bins
        # there is at least one bin:
        expected_params = (
                           'epsilon_bin1', 
                           'lambda_bin1', 
                           'binnorm_bin1',
                          )
        # if there are more, add them:
        if self.num_bins > 1:
            for ibin in range(2,self.num_bins+1):
                if ibin<self.num_bins:
                    expected_params += ('epsilon_bin%d' % ibin, 
                                        'lambda_bin%d' % ibin,
                                        'binnorm_bin%d' % ibin
                                       ) 
                else:
                    expected_params += ('epsilon_bin%d' % ibin, 
                                        'lambda_bin%d' % ibin,
                                       ) 
                
        # init base class
        super(inelasticity_binned_simple_eps, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )
        
    def setup_function(self):
        
        # load initial inelasticity distribution splines
        init_y_disrt_splines = from_file('inelasticity/initial_genie_y_distr_fit.pckl')
        spl_nucc = init_y_disrt_splines['nu_cc']
        spl_nubarcc = init_y_disrt_splines['nubar_cc']
        
        # create empty containers for weight corrections
        self.data.representation = self.apply_mode
        for container in self.data:
            # for now reweighting all CC events
            # TODO: figure out what events to reweight (numu(bar) CC, all CC or all CC+NC?)
            if container.name.endswith('_cc'):
                container["energy_bin"] = np.zeros(container.size, dtype=FTYPE)
                container["new_inelasticity_distr"] = np.ones(container.size, dtype=FTYPE)
                container["initial_inelasticity_distr"] = np.ones(container.size, dtype=FTYPE)
                
                for ibin in range(self.num_bins):
                    ibin_mask = ((container["reco_energy"] > self.bin_edges[ibin])*
                                 (container["reco_energy"] <= self.bin_edges[ibin+1]))
                    container["energy_bin"][ibin_mask] +=  1. + ibin
                    
                # storing initial true inelasticity distribution 
                # will later use it to devide by it
                # TODO: make parametrization with E<100 GeV included
                
                if container.name in ['nue_cc', 'numu_cc', 'nutau_cc']:
                    init_y_spl = spl_nucc
                elif container.name in ['nuebar_cc', 'numubar_cc', 'nutaubar_cc']:
                    init_y_spl = spl_nubarcc
                else:
                    raise ValueError('Incorrect container type "%s"' % container.name)
                
                dis_mask = (container['dis'] > 0)
                lg_true_energy_dis = np.log10(container["true_energy"])
                true_y_dis = container['bjorken_y']
                
                lgE_min = 2. # 100 GeV
                valid_mask = (dis_mask*(lg_true_energy_dis >= lgE_min))
                extrp_mask = (dis_mask*(~valid_mask))
                
                initial_inelasticity_distr_valid_erange   = init_y_spl.ev(lg_true_energy_dis[valid_mask],
                                                                          true_y_dis[valid_mask])
                initial_inelasticity_distr_extrap_erange = init_y_spl.ev(lgE_min,
                                                                          true_y_dis[extrp_mask])
                
                container["initial_inelasticity_distr"][valid_mask] = initial_inelasticity_distr_valid_erange
                container["initial_inelasticity_distr"][extrp_mask] = initial_inelasticity_distr_extrap_erange
                
    def compute_function(self):
        # (re)calculate corrections to inelasticity distribution in each bin
        # have to do it every time input param change
        for container in self.data:
            # for now reweighting all CC events
            # TODO: figure out what events to reweight (numu(bar) CC, all CC or all CC+NC?)
            if container.name.endswith('_cc'):
                
                total_num_events = 0.
                modif_num_events = 0.
                for ibin in range(self.num_bins):
                    ibin_mask = (container["energy_bin"] == ibin+1.)
                    comp_mask = (ibin_mask * (container['dis'] > 0))
                    
                    lambda_bin = getattr(self.params, 'lambda_bin%d' % (ibin+1)).value.m_as("dimensionless")
                    
                    epsilon_bin = getattr(self.params, 'epsilon_bin%d' % (ibin+1)).value.m_as("dimensionless")
                    
                    # apparently setting out to masked array doesn't work?
                    calc_output = np.ones(container["new_inelasticity_distr"][comp_mask].size, dtype=FTYPE)
                    calc_new_inelasticity_distr(
                        container['bjorken_y'][comp_mask],
                        FTYPE(lambda_bin),
                        FTYPE(epsilon_bin),
                        FTYPE(1.), # FTYPE(norm_bin),
                        out=calc_output,
                    )
                    container["new_inelasticity_distr"][comp_mask] = calc_output
                    
    def apply_function(self):
        # modify weights
        for container in self.data:
            # for now reweighting all CC events
            # TODO: figure out what events to reweight (numu(bar) CC, all CC or all CC+NC?)
            if container.name.endswith('_cc'):
                if sum(container["weights"]) == 0.: continue
                total_num_events = 0.
                modif_num_events = 0.
                for ibin in range(self.num_bins):
                    ibin_mask = (container["energy_bin"] == ibin+1.)
                    comp_mask = (ibin_mask * (container['dis'] > 0))
                    
                    exp_ev_count = sum(container["weights"][comp_mask])
                    mod_ev_count_init = sum(container["weights"][comp_mask]/container["initial_inelasticity_distr"][comp_mask])
                    unmodified_weights_init = (container["weights"][comp_mask]/container["initial_inelasticity_distr"][comp_mask])*(exp_ev_count/mod_ev_count_init)
                    
                    total_num_events += exp_ev_count
                    if ibin < self.num_bins-1:
                        norm_bin = getattr(self.params, 'binnorm_bin%d' % (ibin+1)).value.m_as("dimensionless")
                        modif_num_events += norm_bin * exp_ev_count
                    else:
                        # last bin
                        norm_bin = max(0, (total_num_events - modif_num_events) / exp_ev_count)
                        
                    modified_weights = norm_bin * container["new_inelasticity_distr"][comp_mask] * unmodified_weights_init
                    container["weights"][comp_mask] = modified_weights
                    
                
        
# vectorized functions
# must be outside class
if FTYPE == np.float64:
    FX = 'f8'
    IX = 'i8'
else:
    FX = 'f4'
    IX = 'i4'
@guvectorize([f'({FX}[:], {FX}, {FX}, {FX}, {FX}[:])'], '(),(),(),()->()', target=TARGET)
def calc_new_inelasticity_distr(y, lam, eps, norm, out):
    out[0] = max(0, (norm * (1. + eps*(1 - y[0])**2) * y[0]**(lam-1)))
    
@guvectorize([f'({FX}[:], {FX}[:], {FX}[:])'], '(),()->()', target=TARGET)
def apply_inelasticity_correction(new_y_distr_w, old_y_distr_w, out):
    out[0] *= new_y_distr_w[0]/old_y_distr_w[0]