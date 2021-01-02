"""
A stage to change inelasticity distribution in multiple [energy] bins.
Intended for the inelasticity measurement.

This particular stage implements simple inelasticity reweighting sheme
previusely used in high-energy inelasticity measurement.
Note that this parameterization is not really phenomenologically 
motivated at energies below ~1TeV (because of relatively big 
contribution from events with high Bjorken x)!

WARNING: at the moment initial inelasticity distribution 
parametrization is wrong! It is there temporarily for 
technical purposes and will be soon replaced with a proper 
parametrization.

Maria Liubarska
"""

import numpy as np
import pickle
from numba import guvectorize

from pisa.core.stage import Stage
from pisa.utils.resources import open_resource
from pisa.utils import vectorizer
from pisa.utils.log import logging
from pisa import FTYPE, TARGET
from pisa.utils.numba_tools import WHERE

class inelasticity_binned_simple(Stage):
    """
    blah

    Parameters
    ----------
    inelasticity_ana_binning : MultiDimBinning
        Analysis binning
    
    params : ParamSet or sequence with which to instantiate a ParamSet.
        Expected params .. ::
        
            mean_y : quantity (dimensionless)
                Mean inelasticity for a given energy

            lambda : quantity (dimensionless)
                Another parameter (see 
                https://icecube.wisc.edu/~gabinder/inel/docs/build/html/inel.html#parametrization)

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
        expected_params = ('mean_y_bin1', 'lambda_bin1')
        # if there are more, add them:
        if self.num_bins > 1:
            for ibin in range(2,self.num_bins+1):
                expected_params += ('mean_y_bin%d' % ibin, 'lambda_bin%d' % ibin) 
                
        # init base class
        super(inelasticity_binned_simple, self).__init__(
            expected_params=expected_params,
            **std_kwargs,
        )
        
    def setup_function(self):
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
                # first iteration - using the same parametrization with vals for 
                # the lowest energy bin from IC inelasticity measurement
                # hardcoding this for now
                # TODO: change to actual good parametrization of initial GENIE dp/dy
                dis_mask = (container['dis'] > 0)
                lambda_initial = 1.06
                mean_y_initial = 0.42
                epsilon_initial = calc_epsilon_from_mean_y(mean_y_initial, lambda_initial)
                norm_initial = calc_bin_norm(lambda_initial, epsilon_initial)
                
                calc_output = np.ones(container["initial_inelasticity_distr"][dis_mask].size, dtype=FTYPE)
                calc_new_inelasticity_distr(
                    container['bjorken_y'][dis_mask],
                    FTYPE(lambda_initial),
                    FTYPE(epsilon_initial),
                    FTYPE(norm_initial),
                    out=calc_output,
                )
                container["initial_inelasticity_distr"][dis_mask] = calc_output
        
    def compute_function(self):
        # (re)calculate corrections to inelasticity distribution in each bin
        # have to do it every time input param change
        for container in self.data:
            # for now reweighting all CC events
            # TODO: figure out what events to reweight (numu(bar) CC, all CC or all CC+NC?)
            if container.name.endswith('_cc'):
                for ibin in range(self.num_bins):
                    ibin_mask = (container["energy_bin"] == ibin+1.)
                    comp_mask = (ibin_mask * (container['dis'] > 0))
                    
                    lambda_bin = getattr(self.params, 'lambda_bin%d' % (ibin+1)).value.m_as("dimensionless")
                    mean_y_bin = getattr(self.params, 'mean_y_bin%d' % (ibin+1)).value.m_as("dimensionless")
                    
                    epsilon_bin = calc_epsilon_from_mean_y(mean_y_bin, lambda_bin)
                    norm_bin = calc_bin_norm(lambda_bin, epsilon_bin)
                    
                    calc_output = np.ones(container["new_inelasticity_distr"][comp_mask].size, dtype=FTYPE)
                    calc_new_inelasticity_distr(
                        container['bjorken_y'][comp_mask],
                        FTYPE(lambda_bin),
                        FTYPE(epsilon_bin),
                        FTYPE(norm_bin),
                        out=calc_output,
                    )
                    container["new_inelasticity_distr"][comp_mask] = calc_output
                    
    def apply_function(self):
        # modify weights
        for container in self.data:
            # for now reweighting all CC events
            # TODO: figure out what events to reweight (numu(bar) CC, all CC or all CC+NC?)
            if container.name.endswith('_cc'):
                apply_inelasticity_correction(
                    container["new_inelasticity_distr"],
                    container["initial_inelasticity_distr"],
                    out=container["weights"]
                )
        
def calc_epsilon_from_mean_y(mean_y, lam):
    return -1. * ((lam+2.)*(lam+3.)/2.) * ((mean_y*(lam+1.)-lam)/(mean_y*(lam+3.)-lam))

def calc_bin_norm(lam, eps):
    return (lam*(lam+1.)*(lam+2.)) / (2.*eps+(lam+1.)*(lam+2.))  
    
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