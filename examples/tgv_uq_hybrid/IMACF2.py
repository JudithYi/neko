####################################################################
# Incremental Modeled Autocorrelation Function (MACF) Method based on
# 'updating_ts.py' and iMACF.py by Saleh Rezaeiravesh (KTH)
####################################################################
#----------------------------------------------------------------
# Christian Gscheidle, christian.gscheidle@scai.fraunhofer.de
#----------------------------------------------------------------

import numpy as np
import math as mt

#import os, sys
#sys.path.append(os.getenv("UQit_ts_path"))

#from IStats import IMeanVarACF

class IMACF2():
   """
   Incremental Modeled Autocovaraince Function (MACF) Method

   Note: Output (self.out) will only be generated every 'self.output_freq' samples

   Args:
      `uncertOpts`: dict
         Options for IMACF with the following keys:
         * `max_acf_lag`: integer, maximum lag to compute the ACF
         * `sample_freq`: integer, rate to take samples at for ACF
         * `output_freq`: integer, rate at which to compute uncertainty estimate for all points
         * 'maxIter': maximum iterations when curve fitting for modeled ACF, default=50000
         * 'nMax': maximum number of samples at which modeled ACF is evaluated, default=100000 

   Methods:
      `update_stats` : incremental update of mean, var, acf 

   """
   def __init__(self, uncertOpts):

      # list of lags
      self.m_list = None
      # only store the
      self.mean_var_acf = None

      self.var_mu = None
      self.var_mu_n = None
      self.uncertOpts = uncertOpts

      # init incremental batch mean
      max_acf_lag_raw = int( uncertOpts['max_acf_lag'] / uncertOpts['sample_freq'] )

      # init incremental batch mean
      #self.mean_var_acf = IMeanVarACF(with_var=True, with_acf=True, max_acf_lag=max_acf_lag_raw)

      # continuous sample count
      self.sample_count = 0

      # list of lags
      self.m_list = None

      # uncert estimate
      self.var_mu = None
      self.var_mu_n = None

      # coarse samples
      self.reduce_lags = uncertOpts.get('reduce_lags', 1)

      # numerical settings for curve_fit
      self.maxIter = uncertOpts.get('maxIter', 50000)

      # cache acf_model (only for validation)
      self.store_acf_model = uncertOpts.get('store_acf_model', False)
      self.cache_acf_model = []

      # additional setting
      self.iPlots = uncertOpts.get('plot_acf', False)
      self.compute_uncert_estim = uncertOpts.get('compute_uncert_estim', False)

      # set cache flags
      self.cache_var_mu_n = uncertOpts.get('cache_var_mu_n', False)
      self.cache_acf = uncertOpts.get('cache_acf', False)
      self.cache_mean = uncertOpts.get('cache_mean', False)
      self.cache_var = uncertOpts.get('cache_var', False)

      # create caches
      self.var_mu_n_cache = []
      self.acf_cache = []
      self.mean_cache = []
      self.var_cache = []

      # create output object
      self.output_fields = {}

   def update_stats(self, m_list_in, mean_var_acf_in):
      # list of lags
      self.m_list = m_list_in
      # only store the
      self.mean_var_acf = mean_var_acf_in
      self.finalize_macf()

   def finalize_macf(self):
      block_length = self.mean_var_acf.shape[0]

      if self.var_mu is None:
         self.var_mu = np.zeros(block_length)
         self.var_mu_n = np.zeros(block_length)

      #print("Shape of block: {0}".format(self.mean_var_acf.acf.shape))
      for i in range(block_length):

         try:
            (self.var_mu[i], self.var_mu_n[i]) = self._varEstim(self.m_list, self.mean_var_acf[i, :], self.mean_var_acf.var[i], 
                                                                  nMax=100000, maxIter=self.maxIter, iPlot=self.iPlots)
         except ValueError:
            (self.var_mu[i], self.var_mu_n[i]) = (0, 0)

      # set output
      self.output_fields['SME_var'] = self.var_mu

      # cache output
      if self.cache_var_mu_n:
         self.var_mu_n_cache.append(list(self.var_mu_n))

   def finalize_macf2(self, m_list_in, mean_var_acf_in, var_in):
      block_length = mean_var_acf_in.shape[0]

      if self.var_mu is None:
         self.var_mu = np.zeros(block_length)
         self.var_mu_n = np.zeros(block_length)

      #print("Shape of block: {0}".format(self.mean_var_acf.acf.shape))
      for i in range(block_length):

         try:
            (self.var_mu[i], self.var_mu_n[i]) = self._varEstim(m_list_in, mean_var_acf_in[i, :], var_in[i], nMax=100000, maxIter=self.maxIter, iPlot=False)
         except ValueError:
            (self.var_mu[i], self.var_mu_n[i]) = (0, 0)

      # set output
      self.output_fields['SME_var'] = self.var_mu

      # cache output
      if self.cache_var_mu_n:
         self.var_mu_n_cache.append(list(self.var_mu_n))

   def _varEstim(self, acfTrainLags, acfTrainVals, var_f, maxIter=50000, nMax=100000, iPlot=False):

      """
      Estimator for \sigma^2(\mu) using MACF with training ACF computed from in-situ algorithm

      Returns
         var_mu: scalar, estimated varaince for \mu=E[f] using SME
      """

      from scipy.optimize import curve_fit

      # 1. Fit ACF to training sample ACFs 
      #popt, _ = curve_fit(self._MACF_fun, acfTrainLags, acfTrainVals, maxfev=maxIter,  bounds=(0,np.inf), jac=self._MACF_jac)  #  <--- bottle neck!!
      popt, _ = curve_fit(self._MACF_fun, acfTrainLags, acfTrainVals, maxfev=maxIter,  bounds=(0,np.inf))  #  <--- bottle neck!!

      #(2) Evaluate modeled ACF at all n lags
      n = nMax
      k_test = np.arange(n)
      acf_model = self._MACF_fun(k_test,*popt)
      if self.store_acf_model:
         self.cache_acf_model.append(acf_model)
      
      # (3) Estimate uncertainty in SME
      var_mu = (1. + 2 * np.sum(( 1 - k_test[1:] / n) * acf_model[1:])) * var_f / float(n)

      return (var_mu, var_mu * nMax)

   def _MACF_fun(self,k,a,b,c):
      """
      Function to model Autocovariance function (ACF)
      Inputs:
         k: Integer, lag in ACF
         a,b,c: model parameters
      Outputs:
         function value at (k) given (a,b,c)     
      #Note parameters in the exp(.) should not become negative => constraint optimization
      """

      ##Multiscale exponential stochastic process
      return((a*np.exp(-b*k))+(1-a)*np.exp(-c*k))   
   def _MACF_jac(self,k,a,b,c):
      jac = np.empty((len(k),3))
      jac[:,0] = np.exp(-b*k)-np.exp(-c*k)
      jac[:,1] = -a*k*np.exp(-b*k)
      jac[:,2] = (a-1)*k*np.exp(-c*k)
      return jac



