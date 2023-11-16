####################################################################
# Incremental Modeled Autocorrelation Function (MACF) Method based on
# 'updating_ts.py' and iMACF.py by Saleh Rezaeiravesh (KTH)
####################################################################
#----------------------------------------------------------------
# Christian Gscheidle, christian.gscheidle@scai.fraunhofer.de
#----------------------------------------------------------------

import numpy as np
import math as mt
from mpi4py import MPI
from IStats import IMeanVarACF
import adios2

#import os, sys
#sys.path.append(os.getenv("UQit_ts_path"))



class IMACF():
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
   def __init__(self, uncertOpts, comm_in, nelv_in):

      # set options
      self.uncertOpts = uncertOpts

      # init incremental batch mean
      max_acf_lag_raw = int( uncertOpts['max_acf_lag'] / uncertOpts['sample_freq'] )

      # init incremental batch mean
      self.mean_var_acf = IMeanVarACF(with_var=True, with_acf=True, max_acf_lag=max_acf_lag_raw)

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
      # initial adios2
      global_comm = MPI.COMM_WORLD
      global_rank = global_comm.Get_rank()
      global_size = global_comm.Get_size()
      colour=0
      self.comm=MPI.Comm.f2py(comm_in)
      self.rank = self.comm.Get_rank()
      self.size = self.comm.Get_size()
      self.insitu_size = global_size - self.size
      
      if (self.rank == (self.size - 1) and self.insitu_size > self.size) :
         vINT_CONST=np.zeros(2 + self.insitu_size, dtype=np.int32)
      else:
         vINT_CONST=np.zeros(2, dtype=np.int32)
      vINT_CONST[0]=max_acf_lag_raw+1
      self.nelgv=nelv_in #TODO nelgv from nek
      vINT_CONST[1]=self.nelgv
      init_total = self.size * 2
      init_start = self.rank * 2
      init_count = 2
      if self.insitu_size > self.size:
         init_total += self.insitu_size
         if self.rank == (self.size - 1):
            init_count += self.insitu_size
      self.nelv = int(self.nelgv/self.size)
      if(self.rank < self.nelgv % self.size):
         self.nelv += 1
      temp=self.comm.allgather(self.nelv)
      self.nelbv=0
      for i in range(int(self.rank)):
         self.nelbv+=temp[i]
      if self.rank==0:
          print("init")
      self.adios = adios2.ADIOS("config/config.xml", self.comm, True)

      self.inIO = self.adios.DeclareIO("writer")

      self.writer = self.inIO.Open("globalArray", adios2.Mode.Write, self.comm)
      init_int = self.inIO.DefineVariable("INT_CONST", vINT_CONST, [init_total], [init_start], [init_count],adios2.ConstantDims)
      self.writer.BeginStep()
      self.writer.Put(init_int, vINT_CONST)
      self.writer.EndStep()
      '''
      ml_count = max_acf_lag_raw+1
      ml_total = self.size * ml_count
      ml_start = self.rank * ml_count
      if self.insitu_size > self.size:
         ml_total *= (self.insitu_size/self.size)
         ml_start *= (self.insitu_size/self.size)
         ml_count *= (self.insitu_size/self.size)
      self.ml_send = np.zeros(ml_count, dtype=np.float64)
      '''
      self.acf_send = np.zeros([self.nelv,  max_acf_lag_raw+1], dtype=np.float64)
      self.var_send = np.zeros(self.nelv, dtype=np.float64)
      #self.ml = self.inIO.DefineVariable("M_LIST_CONST", self.ml_send, [ml_total], [ml_start], [ml_count],  adios2.ConstantDims)
      self.acf = self.inIO.DefineVariable("ACF", self.acf_send, [self.nelgv, max_acf_lag_raw+1], [self.nelbv, 0], [self.nelv,  max_acf_lag_raw+1],  adios2.ConstantDims)
      self.var = self.inIO.DefineVariable("VAR", self.var_send, [self.nelgv], [self.nelbv], [self.nelv], adios2.ConstantDims)


   def __del__(self):
      self.writer.Close()



   def update_stats(self, t, F):

      # update mean, var and acf for current batch 
      if np.mod(self.sample_count + 1, self.uncertOpts['sample_freq']) == 0:

         self.mean_var_acf.partial_fit(F)

      # update uncertainty estimate and set output
      if np.mod(self.sample_count + 1, self.uncertOpts['output_freq']) == 0:

         self.finalize_macf()

      else:
         
         self.output_fields = {}

      self.sample_count += 1
      return self.var_mu

   def finalize_macf(self):

      try:
         #finialize statistics
         self.mean_var_acf.finalize_stats() 
      except AttributeError:

        if self.mean_var_acf.m_list[0] != 0:
            self.mean_var_acf.m_list.insert(0, 0)
      
      # create list of m for original signal
      self.m_list = [i * self.uncertOpts['sample_freq'] for i in self.mean_var_acf.m_list]
      #print(len(self.m_list))
      # compute uncertainty in SME
      if self.compute_uncert_estim or self.cache_var_mu_n:

         block_length = self.mean_var_acf.acf.shape[0]

         if self.var_mu is None:
            self.var_mu = np.zeros(block_length)
            self.var_mu_n = np.zeros(block_length)

         if self.reduce_lags > 1:
            self.m_list = self.m_list[:self.reduce_lags]
            self.mean_var_acf.acf = self.mean_var_acf.acf[:, :self.reduce_lags]
         
         #self.ml_send=np.array(self.m_list,dtype=np.float64)
         #for i in range(1,int(self.insitu_size/self.size)):
         #    np.append(self.ml_send, self.m_list)
         self.acf_send=np.array(self.mean_var_acf.acf,dtype=np.float64)
         self.var_send=np.array(self.mean_var_acf.var,dtype=np.float64)
         self.writer.BeginStep()
         #self.writer.Put(self.ml, self.ml_send)
         self.writer.Put(self.acf, self.acf_send)
         self.writer.Put(self.var, self.var_send)
         self.writer.EndStep()

         #print("Shape of block: {0}".format(self.mean_var_acf.acf.shape))
         '''
         for i in range(block_length):

            if self.iPlots:
               print("ACF at sample: {0} for point: {1}".format(self.sample_count, i))

            try:
               (self.var_mu[i], self.var_mu_n[i]) = self._varEstim(self.m_list, self.mean_var_acf.acf[i, :], self.mean_var_acf.var[i], 
                                                                     nMax=100000, maxIter=self.maxIter, iPlot=self.iPlots)
            except ValueError:
               (self.var_mu[i], self.var_mu_n[i]) = (0, 0)
         '''
         # set output
         self.output_fields['SME_var'] = self.var_mu

         # cache output
         if self.cache_var_mu_n:
            self.var_mu_n_cache.append(list(self.var_mu_n))

      if self.cache_mean:
         self.mean_cache.append(list(self.mean_var_acf.mean))

      if self.cache_var:
         self.var_cache.append(list(self.mean_var_acf.var))

      if self.cache_acf:
         self.acf_cache.append(self.mean_var_acf.acf)

      # set output
      self.output_fields['ACF'] = self.mean_var_acf.acf
      self.output_fields['SME'] = self.mean_var_acf.mean
      self.output_fields['VAR'] = self.mean_var_acf.var

   def _varEstim(self, acfTrainLags, acfTrainVals, var_f, maxIter=50000, nMax=100000, iPlot=False):

      """
      Estimator for \sigma^2(\mu) using MACF with training ACF computed from in-situ algorithm

      Returns
         var_mu: scalar, estimated varaince for \mu=E[f] using SME
      """

      from scipy.optimize import curve_fit

      # 1. Fit ACF to training sample ACFs 
      popt, _ = curve_fit(self._MACF_fun, acfTrainLags, acfTrainVals, maxfev=maxIter, bounds=(0,np.inf))  #  <--- bottle neck!!

      #(2) Evaluate modeled ACF at all n lags
      n = nMax
      k_test = np.arange(n)
      acf_model = self._MACF_fun(k_test,*popt)
      if self.store_acf_model:
         self.cache_acf_model.append(acf_model)
      
      # (3) Estimate uncertainty in SME
      var_mu = (1. + 2 * np.sum(( 1 - k_test[1:] / n) * acf_model[1:])) * var_f / float(n)

      if iPlot:

         import matplotlib.pyplot as plt

         print("var_mu: {0}, var_mu*nMax: {1}".format(var_mu, var_mu * nMax))

         plt.figure(figsize=(10,4))
         plt.plot(k_test, acf_model, '--r',lw=2, label='Curve-fit, LM')
         plt.semilogx(acfTrainLags, acfTrainVals, 'sk', mfc='y', ms=5, label='Training samples')
         #plt.plot(k_test,MACF_fun(k_test,q[0],q[1],q[2]),':k',label='curve-fit, Newton')
         plt.legend(loc='best',fontsize=14)
         plt.xlabel('Lag',fontsize=14)
         plt.ylabel('ACF',fontsize=14)
         plt.grid(alpha=0.3)
         plt.show()


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
 


