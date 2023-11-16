################################################################
# Incremental computation of mean and variance and acf
#################################################################
#----------------------------------------------------------------
# Christian Gscheidle, christian.gscheidle@scai.fraunhofer.de
#----------------------------------------------------------------

import numpy as np

class IMeanVarACF():

    def __init__(self, with_var=False, with_acf=False, max_acf_lag=None):

        self.with_var = with_var
        self.with_acf = with_acf
        self.max_acf_lag = max_acf_lag

        if max_acf_lag is not None and with_acf:
            self.m_list = [(i + 1) for i in range(self.max_acf_lag)]
        else:
            self.m_list = None

    def fit(self, X):

        self._reset()

        self.mean = np.mean(X, axis=0)

        if self.with_var:
            self.var = np.var(X, axis=0)
        else:
            self.var = None

        return self

    def partial_fit(self, X, print_term_sizes=False):

        # init arrays if necessary
        if not hasattr(self, 'mean'):

            try:
                shape0 = X.shape[0]
            except AttributeError:
                shape0 = 1

            self.mean = np.zeros(shape0, dtype=np.float64)
            if self.with_var:
                self.sqr_sum = np.zeros(shape0, dtype=np.float64)
                self.var = np.zeros(shape0, dtype=np.float64)
            else:
                self.sqr_sum = None
                self.var = None

            if self.with_acf:
                self.acovf = np.zeros((shape0, self.max_acf_lag), dtype=np.float64)
                self.sum_m = np.zeros((shape0, self.max_acf_lag), dtype=np.float64)
                self.buffer = np.zeros((shape0, self.max_acf_lag), dtype=np.float64)
            else:
                self.acovf = None
                self.sum_m = None
                self.buffer = None

            self.sample_count = 0
            self.continuous_sample_count = 0

        # update mean and var
        # self.mean, self.sqr_sum, self.sample_count, self.var = self._incremental_mean_and_var(X, self.mean, self.sqr_sum, self.sample_count)
        self.sample_count, self.mean, self.sqr_sum, self.acovf, self.sum_m, self.buffer \
                                            = self._incremental_mean_var_acovf(X,   self.mean, self.sqr_sum, self.sample_count,
                                                                                    self.acovf, self.sum_m, self.buffer, self.max_acf_lag, \
                                                                                        print_term_sizes=print_term_sizes)

        return self

    def finalize_stats(self):

        if self.with_acf or self.with_var:
            self.var = self.sqr_sum / self.sample_count

        if self.with_acf:
            if self.m_list[0] != 0:
                self.m_list.insert(0, 0)

            self.acf = np.insert(self.acovf, 0, self.sqr_sum, axis=1)
            self.acf /= self.sample_count
            self.acf = self.acf / self.acf[:, 0][:, None]
        else:
            pass

    def _reset(self):

        try: 
            del self.mean
            del self.var
            del self.sqr_sum
            del self.acf
            del self.acovf
            del self.sum_m
            del self.buffer
        except AttributeError:
            pass

    def _incremental_mean_var_acovf(self, X, last_mean, last_sqr_sum, last_sample_count, acovf, sum_m, buffer, max_lag, print_term_sizes=False):

        """ 1) Calculate mean update and Welford's variance update.

            See: "Algorithms for computing the sample variance: analysis and recommendations", 
                by Chan, Golub, and LeVeque

            2) Calculates autocovariance using updating (in-situ) algorithms with caching 

        """

        # update parameters
        new_sample_count = last_sample_count + 1
        new_diff = X - last_mean
        new_diff_scaled = new_diff / new_sample_count

        # compute updated mean 
        new_mean = last_mean + new_diff_scaled

        # compute updated variance
        if last_sqr_sum is None:
            new_sqr_sum = None
        else:
            tmp = new_diff * new_diff_scaled
            new_sqr_sum = last_sqr_sum + last_sample_count * tmp

        # compute updated ACov
        if acovf is None:
            acovf = None
            sum_m = None
            buffer = None

        else:
 
            # run through all lags 
            for k in range(0, max_lag): 

                # current lag 
                m = k + 1    

                # j-m
                j_m = last_sample_count - m  

                if last_sample_count >= m:   

                    # X(j-m)
                    X_j_m = buffer[:, np.mod(j_m, max_lag)]

                    # update acovf
                    acovf[:, k] += X * X_j_m - new_diff_scaled * sum_m[:, k] - (last_mean + new_diff_scaled) * (X + X_j_m) \
                            + (j_m + 1) * (last_mean + new_diff_scaled)**2 - j_m * last_mean**2 
                            
                    sum_m[:, k] += X + X_j_m 
                
            # overwrite oldest buffer column with new value
            buffer[:, np.mod(last_sample_count, max_lag)] = X

        return new_sample_count, new_mean, new_sqr_sum, acovf, sum_m, buffer

    def _incremental_mean_and_var(self, X, last_mean, last_sqr_sum, last_sample_count):

        """ Calculate mean update and Welford's variance update.

            See: "Algorithms for computing the sample variance: analysis and recommendations", 
                by Chan, Golub, and LeVeque

        """

        # update parameters
        new_sample_count = last_sample_count + 1
        new_diff = X - last_mean
        new_diff_scaled = new_diff / new_sample_count

        # compute updated mean 
        new_mean = last_mean + new_diff / new_sample_count

        if last_sqr_sum is None:
            new_sqr_sum = None
            new_variance = None
        else:
            # compute updated variance
            tmp = new_diff * new_diff_scaled
            new_sqr_sum = last_sqr_sum + (new_sample_count - 1) * tmp
            new_variance = new_sqr_sum / new_sample_count

        return new_mean, new_sqr_sum, new_sample_count, new_variance


