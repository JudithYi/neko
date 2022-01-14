#include "math_kernel.h"
#include <device/device_config.h>

extern "C" {

  /** Fortran wrapper for copy
   * Copy a vector \f$ a = b \f$
   */
  void cuda_copy(void *a, void *b, int *n) {
    cudaMemcpyAsync(a, b, (*n) * sizeof(real), cudaMemcpyDeviceToDevice);
  }

  /** Fortran wrapper for rzero
   * Zero a real vector
   */
  void cuda_rzero(void *a, int *n) {
    cudaMemsetAsync(a, 0, (*n) * sizeof(real));
  }

  /** Fortran wrapper for cmult
   * Multiplication by constant c \f$ a = c \cdot a \f$
   */
  void cuda_cmult(void *a, real *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    cmult_kernel<real><<<nblcks, nthrds>>>((real *) a,
					   *c, *n);

  }

  /** Fortran wrapper for cmult2
   * Multiplication by constant c \f$ a = c \cdot b \f$
   */
  void cuda_cmult2(void *a, void *b, real *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    cmult2_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					   *c, *n);

  }
  /** Fortran wrapper for cadd
   * Add a scalar to vector \f$ a = \sum a_i + s \f$
   */
  void cuda_cadd(void *a, real *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    cadd_kernel<real><<<nblcks, nthrds>>>((real *) a,
					  *c, *n);

  }

  
  /** Fortran wrapper for cfill
   * Set all elements to a constant c \f$ a = c \f$
   */
  void cuda_cfill(void *a, real *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    cfill_kernel<real><<<nblcks, nthrds>>>((real *) a,
					   *c, *n);

  }

  /**
   * Fortran wrapper for add2
   * Vector addition \f$ a = a + b \f$
   */
  void cuda_add2(void *a, void *b, int *n) {
    
    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    add2_kernel<real><<<nblcks, nthrds>>>((real *) a,
					  (real *) b, *n);
    
  }
  
  /**
   * Fortran wrapper for add2s1
   * Vector addition with scalar multiplication \f$ a = c_1 a + b \f$
   * (multiplication on first argument) 
   */
  void cuda_add2s1(void *a, void *b, real *c1, int *n) {
    
    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    add2s1_kernel<real><<<nblcks, nthrds>>>((real *) a,
					    (real *) b,
					    *c1, *n);
    
  }

  /**
   * Fortran wrapper for add2s2
   * Vector addition with scalar multiplication \f$ a = a + c_1 b \f$
   * (multiplication on second argument) 
   */
  void cuda_add2s2(void *a, void *b, real *c1, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    add2s2_kernel<real><<<nblcks, nthrds>>>((real *) a,
					    (real *) b,
					    *c1, *n);

  }

  void cuda_add2s2_many(void *x, void **p, void *alpha, int *j, int *n) {
	
    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);
    
    add2s2_many_kernel<real><<<nblcks, nthrds>>>((real *) x, (const real **) p, (real *) alpha, *j, *n);

  }
  /**
   * Fortran wrapper for addsqr2s2
   * Vector addition with scalar multiplication \f$ a = a + c_1 (b * b) \f$
   * (multiplication on second argument) 
   */
  void cuda_addsqr2s2(void *a, void *b, real *c1, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    addsqr2s2_kernel<real><<<nblcks, nthrds>>>((real *) a,
					       (real *) b,
					       *c1, *n);

  }

  /**
   * Fortran wrapper for add3s2
   * Vector addition with scalar multiplication \f$ a = c_1 b + c_2 c \f$
   * (multiplication on second argument) 
   */
  void cuda_add3s2(void *a, void *b, void *c, real *c1, real *c2, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    add3s2_kernel<real><<<nblcks, nthrds>>>((real *) a,
					    (real *) b,
					    (real *) c,
					    *c1, *c2, *n);

  }

 
  /**
   * Fortran wrapper for invcol1
   * Invert a vector \f$ a = 1 / a \f$
   */
  void cuda_invcol1(void *a, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    invcol1_kernel<real><<<nblcks, nthrds>>>((real *) a,
					     *n);
  }
  /**
   * Fortran wrapper for invcol2
   * Vector division \f$ a = a / b \f$
   */
  void cuda_invcol2(void *a, void *b, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    invcol2_kernel<real><<<nblcks, nthrds>>>((real *) a,
					       (real *) b, *n);
  }
  
  /**
   * Fortran wrapper for col2
   * Vector multiplication with 2 vectors \f$ a = a \cdot b \f$
   */
  void cuda_col2(void *a, void *b, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    col2_kernel<real><<<nblcks, nthrds>>>((real *) a, 
					    (real *) b, *n);
  }
  
  /**
   * Fortran wrapper for col3
   * Vector multiplication with 3 vectors \f$ a = b \cdot c \f$
   */
  void cuda_col3(void *a, void *b, void *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    col3_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					    (real *) c, *n);
  }

  /**
   * Fortran wrapper for subcol3
   * Vector multiplication with 3 vectors \f$ a = a - b \cdot c \f$
   */
  void cuda_subcol3(void *a, void *b, void *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    subcol3_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					     (real *) c, *n);
  }
  

  /**
   * Fortran wrapper for sub2
   * Vector subtraction \f$ a = a - b \f$
   */
  void cuda_sub2(void *a, void *b, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    sub2_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b, *n);
  }

  /**
   * Fortran wrapper for sub3
   * Vector subtraction \f$ a = b - c \f$
   */
  void cuda_sub3(void *a, void *b, void *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    sub3_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b, 
					    (real *) c, *n);
  }

  /**
   * Fortran wrapper for addcol3
   * \f$ a = a + b * c \f$
   */
  void cuda_addcol3(void *a, void *b, void *c, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    addcol3_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					       (real *) c, *n);
  }

  /**
   * Fortran wrapper for addcol4
   * \f$ a = a + b * c * d\f$
   */
  void cuda_addcol4(void *a, void *b, void *c, void *d, int *n) {

    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);

    addcol4_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					     (real *) c, (real *) d, *n);
  }

  /**
   * Fortran wrapper glsc3
   * Weighted inner product \f$ a^T b c \f$
   */
  real cuda_glsc3(void *a, void *b, void *c, int *n) {
	
    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);
    const int nb = ((*n) + 1024 - 1)/ 1024;
    
    real * buf = (real *) malloc(nb * sizeof(real));
    real * buf_d;

    cudaMalloc(&buf_d, nb*sizeof(real));
     
    glsc3_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					     (real *) c, buf_d, *n);

    cudaMemcpy(buf, buf_d, nb * sizeof(real), cudaMemcpyDeviceToHost);

    real res = 0.0;
    for (int i = 0; i < nb; i++) {
      res += buf[i];
    }

    free(buf);
    cudaFree(buf_d);

    return res;
  }
int red_s = 0;
real * bufred;
real * bufred_d;
/**
   * Fortran wrapper cg_part_2
   * Weighted inner product \f$ a^T b c \f$
   */
  void cuda_glsc3_many(real *h, void * w, void *v,void *mult, int *j, int *n){ 
    int pow2 = 1;
    while(pow2 < (*j)){
      pow2 = 2*pow2;
    }
    const int nt = 1024/pow2;	
    const dim3 nthrds(nt, pow2, 1);
    const dim3 nblcks(((*n)+nt - 1)/nt, 1, 1);
    const int nb = ((*n) + nt - 1)/nt;
    if((*j)>red_s){
      red_s = *j;
      free(bufred);
      cudaFree(bufred_d);
      bufred = (real *) malloc((*j)*nb * sizeof(real));
      cudaMalloc(&bufred_d, (*j)*nb*sizeof(real));
    }
    glsc3_many_kernel<real><<<nblcks, nthrds>>>((const real *) w, (const real **) v, (const real *)mult, bufred_d, *j, *n);
    cudaMemcpy(bufred, bufred_d, (*j)*nb * sizeof(real), cudaMemcpyDeviceToHost);
    for (int k = 0; k < (*j); k++) {
      h[k] = 0.0;
    }
    
    for (int i = 0; i < nb; i++) {
      for (int k = 0; k < (*j); k++) {
        h[k] += bufred[i*(*j)+k];
      }
    }
  }


  /**
   * Fortran wrapper glsc2
   * Weighted inner product \f$ a^T b c \f$
   */
  real cuda_glsc2(void *a, void *b, int *n) {
	
    const dim3 nthrds(1024, 1, 1);
    const dim3 nblcks(((*n)+1024 - 1)/ 1024, 1, 1);
    const int nb = ((*n) + 1024 - 1)/ 1024;
    
    real * buf = (real *) malloc(nb * sizeof(real));
    real * buf_d;

    cudaMalloc(&buf_d, nb*sizeof(real));
     
    glsc2_kernel<real><<<nblcks, nthrds>>>((real *) a, (real *) b,
					      buf_d, *n);

    cudaMemcpy(buf, buf_d, nb * sizeof(real), cudaMemcpyDeviceToHost);

    real res = 0.0;
    for (int i = 0; i < nb; i++) {
      res += buf[i];
    }

    free(buf);
    cudaFree(buf_d);

    return res;
  }

}
