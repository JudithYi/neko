#include "comm/comm_c.h"
//#include <mpi.h>
//#include <stdio.h>
//MPI_Comm NEKO_COMM_C;
void init_c_comm_(int comm_f){
  //printf("%d", comm_f);
  NEKO_COMM_C=MPI_Comm_f2c(comm_f); 
}
