#ifndef __NEKO_COMM_H
#define __NEKO_COMM_H

#include <mpi.h>

MPI_Comm NEKO_COMM_C;
void init_c_comm_(int comm_f);

#endif
