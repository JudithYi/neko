#!/usr/bin/env python3
from mpi4py import MPI
import numpy 
import scipy.optimize
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def set_comm(comm_in):
    global comm
    global rank
    global size
    comm = MPI.Comm.f2py(comm_in)
    rank = comm.Get_rank()
    size = comm.Get_size()

def pyArray (a):
    print ("1st Contents of a :")
    print (a[0])
    c = 0
    return c

def distributed_svd(Xi,n,m):
    #Take the partiotioned data Xi in each rank and do the SVD.
    #The outputs are:
        # A partitioned mode matrix U
        # The eigen values D
        # The right orthogonal matrix trasposed Vt


    #Perfrom Svd in all ranks
    tic_in = time.perf_counter()
    Xi=Xi.reshape((Xi.size, 1))
    Ui,Di,Vti=numpy.linalg.svd(Xi, full_matrices=False)
    toc_in = time.perf_counter()
    Yi=numpy.diag(Di)@Vti
    #print(f"Time for SVD of Xi in rank {rank}: {toc_in - tic_in:0.4f} seconds")

    #Gather Yi into Y in rank 0
    #prepare the buffer for recieving
    Y = None
    if rank == 0:
        #Generate the buffer to gather in rank 0
        Y = numpy.empty((m*size,m))
    comm.Gather(Yi, Y, root=0)

    if rank == 0:
        #If tank is zero, calculate the svd of the combined eigen matrix
        #Perform the svd of the combined eigen matrix
        tic_in = time.perf_counter()
        Uy,Dy,Vty=numpy.linalg.svd(Y, full_matrices=False)
        toc_in = time.perf_counter()
        #print(f"Time for SVD of Y in rank {rank}: {toc_in - tic_in:0.4f} seconds")
    else:
        #If the rank is not zero, simply create a buffer to recieve the Uy Dy and Vty
        Uy  = numpy.empty((m*size,m))
        Dy  = numpy.empty((m))
        Vty = numpy.empty((m,m))
    comm.Bcast(Uy, root=0)
    comm.Bcast(Dy, root=0)
    comm.Bcast(Vty, root=0)
    #Now matrix multiply each Ui by the corresponding entries in Uy
    U_local=Ui@Uy[rank*m:(rank+1)*m,:]

    return U_local, Dy, Vty

def dist_svd_update(U_1t,D_1t,Vt_1t,Xi,n,j): 

    if j==0:
        #Perform the distributed SVD and don't accumulate
        U_1t,D_1t,Vt_1t=distributed_svd(Xi,n,j+1)
    else:
        #Find the svd of the new snapshot
        U_tp1,D_tp1,Vt_tp1=distributed_svd(Xi,n,1)
        #2 contruct matrices to Do the updating
        V_tilde=scipy.linalg.block_diag(Vt_1t.T,Vt_tp1.T)
        W=numpy.append(U_1t@np.diag(D_1t),U_tp1@np.diag(D_tp1),axis=1)
        Uw,Dw,Vtw=distributed_svd(W,n,j+1)
        #3 Update
        U_1t=Uw
        D_1t=Dw
        Vt_1t=(V_tilde@Vtw.T).T
  
    return U_1t,D_1t,Vt_1t 

def gathermodes(U_1t, n, m):
    U = None #prepare the buffer for recieving
    if rank == 0:
        #Generate the buffer to gather in rank 0
        U = numpy.empty((n,m))
    comm.Gather(U_1t, U, root=0)
    return U

def gathermodesAndmass(U_1t,BM1, n, m):
    U = None #prepare the buffer for recieving
    bm1sqrt = None #prepare the buffer for recieving
    if rank == 0:
        #Generate the buffer to gather in rank 0
        U = numpy.empty((n,m))
        bm1sqrt = numpy.empty((n,1))
    comm.Gather(U_1t, U, root=0)
    comm.Gather(BM1, bm1sqrt, root=0)
    return U, bm1sqrt

