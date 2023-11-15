#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import sys
import time
sys.path.append('/u/yju/software_gcc11_cuda11.4_python3.9/lib/python3.9/site-packages/adios2/')
import adios2
sys.path.append('/u/yju/ADMIRE/NEKO/UQ/neko/examples/tgv_uq_hybrid')
from IMACF2 import IMACF2


tic_total=time.perf_counter()
global_comm = MPI.COMM_WORLD
global_rank = global_comm.Get_rank()
global_size = global_comm.Get_size()
colour=1
comm=global_comm.Split(colour,global_rank)
rank = comm.Get_rank()
size = comm.Get_size()
adios = adios2.ADIOS("config/config.xml", comm, True)
inIO = adios.DeclareIO("reader")
reader = inIO.Open("globalArray", adios2.Mode.Read, comm)
insituTime=0.0
step=0
nek_size=global_size-size
if(nek_size > size):
    init_int_total = nek_size*2+size
else:
    init_int_total = nek_size*2
if rank == 0:
    init_int_start=0
    init_int_count=2
else:
    init_int_count=int((init_int_total-2)/(size-1))
    init_int_start=2+(rank-1)*init_int_count
int_const=np.zeros(init_int_count,dtype=np.int32)
if(rank==0):
    print("start")
reader.BeginStep()
init_int = inIO.InquireVariable("INT_CONST")
init_int.SetSelection([[init_int_start],[init_int_count]])
reader.Get(init_int,int_const)
reader.EndStep()
int_const_all=np.zeros(2, dtype=np.int32)
if(rank == 0):
    int_const_all[0] = int_const[0]
    int_const_all[1] = int_const[1]
comm.Bcast(int_const_all, root=0)
lag=int_const_all[0]
if(rank==0):print(lag)
nelgv=int_const_all[1]
nelv=int(nelgv/size)
if(rank < nelgv%size):
    nelv+=1
temp=comm.allgather(nelv)
nelbv=0
for i in range(int(rank)):
    nelbv+=temp[i]
'''
if(nek_size > size):
    m_list_total = int(nek_size*lag)
    m_list_start = int(nek_size/size*rank*lag)
    m_list_count = int(nek_size/size*lag)
else:
    m_list_total = int(size*lag)
    m_list_start = int(rank*lag)
    m_list_count = int(lag)
if rank == 0:
    m_list_start=0
    m_list_count=lag
else:
    m_list_count=int((m_list_total-2)/(size-1))
    m_list_start=2+(rank-1)*m_list_count
'''
#print(str(rank)+": "+str(m_list_total)+" , "+str(m_list_start)+" , "+str(m_list_count))
uncertOpts = {'sample_freq': 1, 'max_acf_lag': 50, 'output_freq': 50, 'compute_uncert_estim': True}
imacf = IMACF2(uncertOpts)
m_list = [i * uncertOpts['sample_freq'] for i in range(uncertOpts['max_acf_lag']+1)]
while True:
    status = reader.BeginStep()
    if status == adios2.StepStatus.OK:
        #ml_in = inIO.InquireVariable("M_LIST_CONST")
        acf_in = inIO.InquireVariable("ACF")
        var_in = inIO.InquireVariable("VAR")
        #ml_in.SetSelection([[m_list_start], [m_list_count]])
        acf_in.SetSelection([[nelbv,0], [nelv,lag]])
        var_in.SetSelection([[nelbv], [nelv]])
        #m_list_in=np.zeros(m_list_count, dtype=np.float64)
        mean_var_acf=np.zeros([nelv,lag], dtype=np.float64)
        mean_var_acf_var=np.zeros(nelv, dtype=np.float64)
        #reader.Get(ml_in, m_list_in)
        reader.Get(acf_in, mean_var_acf)
        reader.Get(var_in, mean_var_acf_var)
        reader.EndStep()
        tic_in = time.perf_counter()
        '''
        if rank == 0:
            m_list = m_list_in 
        else:
            m_list = np.zeros(lag, dtype=np.float64)
        comm.Bcast(m_list)
        '''
        #m_list = m_list_in[0:lag]
        imacf.finalize_macf2(m_list, mean_var_acf, mean_var_acf_var) #output: imacf.var_mu
        toc_in = time.perf_counter()
        insituTime+=(toc_in-tic_in)
        step+=1
    elif status == adios2.StepStatus.EndOfStream:
        print("End of stream")
        reader.Close()
        break
toc_total=time.perf_counter()
print(str(rank)+": total time: "+str(toc_total-tic_total)+"s, insitu: "+str(insituTime))
MPI.Finalize()
