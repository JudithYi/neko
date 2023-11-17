#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import sys
import time
sys.path.append('/u/yju/software_gcc11_cuda11.4_python3.9/lib/python3.9/site-packages/adios2/')
import adios2
sys.path.append('/u/yju/ADMIRE/NEKO/UQ/neko/examples/tgv_uq_async')
from IMACF import IMACF


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
#vINT_CONST=np.zeros(2000, dtype=np.long)
#vP=np.zeros(200000, dtype=np.float64)
#vVX=np.zeros(200000, dtype=np.float64)
#vVY=np.zeros(200000, dtype=np.float64)
#vVZ=np.zeros(200000, dtype=np.float64)
#vBM1=np.zeros(200000, dtype=np.float64)
#vNELV=np.zeros(200, dtype=np.float64)
#vNELT=np.zeros(200, dtype=np.float64)
nek_size=global_size-size
if(nek_size>=size):
    if(nek_size % size != 0):
        MPI.Finalize()
    else:
        if(rank==0):
            print(nek_size)
            print(size)
            print(int(nek_size/size))
        vNELV=np.zeros(int(nek_size/size), dtype=np.int32)
        vNELT=np.zeros(int(nek_size/size), dtype=np.int32)
reader.BeginStep()
init_int_vec1 = inIO.InquireVariable("INT_CONST")
int_vec1_size = init_int_vec1.Shape()[0]
if(int_vec1_size != 8 * nek_size):
    MPI.Finalize()
if(nek_size>=size):
    int_vec1_count = int(nek_size * 8 / size) 
    vINT_CONST=np.zeros(int_vec1_count, dtype=np.int32)
    int_vec1_start = int(rank * int_vec1_count)
    init_int_vec1.SetSelection([[int_vec1_start], [int_vec1_count]])
    reader.Get(init_int_vec1,vINT_CONST)
    reader.EndStep()
else:
    int_vec1_count = 8 * nek_size
    vINT_CONST=np.zeros(int_vec1_count, dtype=np.int32)
    if(rank!=0):
        int_vec1_count = 0
    int_vec1_start = 0
    init_int_vec1.SetSelection([[int_vec1_start], [int_vec1_count]])
    reader.Get(init_int_vec1,vINT_CONST)
    reader.EndStep()
    comm.Bcast(vINT_CONST,root=0)
nelv=0
nelt=0
for i in range(int(nek_size/size)):
    vNELV[i]=int(vINT_CONST[i*8+6])
    nelv+=int(vINT_CONST[i*8+6]);
    vNELT[i]=int(vINT_CONST[i*8+7]);
    nelt+=int(vINT_CONST[i*8+7]);
nelgv=0
nelgt=0
temp=np.zeros(size)
if(nek_size>=size):
    nelgv=int(comm.allreduce(nelv))
    nelgt=int(comm.allreduce(nelt))
else:
    for i in range(nek_size):
        nelgv+=vINT_CONST[i*8+6]
        nelgt+=vINT_CONST[i*8+7]
    nelv = int(nelgv / size)
    if(rank < nelgv % size):
        nelv+=1
    nelt = int(nelgt / size)
    if(rank < nelgt % size):
        nelt+=1

if(rank==0):
    print("python setting up")

temp=comm.allgather(nelv)
nelbv=0
for i in range(rank):
    nelbv+=temp[i]

temp=comm.allgather(nelt)
nelbt=0
for i in range(rank):
    nelbt+=temp[i]

lx1=int(vINT_CONST[0])
ly1=int(vINT_CONST[1])
lz1=int(vINT_CONST[2])
lx2=int(vINT_CONST[3])
ly2=int(vINT_CONST[4])
lz2=int(vINT_CONST[5])

#print(nelv)

uncertOpts = {'sample_freq': 1, 'max_acf_lag': 50, 'output_freq': 50, 'compute_uncert_estim': True}
imacf = IMACF(uncertOpts)

reader.BeginStep()

count = lx1*ly1*lz1*nelv
total = lx1*ly1*lz1*nelgv
start = lx1*ly1*lz1*nelbv

p = inIO.InquireVariable("P")
if(p.Shape()[0]!=total):
    print("p!")
    
vx = inIO.InquireVariable("VX")
vy = inIO.InquireVariable("VY")
vz = inIO.InquireVariable("VZ")

if(vx.Shape()[0]!=total):
    print("vx!")

tmp = inIO.InquireVariable("T")

if(tmp.Shape()[0]!=total):
    print("bm1!")

vP=np.zeros(count, dtype=np.float64)
vVX=np.zeros(count, dtype=np.float64)
vVY=np.zeros(count, dtype=np.float64)
vVZ=np.zeros(count, dtype=np.float64)
vT=np.zeros(count, dtype=np.float64)


p.SetSelection([[start], [count]])
vx.SetSelection([[start], [count]])
vy.SetSelection([[start], [count]])
vz.SetSelection([[start], [count]])
tmp.SetSelection([[start], [count]])

reader.Get(p, vP)
reader.Get(vx, vVX)
reader.Get(vy, vVY)
reader.Get(vz, vVZ)
reader.Get(tmp, vT)
reader.EndStep()


tic_in = time.perf_counter()
v_in = np.zeros(nelt, dtype=np.float64)
for i in range(nelt):
    for j in range(lx1*ly1*lz1):
        v_in[i] += vVX[i*lx1*ly1*lz1+j]/(lx1*ly1*lz1)
#print(v_in[0])
imacf.update_stats(step,v_in)
toc_in = time.perf_counter()
insituTime+=(toc_in-tic_in)
step+=1

while True:
    status = reader.BeginStep()
    if status == adios2.StepStatus.OK:
        p = inIO.InquireVariable("P")
        vx = inIO.InquireVariable("VX")
        vy = inIO.InquireVariable("VY")
        vz = inIO.InquireVariable("VZ")
        tmp = inIO.InquireVariable("T")

        p.SetSelection([[start], [count]])
        vx.SetSelection([[start], [count]])
        vy.SetSelection([[start], [count]])
        vz.SetSelection([[start], [count]])
        tmp.SetSelection([[start], [count]])
        reader.Get(p, vP)
        reader.Get(vx, vVX)
        reader.Get(vy, vVY)
        reader.Get(vz, vVZ)
        reader.Get(tmp, vT)
        reader.EndStep()
        tic_in = time.perf_counter()
        v_in = np.zeros(nelt)
        for i in range(nelt):
            for j in range(lx1*ly1*lz1):
                v_in[i] += vVX[i*lx1*ly1*lz1+j]/(lx1*ly1*lz1)
        imacf.update_stats(step,v_in)
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
