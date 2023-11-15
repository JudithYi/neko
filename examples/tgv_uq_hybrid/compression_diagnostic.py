import matplotlib as mpl
import matplotlib.pyplot as plt
import pymech as pm
from pymech.neksuite import readnek
import numpy as np
import sys


#=================================================================

#Define some functions

def get_metrics(rctr_field,orgl_field,qoi,nel):

    #Declare variables for check up.
    elem=np.zeros(nel)
    maxabserr_v=np.zeros(nel)
    maxrelerr_v=np.zeros(nel)
    rmse_v=np.zeros(nel)
    nrmse_v=np.zeros(nel)
    rho_v=np.zeros(nel)
    qoi=qoi

    for e in range(0,nel):
        #Assign the element number
        elem[e]=e+1

        #Read the data in each element (ed=element data)
        rctr_ed=rctr_field.elem[e]  #for the reconstructed field
        orgl_ed=orgl_field.elem[e]  #for the original field

        #Rearange the velocities in an easier vector to read.
        rctr_v=rctr_ed.vel[qoi,:,:,:].reshape((nxyz,1))
        orgl_v=orgl_ed.vel[qoi,:,:,:].reshape((nxyz,1))
        R_v=np.max(orgl_v)-np.min(orgl_v)

        #1. Calculate the first metric (Absolute error)
        abserr_v=orgl_v-rctr_v
        #get the maximun to print only one value
        maxabserr_v[e]=np.max(np.abs(abserr_v))

        #2. Calculate the second metric (Relative error)
        relerr_v=abserr_v/R_v
        #get the maximun to print only one value
        maxrelerr_v[e]=np.max(np.abs(relerr_v))

        #3. Calculate the third metric (RMSE)
        temp=np.matmul(abserr_v.T,abserr_v)/nxyz
        rmse_v[e]=np.sqrt(temp[0,0])
        # Also calculate the normalized one
        nrmse_v[e]=rmse_v[e]/R_v

        #4. Calculate the pearson correlation coefficient.
        rho=np.corrcoef(rctr_v[:,0],orgl_v[:,0])
        rho_v[e]=rho[0,1]

    return elem,maxabserr_v,maxrelerr_v,rmse_v,nrmse_v,rho_v

def plot_metrics(elem,maxabserr_v,maxrelerr_v,rmse_v,nrmse_v,rho_v,tag):
    mpl.rcParams["text.usetex"]= 'true'
    mpl.rcParams["figure.figsize"]= [10,7]
    tag=tag

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(r'Metrics for '+tag)
    axs[0, 0].plot(elem, maxabserr_v)
    axs[0,0].set_ylabel(r'$e=X-\tilde{X}$')
    axs[0, 1].plot(elem, rmse_v,)
    axs[0,1].set_ylabel(r'$rmse$')
    axs[1, 0].plot(elem, nrmse_v, )
    axs[1,0].set_ylabel(r'$nrmse$')
    axs[1, 1].plot(elem, rho_v, )
    axs[1,1].set_ylabel(r'$\rho$')
    plt.show()
    
    return

#====================================================================


#Define the names of the files to compare
rctr_filename='dssturbPipe0.f00003'
orgl_filename='fllturbPipe0.f00003'

#Read the whole files information
rctr_field=readnek(rctr_filename)
orgl_field=readnek(orgl_filename)

#Get some general information from the data
nel=orgl_field.nel
lxr=orgl_field.lr1
nx =lxr[1]
nxyz=np.prod(lxr) 

#get the metrics
#for vx
[elemx,maxabserr_vx,maxrelerr_vx,rmse_vx,nrmse_vx,rho_vx]=get_metrics(rctr_field,orgl_field,0,nel)
#for vy
[elemy,maxabserr_vy,maxrelerr_vy,rmse_vy,nrmse_vy,rho_vy]=get_metrics(rctr_field,orgl_field,1,nel)
#for vz
[elemz,maxabserr_vz,maxrelerr_vz,rmse_vz,nrmse_vz,rho_vz]=get_metrics(rctr_field,orgl_field,2,nel)

#Plot the metrics
#for vx
plot_metrics(elemx,maxabserr_vx,maxrelerr_vx,rmse_vx,nrmse_vx,rho_vx,'$v_x$')
#for vy
plot_metrics(elemy,maxabserr_vy,maxrelerr_vy,rmse_vy,nrmse_vy,rho_vy,'$v_y$')
#for vz
plot_metrics(elemz,maxabserr_vz,maxrelerr_vz,rmse_vz,nrmse_vz,rho_vz,'$v_z$')

