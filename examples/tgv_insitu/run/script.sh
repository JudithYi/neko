#!/bin/bash -l
#SBATCH --job-name=neko_async
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=3
#SBATCH --constraint="gpu"
#SBATCH --switch=1
# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:4

#SBATCH --nvmps

#SBATCH --mail-type=NONE
#SBATCH --mail-user=email@adress
#SBATCH --mem=500000
#SBATCH --time=00:30:00
#SBATCH -o async_02.12_12.io20.log.%j

module purge
module load gcc/11 anaconda/3/2021.11 cuda/11.4 openmpi/4.1 openmpi_gpu/4.1 mkl_parts-mpi/1 mpi4py/3.0.3
module list

rm -rf perf
mkdir perf
rm -rf data
mkdir data
rm -rf fig
mkdir fig
rm -rf field0.*
rm -rf bdry0.*
rm -rf globalArray*

export PARAVIEW=/path/to/paraview
export PATH=$PARAVIEW/bin:$PATH
export LD_LIBRARY_PATH=$PARAVIEW/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$PARAVIEW/lib/python3.9/site-packages:$PYTHONPATH
export PYTHONPATH=$PARAVIEW/lib/python3.9/site-packages/paraview/:$PYTHONPATH

export PYTHONPATH=$PATH:$LD_LIBRARY_PATH:`pwd`'/'
echo $PYTHONPATH

#srun ./run_gpu.sh ./neko tgv.case

cat mpmd.conf
echo
srun --multi-prog mpmd.conf
rm -rf field0.*
rm -rf bdry0.*
rm -rf globalArray*