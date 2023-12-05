#!/bin/bash -l
# Standard output and error:
##SBATCH -o ./neko_hetro.out.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J life_science
#
# --- default case: use a single GPU on a shared node ---
#SBATCH --cpus-per-task=18 --mem=10G --nodes=1 --ntasks-per-node=4 --time=00:05:00 -o ./async0.out.%j
#SBATCH hetjob
#
#SBATCH --cpus-per-task=18 --mem=120G --nodes=1 --ntasks-per-node=4 --constraint="gpu" --gres=gpu:a100:4 --time=00:05:00 -o ./async1.out.%j 

#SBATCH --mail-type=none
#SBATCH --mail-user=userid@example.de


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

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores

srun --het-group=0 ./catalystSpace  : --het-group=1 ./neko tgv.case
