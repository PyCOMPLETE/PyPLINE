# PyPLINE

PyPLINE implements an asynchronous pipeline parallelisation based on MPI featuring tracking capabilities from PyHEADTAIL and xsuite

S. Furuseth and X. Buffat. Parallel high-performance multi-beam multi-bunch simulations. Computer Physics Communications, 244, 06 2019 http://dx.doi.org/10.1016/j.cpc.2019.06.006

## Example of run command with mpi4py:
mpiexec -n 2 python -m mpi4py BeamBeam4DAtTwoIPs.py

Note:
 - h5py from PyHEADTAIL main branch does not support MPI. Either use a multibunch branch or don't use PyHEADTAIL monitors 
 - On some machines H5 locking mechanism does not work, it can be disabled with:
export HDF5_USE_FILE_LOCKING='FALSE'

## Common code structure:

 - Instanciation of PyPLINEDParticles objects with different combinations of attributes 'rank' and 'number' (the attribute 'name' can be used for convenience, but the 'rank' and 'number' must uniquely identify a PyPLINEDParticles object). During instanciation, the particles coordinate are allocated in memory only for PyPLINEDParticle objects with an attribute 'rank' correspdoning to the actual MPI rank of the process (their attribute is_real is set to True). The others are 'fake' and are used only to generate their corresponding IDs.
 - Instanciation of the elements (They can be PyPLINEDElement or xtrack / xfield / PyHEADTAIL Element)
 - Building of the pipeline of the real PyPLINEDParticles objects. For all PyPLINEDElement, a list of IDs of the other PyPLINEDParticles objects with which communication will be required (i.e. 'partners') should be provided.
 - Tracking by calling 'step' iteratively on all the real PyPLINEDParticles objects. (When calling step, the PyPLINEDParticles objects will try to execute the action described by the next element in its pipeline. If the messages cannot be obtained from the partners, no actions take place and the executions will be attempted again at the next call.)
 - Post processing

## Running on CERN's HPC cluster with miniconda and MVAPICH2

# Installation

cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source miniconda3/bin/activate
pip install numpy scipy matplotlib cython
module load mpi/mvapich2/2.3
env MPICC=/usr/local/mpi/mvapich2/2.3/bin/mpicc python -m pip install mpi4py
git clone https://github.com/xsuite/xobjects
pip install -e xobjects
git clone https://github.com/xsuite/xline
pip install -e xline
git clone https://github.com/xsuite/xpart
pip install -e xpart
git clone https://github.com/xsuite/xtrack
pip install -e xtrack
git clone https://github.com/xsuite/xfields
pip install -e xfields
git clone https://github.com/PyCOMPLETE/PyHEADTAIL.git
git install -e PyHEADTAIL
git clone https://github.com/PyCOMPLETE/PyPLINE.git
git install -e PyPLINE

# Example job file

#!/bin/bash

#SBATCH --partition inf-long
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 40
#SBATCH --cpus-per-task 1
#SBATCH --time 10:00:00

module load mpi/mvapich2/2.3
source ~/miniconda3_mvapich2/bin/activate
export MV2_ENABLE_AFFINITY=0
export LD_PRELOAD=/usr/local/mpi/mvapich2/2.3/lib64/libmpi.so
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -np $SLURM_NTASKS python -m mpi4py mycode.py



