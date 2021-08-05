# PyPLINE

PyPLINE implements an asynchronous pipeline parallelisation based on MPI featuring tracking capabilities from PyHEADTAIL and xsuite

S. Furuseth and X. Buffat. Parallel high-performance multi-beam multi-bunch simulations. Computer Physics Communications, 244, 06 2019 http://dx.doi.org/10.1016/j.cpc.2019.06.006

## Example of run command with mpi4py:
mpiexec -n 2 python -m mpi4py BeamBeam4DAtTwoIPs.py

Note:
 - h5py from PyHEADTAIL main branch does not support MPI. Either use a multibunch branch or don't use PyHEADTAIL monitors 
 - On some machines H5 locking mechanism does not work, it can be disabled with:
export HDF5_USE_FILE_LOCKING='FALSE'
