import mpi4py
from mpi4py import MPI

import time
import numpy as np
from scipy import constants
proton_mass = constants.value('proton mass energy equivalent in MeV')*1E6

import xobjects as xo
import xtrack as xt
xt.enable_pyheadtail_interface()

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable
from PyHEADTAIL.monitors.monitors import BunchMonitor

from PyPLINE.PyPLINEDParticles import PyPLINEDParticles
from PyPLINE.PyPLINEDWakeField import PyPLINEDWakeField

context = xo.ContextCpu(omp_num_threads=0)
n_available_procs = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()

n_turn = int(1E4)
n_bunch = 8
bunch_spacing = 25E-9
bunch_intensity = 1.8E11
n_macroparticles = int(1E4)
energy = 7E3 # [GeV]
gamma = energy*1E9/proton_mass
betar = np.sqrt(1-1/gamma**2)
epsn = 1.8E-6
beta_x = 68.9
beta_y = 70.34
Q_x = 0.31
Q_y = 0.32
chroma = 10.0
sigma_4t = 1.2E-9
sigma_z = sigma_4t/4.0*constants.c
momentum_compaction = 3.483575072011584e-04
eta = momentum_compaction-1.0/gamma**2
voltage = 12.0E6
h = 35640
p0=constants.m_p*betar*gamma*constants.c
Q_s=np.sqrt(constants.e*voltage*eta*h/(2*np.pi*betar*constants.c*p0))
circumference = 26658.883199999
average_radius = circumference/(2*np.pi)
sigma_delta = Q_s*sigma_z/(average_radius*eta)
beta_s = sigma_z/sigma_delta
emit_s = 4*np.pi*sigma_z*sigma_delta*p0/constants.e # eVs for PyHEADTAIL

# setting up wake fields
n_slices_wakes = 200
n_turns_wake = 1
limit_z = 3 * sigma_z
wakefile = '/afs/cern.ch/work/x/xbuffat/PyCOMPLETE/PyPLINE/examples/wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV_B1_2021_TeleIndex1_wake.dat'
slicer_for_wakefields = UniformBinSlicer(n_slices_wakes, z_cuts=(-limit_z, limit_z))
waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'],n_turns_wake=n_bunch*n_turns_wake) # Trick to remain compatible with PyHEADTAIL single bunch version (n_turn_wake is in fact the maximum number of slice sets that will be taken into account in WakeKick._accumulate_source_signal). The attribute name should be changed in PyHEADTAIL.
wake_field = PyPLINEDWakeField('Wake',0,n_turns_wake,slicer_for_wakefields, waketable)

# setting up lattice
arc = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = beta_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = beta_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = Q_x, Q_y = Q_y,
                           beta_s = beta_s, Q_s = -Q_s,
                           energy_ref_increment=0.0,energy_increment=0)

# generating bunches (the rank is assigned according to available processors)
particles_list = []
for bunch_number in range(n_bunch):
    particles_list.append(PyPLINEDParticles(circumference=circumference,particlenumber_per_mp=bunch_intensity/n_macroparticles,
                             _context=context,
                             q0 = 1,
                             mass0 = proton_mass,
                             gamma0 = gamma,
                             x=np.sqrt(epsn*beta_x/gamma/betar)*np.random.randn(n_macroparticles),
                             px=np.sqrt(epsn/beta_x/gamma/betar)*np.random.randn(n_macroparticles),
                             y=np.sqrt(epsn*beta_y/gamma/betar)*np.random.randn(n_macroparticles),
                             py=np.sqrt(epsn/beta_y/gamma/betar)*np.random.randn(n_macroparticles),
                             zeta=sigma_z*np.random.randn(n_macroparticles),
                             delta=sigma_delta*np.random.randn(n_macroparticles),
                             name=f'B1b{bunch_number}',rank=bunch_number%n_available_procs,number=bunch_number,delay=bunch_number*bunch_spacing
                             )
                    )

# determining the bunches that will be tracked by this process and build their pipeline
my_particles_list = []
for particles in particles_list:
    if particles.is_real:
        partners_IDs = []
        for partner_bunch_number in range(n_bunch):
            if partner_bunch_number != bunch_number:
                partners_IDs.append(particles_list[partner_bunch_number].ID)
        particles.add_element_to_pipeline(arc)
        particles.add_element_to_pipeline(wake_field,partners_IDs)
        particles.add_element_to_pipeline(BunchMonitor(filename=f'Multibunch_B1b{bunch_number}',driver='mpio',n_steps=n_turn))

        my_particles_list.append(particles)

print('Start tracking')
abort = False
turn_at_last_print = 0
time_at_last_print = time.time()
while not abort:
    at_least_one_bunch_is_active = False
    if my_rank == 0:
        if particles_list[0].period - turn_at_last_print == 100:
            timePerTurn = (time.time()-time_at_last_print)/(particles_list[0].period - turn_at_last_print)
            print(f'Turn {particles_list[0].period}, time per turn {timePerTurn}s',flush=True)
            turn_at_last_print = particles_list[0].period
            time_at_last_print = time.time()
    for particles in my_particles_list:
        if particles.period <= n_turn:
            particles.step()
            at_least_one_bunch_is_active = True
            
    if not at_least_one_bunch_is_active:
        abort = True
print(f'Rank {my_rank} finished tracking')

if my_rank == 0:
    from matplotlib import pyplot as plt
    import os,h5py

    for bunch_number in range(n_bunch):
        file_name = f'Multibunch_B1b{bunch_number}.h5'
        if os.path.exists(file_name):
            f = h5py.File(file_name, 'r')
            x = np.array(f.get('Bunch/mean_x'))/np.sqrt(epsn*beta_x/gamma/betar)
            f.close()

            plt.figure(bunch_number)
            plt.plot(np.arange(len(x))*1E-3,x,'x')
            plt.xlabel('Turn [$10^{3}$]')
            plt.ylabel(f'Bunch {bunch_number} horizontal position [$\sigma$]')

    plt.show()

