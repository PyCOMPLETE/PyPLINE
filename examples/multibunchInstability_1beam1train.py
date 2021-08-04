import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import time
import numpy as np
from scipy import constants
protonMass = constants.value('proton mass energy equivalent in MeV')*1E6
from scipy.stats import linregress
from scipy.signal import hilbert
from matplotlib import pyplot as plt

import xobjects as xo
import xtrack as xt
xt.enable_pyheadtail_interface()

from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss
from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap
from PyHEADTAIL.trackers.detuners import ChromaticitySegment, AmplitudeDetuningSegment
from PyHEADTAIL.monitors.monitors import BunchMonitor

from PyPLINE.PyPLINEDParticles import PyPLINEDParticles
from PyPLINE.PyPLINEDWakeField import PyPLINEDWakeField

context = xo.ContextCpu(omp_num_threads=0)
nprocs = MPI.COMM_WORLD.Get_size()
myRank = MPI.COMM_WORLD.Get_rank()
nTurn = int(1E4)
n_bunch = 8
bunch_spacing = 25E-9
bunch_intensity = 1.8E11
n_macroparticles = int(1E4)
energy = 7E3 # [GeV]
gamma = energy*1E9/protonMass
betar = np.sqrt(1-1/gamma**2)
normemit = 1.8E-6
beta_x = 68.9
beta_y = 70.34
Q_x = 0.31
Q_y = 0.32
chroma = 10.0
sigma_4t = 1.2E-9
sigma_z = sigma_4t/4.0*constants.c
momentumCompaction = 3.483575072011584e-04
eta = momentumCompaction-1.0/gamma**2
voltage = 12.0E6
h = 35640
p0=constants.m_p*betar*gamma*constants.c
Q_s=np.sqrt(constants.e*voltage*eta*h/(2*np.pi*betar*constants.c*p0))
circumference = 26658.883199999
averageRadius = circumference/(2*np.pi)
sigma_delta = Q_s*sigma_z/(averageRadius*eta)
beta_s = sigma_z/sigma_delta
emit_s = 4*np.pi*sigma_z*sigma_delta*p0/constants.e # eVs for PyHEADTAIL

n_slices_wakes = 200
limit_z = 3 * sigma_z
wakefile = 'wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV_B1_2021_TeleIndex1_wake.dat'
slicer_for_wakefields = UniformBinSlicer(n_slices_wakes, z_cuts=(-limit_z, limit_z))
n_turns_wake = 1
waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'],n_turns_wake=n_bunch*n_turns_wake) # Trick to remain compatible with PyHEADTAIL single bunch version (n_turn_wake is in fact the maximum number of slice sets that will be taken into account in WakeKick._accumulate_source_signal). The attribute name should be changed in PyHEADTAIL.
wake_field = PyPLINEDWakeField('Wake',0,n_turns_wake,slicer_for_wakefields, waketable)

#damper = TransverseDamper(dampingrate_x=33.0, dampingrate_y=33.0)
i_oct = 0.0
detx_x = 1.4E5*i_oct/550.0 # from PTC with ATS optics, telescopic factor 1.0
detx_y = -1.0E5*i_oct/550.0

arc = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = beta_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = beta_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = Q_x, Q_y = Q_y,
                           beta_s = beta_s, Q_s = -Q_s,
                           energy_ref_increment=0.0,energy_increment=0)
#arc = xt.LinearTransferMatrixWithDetuning(alpha_x_0 = 0.0, beta_x_0 = beta_x, disp_x_0 = 0.0,
#                           alpha_x_1 = 0.0, beta_x_1 = beta_x, disp_x_1 = 0.0,
#                           alpha_y_0 = 0.0, beta_y_0 = beta_y, disp_y_0 = 0.0,
#                           alpha_y_1 = 0.0, beta_y_1 = beta_y, disp_y_1 = 0.0,
#                           Q_x = Q_x, Q_y = Q_y,
#                           beta_s = beta_s, Q_s = -Q_s,
#                           chroma_x = chroma,chroma_y = 0.0,
#                           detx_x = detx_x,detx_y = detx_y,dety_y = detx_x, dety_x = detx_y,
#                           energy_ref_increment=0.0,energy_increment=0)
particles_list = []
for bunch_number in range(n_bunch):
    particles_list.append(PyPLINEDParticles(circumference=circumference,particlenumber_per_mp=bunch_intensity/n_macroparticles,
                             _context=context,
                             q0 = 1,
                             mass0 = protonMass,
                             gamma0 = gamma,
                             x=np.sqrt(normemit*beta_x/gamma/betar)*np.random.randn(n_macroparticles),
                             px=np.sqrt(normemit/beta_x/gamma/betar)*np.random.randn(n_macroparticles),
                             y=np.sqrt(normemit*beta_y/gamma/betar)*np.random.randn(n_macroparticles),
                             py=np.sqrt(normemit/beta_y/gamma/betar)*np.random.randn(n_macroparticles),
                             zeta=sigma_z*np.random.randn(n_macroparticles),
                             delta=sigma_delta*np.random.randn(n_macroparticles),
                             name=f'B1b{bunch_number}',rank=bunch_number%nprocs,number=bunch_number,delay=bunch_number*bunch_spacing
                             )
                    )

for bunch_number in range(n_bunch):
    partnerIDs = []
    for partner_bunch_number in range(n_bunch):
        if partner_bunch_number != bunch_number:
            partnerIDs.append(particles_list[partner_bunch_number].ID)
    particles_list[bunch_number].addElementToPipeline(arc)
    particles_list[bunch_number].addElementToPipeline(wake_field,partnerIDs)
    particles_list[bunch_number].addElementToPipeline(BunchMonitor(filename=f'Multibunch_B1b{bunch_number}',n_steps=nTurn))

my_particles_list = []
for particles in particles_list:
    if particles.ID.rank == myRank:
        my_particles_list.append(particles)

print('Start tracking')
abort = False
turnAtLastPrint = 0
timeAtLastPrint = time.time()
while not abort:
    atLeastOneBunchIsActive = False
    if myRank == 0:
        if particles_list[0].period - turnAtLastPrint == 100:
            timePerTurn = (time.time()-timeAtLastPrint)/(particles_list[0].period - turnAtLastPrint)
            print(f'Turn {particles_list[0].period}, time per turn {timePerTurn}s',flush=True)
            turnAtLastPrint = particles_list[0].period
            timeAtLastPrint = time.time()
    for particles in my_particles_list:
        if particles.period <= nTurn:
            particles.step()
            atLeastOneBunchIsActive = True
            
    if not atLeastOneBunchIsActive:
        abort = True
print(f'Rank {myRank} is done')

for bunch_number in range(n_bunch):
    file_name = f'Multibunch_B1b{bunch_number}.h5'
    if os.path.exists(file_name):
        f = h5py.File(file_name, 'r')
        x = np.array(f.get('Bunch/mean_x'))/sigma_x
        f.close()

        plt.figure(bunch_number)
        plt.plot(np.arange(len(x))*1E-3,x,'x')
        plt.xlabel('Turn [$10^{3}$]')
        plt.ylabel(f'Bunch {bunch_number} horizontal position [$\sigma$]')
        plt.savefig(f'PyPLINE_LHCCB_8b_B1b{bunch_number}.png')

plt.show()

