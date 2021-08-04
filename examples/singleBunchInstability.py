from mpi4py import MPI

import time
import numpy as np
from scipy import constants
proton_mass = constants.value('proton mass energy equivalent in MeV')*1E6
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

from PyPLINE.PyPLINEDParticles import PyPLINEDParticles
from PyPLINE.PyPLINEDWakeField import PyPLINEDWakeField

context = xo.ContextCpu(omp_num_threads=0)
my_rank = MPI.COMM_WORLD.Get_rank()

n_turn = int(5E4)
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
damper_time = 33 # turns
i_oct = 300.0 # Ampere in LHC focusing octupoles (ATS optics, telescopic factor 1.0)

# setting up wake fields
n_slices_wakes = 200
n_turns_wake = 1
limit_z = 3 * sigma_z
wakefile = '/afs/cern.ch/work/x/xbuffat/PyCOMPLETE/PyPLINE/examples/wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV_B1_2021_TeleIndex1_wake.dat'
slicer_for_wakefields = UniformBinSlicer(n_slices_wakes, z_cuts=(-limit_z, limit_z))
waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'])
wake_field = PyPLINEDWakeField('Wake0',0,n_turns_wake,slicer_for_wakefields,waketable)

# setting up damper
damper = TransverseDamper(dampingrate_x=damper_time, dampingrate_y=damper_time)

# setting up lattice
detx_x = 1.4E5*i_oct/550.0
detx_y = -1.0E5*i_oct/550.0
arc = xt.LinearTransferMatrixWithDetuning(alpha_x_0 = 0.0, beta_x_0 = beta_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = beta_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = Q_x, Q_y = Q_y,
                           beta_s = beta_s, Q_s = -Q_s,
                           chroma_x = chroma,chroma_y = 0.0,
                           detx_x = detx_x,detx_y = detx_y,dety_y = detx_x, dety_x = detx_y,
                           energy_ref_increment=0.0,energy_increment=0)

# generate beam
particles = PyPLINEDParticles(circumference=circumference,particlenumber_per_mp=bunch_intensity/n_macroparticles,
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
                         name='B1b1',rank=0,number=0
                         )

# setting up pipeline
particles.add_element_to_pipeline(arc)
particles.add_element_to_pipeline(wake_field)
particles.add_element_to_pipeline(damper)

print('Start tracking')
turns = np.arange(n_turn)
x = np.zeros(n_turn,dtype=float)
turn_at_last_print = 0
time_at_last_print = time.time()
while particles.period < n_turn:
    x[particles.period] = np.average(particles.x)
    abort = True
    if my_rank == 0:
        if particles.period - turn_at_last_print == 1000:
            time_per_turn = (time.time()-time_at_last_print)/(particles.period - turn_at_last_print)
            print(f'Turn {particles.period}, time per turn {time_per_turn}s',flush=True)
            turn_at_last_print = particles.period
            time_at_last_print = time.time()
    particles.step()
print('Finished tracking')

x /= np.sqrt(epsn*beta_x/gamma/betar)
plt.figure(1)
plt.plot(turns,x,label=f'{i_oct}A')
iMin = 1000
iMax = n_turn-1000
if iMin >= iMax:
    iMin = 0
    iMax = n_turn
ampl = np.abs(hilbert(x))
b,a,r,p,stderr = linregress(turns[iMin:iMax],np.log(ampl[iMin:iMax]))
plt.plot(turns,np.exp(a+b*turns),'--k',label=f'{1/b:.3E} turns')
print(f'Growth rate {b*1E4} [$10^{-4}$/turn]')
plt.title('PyPLINE')
plt.legend(loc='upper left')
plt.xlabel('Turn')
plt.ylabel('x [$\sigma_x$]')

plt.show()
    
