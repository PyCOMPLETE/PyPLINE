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

from PyPLINE.PyPLINEDParticles import PyPLINEDParticles
from PyPLINE.PyPLINEDWakeField import PyPLINEDWakeField

context = xo.ContextCpu(omp_num_threads=0)
myRank = MPI.COMM_WORLD.Get_rank()
nTurn = int(5E4)
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
waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'])
wake_field = PyPLINEDWakeField('Wake0',0,slicer_for_wakefields,waketable)

damper = TransverseDamper(dampingrate_x=33.0, dampingrate_y=33.0)
i_oct = 0.0
detx_x = 1.4E5*i_oct/550.0 # from PTC with ATS optics, telescopic factor 1.0
detx_y = -1.0E5*i_oct/550.0

particles_B1b1 = PyPLINEDParticles(circumference=circumference,particlenumber_per_mp=bunch_intensity/n_macroparticles,
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
                         name='B1b1',rank=0,number=0
                         )

arc = xt.LinearTransferMatrixWithDetuning(alpha_x_0 = 0.0, beta_x_0 = beta_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = beta_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = Q_x, Q_y = Q_y,
                           beta_s = beta_s, Q_s = -Q_s,
                           chroma_x = chroma,chroma_y = 0.0,
                           detx_x = detx_x,detx_y = detx_y,dety_y = detx_x, dety_x = detx_y,
                           energy_ref_increment=0.0,energy_increment=0)


particles_B1b1.addElementToPipeline(arc)
particles_B1b1.addElementToPipeline(wake_field)
#particles_B1b1.addElementToPipeline(damper)

print('Start tracking')
turns = np.arange(nTurn)
x = np.zeros(nTurn,dtype=float)
turnAtLastPrint = 0
timeAtLastPrint = time.time()
while particles_B1b1.period < nTurn:
    x[particles_B1b1.period] = np.average(particles_B1b1.x)
    abort = True
    if myRank == 0:
        if particles_B1b1.period - turnAtLastPrint == 1000:
            timePerTurn = (time.time()-timeAtLastPrint)/(particles_B1b1.period - turnAtLastPrint)
            print(f'Turn {particles_B1b1.period}, time per turn {timePerTurn}s',flush=True)
            turnAtLastPrint = particles_B1b1.period
            timeAtLastPrint = time.time()
    particles_B1b1.step()

x /= np.sqrt(normemit*beta_x/gamma/betar)
plt.figure(1)
plt.plot(turns,x,label=f'{i_oct}A')
iMin = 1000
iMax = nTurn-1000
if iMin >= iMax:
    iMin = 0
    iMax = nTurn
ampl = np.abs(hilbert(x))
b,a,r,p,stderr = linregress(turns[iMin:iMax],np.log(ampl[iMin:iMax]))
plt.plot(turns,np.exp(a+b*turns),'--k',label=f'{1/b:.3E} turns')
print(f'Growth rate {b*1E4} [$10^{-4}$/turn]')
plt.title('PyPLINE')
plt.legend(loc='upper left')
plt.xlabel('Turn')
plt.ylabel('x [$\sigma_x$]')

plt.show()
    
