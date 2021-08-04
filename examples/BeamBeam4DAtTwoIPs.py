from mpi4py import MPI

import time
import numpy as np
from scipy import constants as cst

import xobjects as xo
import xtrack as xt
import xfields as xf
xt.enable_pyheadtail_interface() # has to be before the imports such that PyPLINEDParticles inherints the right class
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyHEADTAIL.monitors.monitors import BunchMonitor
from PyPLINE.PyPLINEDBeamBeam import PyPLINEDBeamBeam
from PyPLINE.PyPLINEDParticles import PyPLINEDParticles

context = xo.ContextCpu(omp_num_threads=0)
my_rank = MPI.COMM_WORLD.Get_rank()

n_turn = int(1E4)
n_macroparticles = int(1E6)
number_of_particles = 2E11
energy = 7E3
epsn_x = 2E-6
epsn_y = 2E-6
betastar_x = 1.0
betastar_y = 1.0
sigma_z = 0.08
sigma_delta = 1E-4
Qx = 0.31
Qy = 0.32
gamma = 7E3/0.938
betar = np.sqrt(1. - gamma**-2)
p0c = np.sqrt(gamma**2 - 1) * cst.m_p * cst.c
eps_geo_x = epsn_x / (betar * gamma)
eps_geo_y = epsn_y / (betar * gamma)

print('Generating one bunch per beam')
particles_B1b1 = PyPLINEDParticles(_context=context,
                     x=np.sqrt(eps_geo_x*betastar_x)*(np.random.randn(n_macroparticles)),
                     px=np.sqrt(eps_geo_x/betastar_x)*np.random.randn(n_macroparticles),
                     y=np.sqrt(eps_geo_y*betastar_y)*(np.random.randn(n_macroparticles)),
                     py=np.sqrt(eps_geo_y/betastar_y)*np.random.randn(n_macroparticles),
                     zeta=sigma_z*np.random.randn(n_macroparticles),
                     delta=sigma_delta*np.random.randn(n_macroparticles),
                     name='B1b1',rank=0,number=0,
                     )
particles_B2b1 = PyPLINEDParticles(_context=context,
                     x=np.sqrt(eps_geo_x*betastar_x)*(np.random.randn(n_macroparticles)),
                     px=np.sqrt(eps_geo_x/betastar_x)*np.random.randn(n_macroparticles),
                     y=np.sqrt(eps_geo_y*betastar_y)*(np.random.randn(n_macroparticles)),
                     py=np.sqrt(eps_geo_y/betastar_y)*np.random.randn(n_macroparticles),
                     zeta=sigma_z*np.random.randn(n_macroparticles),
                     delta=sigma_delta*np.random.randn(n_macroparticles),
                     name='B2b1',rank=1,number=0,
                     )

print('Instanciating beam-beam elements')
beamBeam_IP1 = PyPLINEDBeamBeam(_context=context,min_sigma_diff=1e-10,name='BBIP1',number=0)
beamBeam_IP2 = PyPLINEDBeamBeam(_context=context,min_sigma_diff=1e-10,name='BBIP2',number=0)

if my_rank == 0:
    my_bunch = particles_B1b1
    beamBeam_IP1.setQ0(particles_B2b1.q0)
    beamBeam_IP1.setBeta0(particles_B2b1.beta0[0])
    beamBeam_IP2.setQ0(particles_B2b1.q0)
    beamBeam_IP2.setBeta0(particles_B2b1.beta0[0])
    print('Instanciating B1 arcs')
    arc12_b1 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = 1.0, D_x_s0 = 0.0,
                            alpha_x_s1 = 0.0, beta_x_s1 = 1.0, D_x_s1 = 0.0,
                            alpha_y_s0 = 0.0, beta_y_s0 = 1.0, D_y_s0 = 0.0,
                            alpha_y_s1 = 0.0, beta_y_s1 = 1.0, D_y_s1 = 0.0,
                             dQ_x = Qx/2, dQ_y=Qy/2)
    arc21_b1 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = 1.0, D_x_s0 = 0.0,
                            alpha_x_s1 = 0.0, beta_x_s1 = 1.0, D_x_s1 = 0.0,
                            alpha_y_s0 = 0.0, beta_y_s0 = 1.0, D_y_s0 = 0.0,
                            alpha_y_s1 = 0.0, beta_y_s1 = 1.0, D_y_s1 = 0.0,
                             dQ_x = Qx/2, dQ_y=Qy/2)
    print('Builiding B1 pipeline')
    particles_B1b1.add_element_to_pipeline(beamBeam_IP1,[particles_B2b1.ID])
    particles_B1b1.add_element_to_pipeline(arc12_b1)
    particles_B1b1.add_element_to_pipeline(beamBeam_IP2,[particles_B2b1.ID])
    particles_B1b1.add_element_to_pipeline(arc21_b1)
    particles_B1b1.add_element_to_pipeline(BunchMonitor(filename=f'BeamBeam_B1b1',n_steps=n_turn))

elif my_rank == 1:
    my_bunch = particles_B2b1
    beamBeam_IP1.setQ0(particles_B1b1.q0)
    beamBeam_IP1.setBeta0(particles_B1b1.beta0[0])
    beamBeam_IP2.setQ0(particles_B1b1.q0)
    beamBeam_IP2.setBeta0(particles_B1b1.beta0[0])
    print('Instanciating B2 arcs')
    arc12_b2 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = 1.0, D_x_s0 = 0.0,
                                alpha_x_s1 = 0.0, beta_x_s1 = 1.0, D_x_s1 = 0.0,
                                alpha_y_s0 = 0.0, beta_y_s0 = 1.0, D_y_s0 = 0.0,
                                alpha_y_s1 = 0.0, beta_y_s1 = 1.0, D_y_s1 = 0.0,
                                dQ_x = Qx/2, dQ_y=Qy/2)
    arc21_b2 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = 1.0, D_x_s0 = 0.0,
                                alpha_x_s1 = 0.0, beta_x_s1 = 1.0, D_x_s1 = 0.0,
                                alpha_y_s0 = 0.0, beta_y_s0 = 1.0, D_y_s0 = 0.0,
                                alpha_y_s1 = 0.0, beta_y_s1 = 1.0, D_y_s1 = 0.0,
                                dQ_x = Qx/2, dQ_y=Qy/2)
    print('Builiding B2 pipeline')
    particles_B2b1.add_element_to_pipeline(beamBeam_IP1,[particles_B1b1.ID])
    particles_B2b1.add_element_to_pipeline(arc12_b2)
    particles_B2b1.add_element_to_pipeline(beamBeam_IP2,[particles_B1b1.ID])
    particles_B2b1.add_element_to_pipeline(arc21_b2)
    particles_B2b1.add_element_to_pipeline(BunchMonitor(filename=f'BeamBeam_B2b1',n_steps=n_turn))
else:
    print('Exiting useless process with rank {my_rank}')
    exit()

print('Start tracking')
turn_at_last_print = 0
time_at_last_print = time.time()
while my_bunch.period < n_turn:
    abort = True
    if my_rank == 0:
        if my_bunch.period - turn_at_last_print == 1:
            time_per_turn = (time.time()-time_at_last_print)/(my_bunch.period - turn_at_last_print)
            print(f'Turn {my_bunch.period}, time per turn {time_per_turn}s',flush=True)
            turn_at_last_print = my_bunch.period
            time_at_last_print = time.time()
    my_bunch.step()


