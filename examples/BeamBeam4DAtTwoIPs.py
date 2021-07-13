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
from PyPLINEDBeamBeam import PyPLINEDBeamBeam
from PyPLINEDParticles import PyPLINEDParticles

context = xo.ContextCpu(omp_num_threads=0)

if __name__ == '__main__':
    myRank = MPI.COMM_WORLD.Get_rank()
    nTurn = int(1E4)
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
    beta = np.sqrt(1. - gamma**-2)
    p0c = np.sqrt(gamma**2 - 1) * cst.m_p * cst.c
    eps_geo_x = epsn_x / (beta * gamma)
    eps_geo_y = epsn_y / (beta * gamma)

    particlesB1b1 = PyPLINEDParticles(_context=context,
                         x=np.sqrt(eps_geo_x*betastar_x)*(np.random.randn(n_macroparticles)),
                         px=np.sqrt(eps_geo_x/betastar_x)*np.random.randn(n_macroparticles),
                         y=np.sqrt(eps_geo_y*betastar_y)*(np.random.randn(n_macroparticles)),
                         py=np.sqrt(eps_geo_y/betastar_y)*np.random.randn(n_macroparticles),
                         zeta=sigma_z*np.random.randn(n_macroparticles),
                         delta=sigma_delta*np.random.randn(n_macroparticles),
                         name='B1b1',rank=0,number=0,
                         )
    particlesB2b1.gamma = gamma #  TODO this is a quick fix, since the setter of gamma is not called by the constructor, so the related quantities are not set
    particlesB2b1 = PyPLINEDParticles(_context=context,
                         x=np.sqrt(eps_geo_x*betastar_x)*(np.random.randn(n_macroparticles)),
                         px=np.sqrt(eps_geo_x/betastar_x)*np.random.randn(n_macroparticles),
                         y=np.sqrt(eps_geo_y*betastar_y)*(np.random.randn(n_macroparticles)),
                         py=np.sqrt(eps_geo_y/betastar_y)*np.random.randn(n_macroparticles),
                         zeta=sigma_z*np.random.randn(n_macroparticles),
                         delta=sigma_delta*np.random.randn(n_macroparticles),
                         name='B2b1',rank=1,number=0,
                         )
    particlesB2b1.gamma = gamma

    print('Instanciating beam-beam elements')
    beamBeamIP1 = PyPLINEDBeamBeam(_context=context,min_sigma_diff=1e-10,name='BBIP1',number=0)
    beamBeamIP2 = PyPLINEDBeamBeam(_context=context,min_sigma_diff=1e-10,name='BBIP2',number=0)

    if myRank == 0:
        myBunch = particlesB1b1
        beamBeamIP1.setQ0(particlesB2b1.q0)
        beamBeamIP1.setBeta0(particlesB2b1.beta0)
        beamBeamIP2.setQ0(particlesB2b1.q0)
        beamBeamIP2.setBeta0(particlesB2b1.beta0)
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
        print('Builiding B1b1 pipeline')
        particlesB1b1.addElementToPipeline(beamBeamIP1,[particlesB2b1.ID])
        particlesB1b1.addElementToPipeline(arc12_b1)
        particlesB1b1.addElementToPipeline(beamBeamIP2,[particlesB2b1.ID])
        particlesB1b1.addElementToPipeline(arc21_b1)
        particlesB1b1.addElementToPipeline(BunchMonitor(filename=f'B1b1',n_steps=nTurn))

    elif myRank == 1:
        myBunch = particlesB2b1
        beamBeamIP1.setQ0(particlesB1b1.q0)
        beamBeamIP1.setBeta0(particlesB1b1.beta0)
        beamBeamIP2.setQ0(particlesB1b1.q0)
        beamBeamIP2.setBeta0(particlesB1b1.beta0)
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
        print('Builiding B2b1 pipeline')
        particlesB2b1.addElementToPipeline(beamBeamIP1,[particlesB1b1.ID])
        particlesB2b1.addElementToPipeline(arc12_b2)
        particlesB2b1.addElementToPipeline(beamBeamIP2,[particlesB1b1.ID])
        particlesB2b1.addElementToPipeline(arc21_b2)
        particlesB2b1.addElementToPipeline(BunchMonitor(filename=f'B2b1',n_steps=nTurn))
    else:
        print('Exiting process with rank {myRank}')
        exit()

    print('Start tracking')
    turnAtLastPrint = 0
    timeAtLastPrint = time.time()
    while myBunch.period < nTurn:
        abort = True
        if myRank == 0:
            if myBunch.period - turnAtLastPrint == 1:
                timePerTurn = (time.time()-timeAtLastPrint)/(myBunch.period - turnAtLastPrint)
                print(f'Turn {myBunch.period}, time per turn {timePerTurn}s',flush=True)
                turnAtLastPrint = myBunch.period
                timeAtLastPrint = time.time()
        myBunch.step()


