from mpi4py import MPI

import time
import numpy as np
from scipy import constants as cst
from generators import PyPLINEDParticleGenerator
from PyHEADTAIL.particles.generators import gaussian2D
from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyHEADTAIL.monitors.monitors import BunchMonitor
from RoundBeamBeam4D import RoundBeamBeam4D

if __name__ == '__main__':
    myRank = MPI.COMM_WORLD.Get_rank()
    nTurn = int(1E5)
    n_macroparticles = 100
    number_of_particles = 5E10
    energy = 7E3
    epsn_x = 2E-6
    epsn_y = 2E-6
    epsn_z = 2.5
    Qx = 0.31
    Qy = 0.32

    gamma = 7E3/0.938
    beta = np.sqrt(1. - gamma**-2)
    p0 = np.sqrt(gamma**2 - 1) * cst.m_p * cst.c
    eps_geo_x = epsn_x / (beta * gamma)
    eps_geo_y = epsn_y / (beta * gamma)
    eps_geo_z = epsn_z * cst.e / (4. * np.pi * p0)
    generator = PyPLINEDParticleGenerator(
        macroparticlenumber=n_macroparticles,
        intensity=number_of_particles,
        charge=cst.e,
        mass=cst.m_p,
        circumference=27E3,
        gamma=gamma,
        distribution_x=gaussian2D(eps_geo_x),
        distribution_y=gaussian2D(eps_geo_y),
        distribution_z=gaussian2D(eps_geo_z))

    B1b1 = generator.generate('B1b1',0,0)
    B2b1 = generator.generate('B2b1',1,0)

    bunches = []
    bunches.append(B1b1)
    bunches.append(B2b1)
    myBunches = []
    for bunch in bunches:
        if bunch.ID.rank == myRank:
            myBunches.append(bunch)


    beamBeamIP1 = RoundBeamBeam4D('BBIP1',0)
    arc1 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = 1.0, D_x_s0 = 0.0,
                                alpha_x_s1 = 0.0, beta_x_s1 = 1.0, D_x_s1 = 0.0,
                                alpha_y_s0 = 0.0, beta_y_s0 = 1.0, D_y_s0 = 0.0,
                                alpha_y_s1 = 0.0, beta_y_s1 = 1.0, D_y_s1 = 0.0,
                                 dQ_x = Qx/2, dQ_y=Qy/2)
    beamBeamIP2 = RoundBeamBeam4D('BBIP2',1)
    arc2 = TransverseSegmentMap(alpha_x_s0 = 0.0, beta_x_s0 = 1.0, D_x_s0 = 0.0,
                                alpha_x_s1 = 0.0, beta_x_s1 = 1.0, D_x_s1 = 0.0,
                                alpha_y_s0 = 0.0, beta_y_s0 = 1.0, D_y_s0 = 0.0,
                                alpha_y_s1 = 0.0, beta_y_s1 = 1.0, D_y_s1 = 0.0,
                                 dQ_x = Qx/2, dQ_y=Qy/2)

    #Buliding pipeline for B1
    B1b1.addElementToPipeline(beamBeamIP1,[B2b1.ID])
    B1b1.addElementToPipeline(arc1)
    B1b1.addElementToPipeline(beamBeamIP2,[B2b1.ID])
    B1b1.addElementToPipeline(arc2)
    B1b1.addElementToPipeline(BunchMonitor(filename=f'B1b1',n_steps=nTurn))
    #Buliding pipeline for B2
    B2b1.addElementToPipeline(beamBeamIP1,[B1b1.ID])
    B2b1.addElementToPipeline(arc1)
    B2b1.addElementToPipeline(beamBeamIP2,[B1b1.ID])
    B2b1.addElementToPipeline(arc2)
    #B2b1.addElementToPipeline(BunchMonitor(filename='B2b1',n_steps=nTurn))

    for bunch in myBunches:
        print('bunch',bunch.ID.name,'on rank',myRank)

    abort = False
    turnAtLastPrint = 0
    timeAtLastPrint = time.time()
    while not abort:
        atLeastOneBunchIsActive = False
        if myRank == 0:
            if myBunches[0].period - turnAtLastPrint == 100:
                timePerTurn = (time.time()-timeAtLastPrint)/(myBunches[0].period - turnAtLastPrint)
                print(f'Turn {myBunches[0].period}, time per turn {timePerTurn}s')
                turnAtLastPrint = myBunches[0].period
                timeAtLastPrint = time.time()
        for bunch in myBunches:
            if bunch.period <= nTurn:
                bunch.step()
                atLeastOneBunchIsActive = True
                
        if not atLeastOneBunchIsActive:
            abort = True

