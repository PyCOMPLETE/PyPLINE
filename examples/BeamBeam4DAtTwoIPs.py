from mpi4py import MPI

import time
import numpy as np
from scipy import constants
proton_mass = constants.value('proton mass energy equivalent in MeV')*1E6

import xobjects as xo
import xtrack as xt
import xfields as xf
xt.enable_pyheadtail_interface() # has to be before the imports such that PyPLINEDParticles inherints the right class
#from PyHEADTAIL.trackers.transverse_tracking import TransverseSegmentMap
from PyPLINE.PyPLINEDBeamBeam import PyPLINEDBeamBeam
from PyPLINE.PyPLINEDParticles import PyPLINEDParticles

context = xo.ContextCpu(omp_num_threads=0)
my_rank = MPI.COMM_WORLD.Get_rank()

n_turn = int(1E4)
n_macroparticles = int(1E4)
bunch_intensity = 2E11
gamma = 7E12/proton_mass
epsn_x = 2E-6
epsn_y = 2E-6
betastar_x = 1.0
betastar_y = 1.0
sigma_z = 0.08
sigma_delta = 1E-4
Q_x = 0.31
Q_y = 0.32
beta_s = sigma_z/sigma_delta
Q_s = 1E-3


print('Generating one bunch per beam')
particles_B1b1 = PyPLINEDParticles(_context=context,
                     particlenumber_per_mp=bunch_intensity/n_macroparticles,
                     q0 = 1,
                     mass0 = proton_mass,
                     gamma0 = gamma,
                     x=np.sqrt(epsn_x*betastar_x/gamma)*(np.random.randn(n_macroparticles)),
                     px=np.sqrt(epsn_x/betastar_x/gamma)*np.random.randn(n_macroparticles),
                     y=np.sqrt(epsn_y*betastar_y/gamma)*(np.random.randn(n_macroparticles)),
                     py=np.sqrt(epsn_y/betastar_y/gamma)*np.random.randn(n_macroparticles),
                     zeta=sigma_z*np.random.randn(n_macroparticles),
                     delta=sigma_delta*np.random.randn(n_macroparticles),
                     name='B1b1',rank=0,number=0,
                     )

particles_B2b1 = PyPLINEDParticles(_context=context,
                     particlenumber_per_mp=bunch_intensity/n_macroparticles,
                     q0 = 1,
                     mass0 = proton_mass,
                     gamma0 = gamma,
                     x=np.sqrt(epsn_x*betastar_x/gamma)*(np.random.randn(n_macroparticles)),
                     px=np.sqrt(epsn_x/betastar_x/gamma)*np.random.randn(n_macroparticles),
                     y=np.sqrt(epsn_y*betastar_y/gamma)*(np.random.randn(n_macroparticles)),
                     py=np.sqrt(epsn_y/betastar_y/gamma)*np.random.randn(n_macroparticles),
                     zeta=sigma_z*np.random.randn(n_macroparticles),
                     delta=sigma_delta*np.random.randn(n_macroparticles),
                     name='B2b1',rank=1,number=0,
                     )

print('Instanciating beam-beam elements')
beamBeam_IP1 = PyPLINEDBeamBeam(_context=context,min_sigma_diff=1e-10,name='BBIP1',number=1)
beamBeam_IP2 = PyPLINEDBeamBeam(_context=context,min_sigma_diff=1e-10,name='BBIP2',number=2)
beamBeam_IP1.set_q0(1.0)
beamBeam_IP2.set_q0(1.0)

if my_rank == 0:
    my_bunch = particles_B1b1
    beamBeam_IP1.set_beta0(particles_B1b1.beta0[0])
    beamBeam_IP2.set_beta0(particles_B1b1.beta0[0])
    print('Instanciating B1 arcs')
    arc12_b1 = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = betastar_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = betastar_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = betastar_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = betastar_y, disp_y_1 = 0.0,
                           Q_x = Q_x/2, Q_y = Q_y/2,
                           beta_s = beta_s, Q_s = -Q_s/2,
                           energy_ref_increment=0.0,energy_increment=0)

    arc21_b1 = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = betastar_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = betastar_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = betastar_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = betastar_y, disp_y_1 = 0.0,
                           Q_x = Q_x/2, Q_y = Q_y/2,
                           beta_s = beta_s, Q_s = -Q_s/2,
                           energy_ref_increment=0.0,energy_increment=0)

    print('Builiding B1 pipeline')
    particles_B1b1.add_element_to_pipeline(beamBeam_IP1,[particles_B2b1.ID])
    particles_B1b1.add_element_to_pipeline(arc12_b1)
    particles_B1b1.add_element_to_pipeline(beamBeam_IP2,[particles_B2b1.ID])
    particles_B1b1.add_element_to_pipeline(arc21_b1)

elif my_rank == 1:
    my_bunch = particles_B2b1
    beamBeam_IP1.set_beta0(particles_B2b1.beta0[0])
    beamBeam_IP2.set_beta0(particles_B2b1.beta0[0])
    print('Instanciating B2 arcs')
    arc12_b2 = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = betastar_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = betastar_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = betastar_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = betastar_y, disp_y_1 = 0.0,
                           Q_x = Q_x/2, Q_y = Q_y/2,
                           beta_s = beta_s, Q_s = -Q_s/2,
                           energy_ref_increment=0.0,energy_increment=0)

    arc21_b2 = xt.LinearTransferMatrix(alpha_x_0 = 0.0, beta_x_0 = betastar_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = betastar_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = betastar_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = betastar_y, disp_y_1 = 0.0,
                           Q_x = Q_x/2, Q_y = Q_y/2,
                           beta_s = beta_s, Q_s = -Q_s/2,
                           energy_ref_increment=0.0,energy_increment=0)
    print('Builiding B2 pipeline')
    particles_B2b1.add_element_to_pipeline(beamBeam_IP1,[particles_B1b1.ID])
    particles_B2b1.add_element_to_pipeline(arc12_b2)
    particles_B2b1.add_element_to_pipeline(beamBeam_IP2,[particles_B1b1.ID])
    particles_B2b1.add_element_to_pipeline(arc21_b2)
else:
    print('Exiting useless process with rank {my_rank}')
    exit()

print('Start tracking')
turn_at_last_print = 0
time_at_last_print = time.time()
multiturn_data = np.zeros((n_turn,2),dtype=float)
while my_bunch.period < n_turn:
    abort = True
    if my_rank == 0:
        if my_bunch.period - turn_at_last_print == 1:
            time_per_turn = (time.time()-time_at_last_print)/(my_bunch.period - turn_at_last_print)
            print(f'Turn {my_bunch.period}, time per turn {time_per_turn}s',flush=True)
            turn_at_last_print = my_bunch.period
            time_at_last_print = time.time()
    current_period = my_bunch.period
    my_bunch.step()
    if current_period != my_bunch.period and current_period<n_turn:
        multiturn_data[current_period,0] = np.average(my_bunch.x)
        multiturn_data[current_period,1] = np.average(my_bunch.y)

print(f'Rank {my_rank} finished tracking')
np.savetxt(f'multiturndata_{my_bunch.ID.name}.csv',multiturn_data,delimiter=',')

if my_rank == 0:
    from matplotlib import pyplot as plt
    import os

    for beam_number in [1,2]:
        file_name = f'multiturndata_B{beam_number}b1.csv'
        if os.path.exists(file_name):
            multiturn_data = np.loadtxt(file_name,delimiter=',')
            positions_x = multiturn_data[:,0]/np.sqrt(epsn_x*betastar_x/gamma)
            positions_y = multiturn_data[:,1]/np.sqrt(epsn_y*betastar_y/gamma)

            plt.figure(beam_number)
            plt.plot(np.arange(len(positions_x))*1E-3,positions_x,'x')
            plt.xlabel('Turn [$10^{3}$]')
            plt.ylabel(f'Beam {beam_number} horizontal position [$\sigma$]')
            plt.figure(10+beam_number)
            plt.plot(np.arange(len(positions_y))*1E-3,positions_y,'x')
            plt.xlabel('Turn [$10^{3}$]')
            plt.ylabel(f'Beam {beam_number} vertical position [$\sigma$]')

            plt.figure(20+beam_number)
            freqs = np.fft.fftshift(np.fft.fftfreq(len(positions_x)))
            mask = freqs > 0
            myFFT = np.fft.fftshift(np.fft.fft(positions_x))
            plt.semilogy(freqs[mask],np.abs(myFFT[mask]))
            myFFT = np.fft.fftshift(np.fft.fft(positions_y))
            plt.semilogy(freqs[mask],np.abs(myFFT[mask]))
            plt.xlabel('Tune')
            plt.ylabel('Amplitude')
    plt.show()

