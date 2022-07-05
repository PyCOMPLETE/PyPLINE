import pickle,copy,time
from collections import deque
import numpy as np

#from mpi4py import MPI

from scipy import constants

import xtrack as xt
#import xfields as xf
from PyPLINE.PyPLINEDElement import PyPLINEDElement

# It is assumed that the beam is matched to the optics at the kicker at entrance of the element (and it still is after the element)
# If scale_gain == True, the bunch spacing should be specified. The gain is then scaled depending on Q, such that central bunches experience the specified gain
class PyPLINEDCrabCavityAmplitudeFeedback(PyPLINEDElement):

    def __init__(self,name,number,slicer,n_turns=1,gain_x=0.0,gain_y=0.0,crab_Q=1,crab_frequency = 400.6E6, demodulation_frequency=400.6E6,group_delay=1,energy_kick=True,linear_kick = False,
                     Q_to_pickup_x=0, Q_to_pickup_y=0,Q_pickup_to_kicker_x = 0.0, Q_pickup_to_kicker_y = 0.0,
                     beta_x_pickup=1.0, beta_x_kicker=1.0, beta_y_pickup=1.0, beta_y_kicker=1.0,
                     alpha_x_pickup=0.0, alpha_x_kicker=0.0, alpha_y_pickup=0.0, alpha_y_kicker=0.0,
                     disp_x_pickup=0.0, disp_x_kicker=0.0, disp_y_pickup=0.0, disp_y_kicker=0.0,
                     x_ref_pickup = 0.0, px_ref_pickup = 0.0, x_ref_kicker = 0.0, px_ref_kicker = 0.0,
                     y_ref_pickup = 0.0, py_ref_pickup = 0.0, y_ref_kicker = 0.0, py_ref_kicker = 0.0,
                     scale_gain = False,bunch_spacing = 0.0):
        PyPLINEDElement.__init__(self,name,number)

        self.n_turns = n_turns
        self.slicer = slicer
        self.crab_omega = 2.0*np.pi*crab_frequency
        self.crab_k = self.crab_omega/constants.c
        self.linear_kick = linear_kick
        self.demod_k = 2.0*np.pi*demodulation_frequency/constants.c
        self.crab_Q = crab_Q
        self.energy_kick = energy_kick

        weight = 1.0
        if scale_gain:
            if bunch_spacing <= 0.0:
                raise ValueError('Invalid bunch spacing',bunch_spacing)
            bunch_scale = self.crab_omega*bunch_spacing/(2*self.crab_Q)
            weight = (np.exp(bunch_scale)+1)/(np.exp(bunch_scale)-1)
        
        self.gain_x = gain_x/weight
        self.gain_y = gain_y/weight

        self.group_delay = group_delay

        self.transfer_x = np.sin(2.0*np.pi*Q_pickup_to_kicker_x)/np.sqrt(beta_x_pickup*beta_x_kicker)
        self.transfer_y = np.sin(2.0*np.pi*Q_pickup_to_kicker_y)/np.sqrt(beta_y_pickup*beta_y_kicker)

        self._transfer_to_pickup = xt.LinearTransferMatrix(alpha_x_0 = alpha_x_kicker, beta_x_0 = beta_x_kicker, disp_x_0 = disp_x_kicker,
                       alpha_x_1 = alpha_x_pickup, beta_x_1 = beta_x_pickup, disp_x_1 = disp_x_pickup,
                       alpha_y_0 = alpha_y_kicker, beta_y_0 = beta_y_kicker, disp_y_0 = disp_y_kicker,
                       alpha_y_1 = alpha_y_pickup, beta_y_1 = beta_y_pickup, disp_y_1 = disp_y_pickup,
                       Q_x = Q_to_pickup_x, Q_y = Q_to_pickup_y,
                       x_ref_1 = x_ref_pickup, px_ref_1 = px_ref_pickup, x_ref_0 = x_ref_kicker, px_ref_0 = px_ref_kicker,
                       y_ref_1 = y_ref_pickup, py_ref_1 = py_ref_pickup, y_ref_0 = y_ref_kicker, py_ref_0 = py_ref_kicker)
        self._transfer_from_pickup = xt.LinearTransferMatrix(alpha_x_1 = alpha_x_kicker, beta_x_1 = beta_x_kicker, disp_x_1 = disp_x_kicker,
                       alpha_x_0 = alpha_x_pickup, beta_x_0 = beta_x_pickup, disp_x_0 = disp_x_pickup,
                       alpha_y_1 = alpha_y_kicker, beta_y_1 = beta_y_kicker, disp_y_1 = disp_y_kicker,
                       alpha_y_0 = alpha_y_pickup, beta_y_0 = beta_y_pickup, disp_y_0 = disp_y_pickup,
                       Q_x = -Q_to_pickup_x, Q_y = -Q_to_pickup_y,
                       x_ref_0 = x_ref_pickup, px_ref_0 = px_ref_pickup, x_ref_1 = x_ref_kicker, px_ref_1 = px_ref_kicker,
                       y_ref_0 = y_ref_pickup, py_ref_0 = py_ref_pickup, y_ref_1 = y_ref_kicker, py_ref_1 = py_ref_kicker)

        self.tilts_x_deque = {}
        self.tilts_y_deque = {}
        self.tilts_age_deque = {}

        self._buffer_size = 3
        self._recv_buffer = np.ones(self._buffer_size,dtype=np.float64)
        self._send_buffers = {}

        self._statistics = ['mean_x','mean_y']

        ###self.counter = 0

    def _tilts_to_buffer(self,tilt_x,tilt_y,delay,key):
        if key not in self._send_buffers.keys():
            self._send_buffers[key] = np.zeros(self._buffer_size,dtype=np.float64)
        self._send_buffers[key][0] = tilt_x
        self._send_buffers[key][1] = tilt_y
        self._send_buffers[key][2] = delay

    def send_messages(self,beam, partners):
        if beam.number not in self._pending_requests.keys():
            self._pending_requests[beam.number] = {}
            self._last_requests_turn[beam.number] = {}
        self._transfer_to_pickup.track(beam)
        slice_set = beam.get_slices(self.slicer,statistics=self._statistics)
        self._transfer_from_pickup.track(beam)
        tilt_x,tilt_y = self.get_tilts(slice_set)
        key = self.get_message_key(beam,beam)
        self._tilts_to_buffer(tilt_x,tilt_y,beam.delay,key)
        if len(partners)>0:
            request_is_pending = False
            for key in self._pending_requests[beam.number].keys():
                if beam.period <= self._last_requests_turn[beam.number][key]:
                    request_is_pending = True
                    break
                else:
                    if not self._pending_requests[beam.number][key].Test():
                        request_is_pending = True
                        break
            if not request_is_pending:
                if not beam.is_real:
                    print('Cannot compute slices on fake beam')
                for partner in partners:
                    if beam.period > 0 or partner.delay > beam.delay: # on the first turn, only send message to partners that are behind
                        tag = self.get_message_tag(beam,partner)
                        key = self.get_message_key(beam,partner)
                        self._tilts_to_buffer(tilt_x,tilt_y,beam.delay,key)
                        self._pending_requests[beam.number][key] = self._comm.Issend(self._send_buffers[key],dest=partner.rank,tag=tag)
                        self._last_requests_turn[beam.number][key] = beam.period

    def messages_are_ready(self,beam, partners):
        for partner in partners:
            if beam.period > 0 or partner.delay < beam.delay: # on the first turn, expect only messages from bunches ahead
                if not self._comm.Iprobe(source=partner.rank, tag=self.get_message_tag(partner,beam)):
                    return False
        return True

    def track(self, beam, partners):
        revolution_time = (beam.circumference / (beam.beta * constants.c))
        if beam.number not in self.tilts_age_deque.keys():
            self.tilts_x_deque[beam.number] = deque([], maxlen=self.n_turns*(1+len(partners)))
            self.tilts_y_deque[beam.number] = deque([], maxlen=self.n_turns*(1+len(partners)))
            self.tilts_age_deque[beam.number] = deque([], maxlen=self.n_turns*(1+len(partners)))
        # delaying past signals
        for i in range(len(self.tilts_age_deque[beam.number])):
            self.tilts_age_deque[beam.number][i] += revolution_time
        # adding signal from other bunches
        for partner in partners:
            if beam.period > 0 or partner.delay < beam.delay: # on the first turn, get only messages from bunches ahead
                if partner.delay < beam.delay:
                    group_delay_in_seconds = self.group_delay*revolution_time # partner is ahead, the last passage was in the same turn
                else:
                    group_delay_in_seconds = (self.group_delay-1)*revolution_time # partner is behind, the last passage was in the previous turn
                tag = self.get_message_tag(partner,beam)
                self._comm.Recv(self._recv_buffer,source=partner.rank,tag=tag)
                self.tilts_x_deque[beam.number].appendleft(self._recv_buffer[0])
                self.tilts_y_deque[beam.number].appendleft(self._recv_buffer[1])
                self.tilts_age_deque[beam.number].appendleft(beam.delay-self._recv_buffer[2]-group_delay_in_seconds) # The age is set in the future corresponding to the group delay. 
        # adding my own signal
        group_delay_in_seconds = self.group_delay*revolution_time 
        key = self.get_message_key(beam,beam)
        self.tilts_x_deque[beam.number].appendleft(self._send_buffers[key][0])
        self.tilts_y_deque[beam.number].appendleft(self._send_buffers[key][1])
        self.tilts_age_deque[beam.number].appendleft(-group_delay_in_seconds)

        if self.linear_kick:
            sine = self.crab_k*beam.zeta
            cosine = np.ones_like(beam.zeta)
        else:
            sine = np.sin(self.crab_k*beam.zeta)
            cosine = np.cos(self.crab_k*beam.zeta)

        for i in range(len(self.tilts_x_deque[beam.number])):
            scale = np.exp(-self.crab_omega*np.abs(self.tilts_age_deque[beam.number][i])/(2*self.crab_Q))
            if self.gain_x != 0.0 and scale != 0.0:
                #print('PyPLINEDCrabCavityAmplitudeFeedback',beam.name,beam.period,self.gain_x,self.tilts_x_deque[beam.number][i],scale)
                kickAmpl = self.gain_x*self.tilts_x_deque[beam.number][i]*scale # correspond to qV/E
                beam.px -= kickAmpl*sine # https://twiki.cern.ch/twiki/bin/viewauth/LHCAtHome/SixTrackDocRFElements
                if self.energy_kick:
                    beam.delta -= kickAmpl*beam.x*self.crab_k*cosine
            if self.gain_y != 0.0 and scale != 0.0:
                #print('PyPLINEDCrabCavityAmplitudeFeedback',beam.name,beam.period,self.gain_y,self.tilts_y_deque[beam.number][i],scale)
                kickAmpl = self.gain_y*self.tilts_y_deque[beam.number][i]*scale
                beam.py -= kickAmpl*sine # https://twiki.cern.ch/twiki/bin/viewauth/LHCAtHome/SixTrackDocRFElements
                if self.energy_kick:
                    beam.delta -= kickAmpl*beam.y*self.crab_k*cosine

    def get_tilts(self,slice_set):
        phases = self.demod_k*slice_set.z_centers
        cosine = np.cos(phases)
        sine = np.sin(phases)
        Isum = np.sum(cosine*slice_set.n_macroparticles_per_slice)
        Qsum = np.sum(sine*slice_set.n_macroparticles_per_slice)
        signal_x = slice_set.mean_x*slice_set.n_macroparticles_per_slice
        Idiff_x = np.sum(cosine*signal_x)
        Qdiff_x = np.sum(sine*signal_x)
        signal_y = slice_set.mean_y*slice_set.n_macroparticles_per_slice
        Idiff_y = np.sum(cosine*signal_y)
        Qdiff_y = np.sum(sine*signal_y)
        norm = Isum**2+Qsum**2
        tilt_x = self.transfer_x*(Qdiff_x*Isum-Idiff_x*Qsum)/norm # tilt prediction after the group delay
        tilt_y = self.transfer_y*(Qdiff_y*Isum-Idiff_y*Qsum)/norm
        return tilt_x,tilt_y

