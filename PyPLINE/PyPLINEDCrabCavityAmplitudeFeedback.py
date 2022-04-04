import pickle,copy
from collections import deque
import numpy as np

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

    def _tilts_to_buffer(self,tilt_x,tilt_y,delay,key):
        if key not in self._send_buffers.keys():
            self._send_buffers[key] = np.zeros(self._buffer_size,dtype=np.float64)
        self._send_buffers[key][0] = tilt_x
        self._send_buffers[key][1] = tilt_y
        self._send_buffers[key][2] = delay

    def send_messages(self,beam, partners_IDs):
        self._transfer_to_pickup.track(beam)
        slice_set = beam.get_slices(self.slicer,statistics=self._statistics)
        self._transfer_from_pickup.track(beam)
        tilt_x,tilt_y = self.get_tilts(slice_set)
        key = self.get_message_key(beam.ID,beam.ID)
        self._tilts_to_buffer(tilt_x,tilt_y,beam.delay,key)
        if len(partners_IDs)>0:
            request_is_pending = False
            for key in self._pending_requests.keys():
                if not self._pending_requests[key].Test():
                    ###print(beam.ID.name,'turn',beam.period,'there is a pending request with key',key)
                    request_is_pending = True
                    break
            if not request_is_pending:
                if not beam.is_real:
                    print('Cannot compute slices on fake beam')
                for partner_ID in partners_IDs:
                    tag = self.get_message_tag(beam.ID,partner_ID)
                    key = self.get_message_key(beam.ID,partner_ID)
                    self._tilts_to_buffer(tilt_x,tilt_y,beam.delay,key)
                    ###print(beam.ID.name,'turn',beam.period,'sending tilts to',partner_ID.name,'with tag',tag,flush=True)
                    self._pending_requests[key] = self._comm.Issend(self._send_buffers[key],dest=partner_ID.rank,tag=tag)

    def messages_are_ready(self,beam_ID, partners_IDs):
        for partner_ID in partners_IDs:
            if not self._comm.Iprobe(source=partner_ID.rank, tag=self.get_message_tag(partner_ID,beam_ID)):
                ###print(beam_ID.name,'waiting on message from',partner_ID.name)
                return False
        return True

    def track(self, beam, partners_IDs):
        revolution_time = (beam.circumference / (beam.beta * constants.c))
        group_delay_in_seconds = self.group_delay*revolution_time
        if beam.ID.number not in self.tilts_age_deque.keys():
            self.tilts_x_deque[beam.ID.number] = deque([], maxlen=self.n_turns*(1+len(partners_IDs)))
            self.tilts_y_deque[beam.ID.number] = deque([], maxlen=self.n_turns*(1+len(partners_IDs)))
            self.tilts_age_deque[beam.ID.number] = deque([], maxlen=self.n_turns*(1+len(partners_IDs)))
        # delaying past signals
        for i in range(len(self.tilts_age_deque[beam.ID.number])):
            self.tilts_age_deque[beam.ID.number][i] += revolution_time
        # adding signal from other bunches
        for partner_ID in partners_IDs:
            tag = self.get_message_tag(partner_ID,beam.ID)
            ###print(beam.ID.name,'turn',beam.period,'waiting to receive tilts from',partner_ID.name,'with tag',tag,flush=True)
            self._comm.Recv(self._recv_buffer,source=partner_ID.rank,tag=tag)
            ###print(beam.ID.name,'turn',beam.period,'got tilts from',partner_ID.name,'with tag',tag,flush=True)
            self.tilts_x_deque[beam.ID.number].appendleft(self._recv_buffer[0])
            self.tilts_y_deque[beam.ID.number].appendleft(self._recv_buffer[1])
            self.tilts_age_deque[beam.ID.number].appendleft(beam.delay-self._recv_buffer[2]-group_delay_in_seconds) # The age is set in the future corresponding to the group delay. 
        # adding my own signal
        key = self.get_message_key(beam.ID,beam.ID)
        self.tilts_x_deque[beam.ID.number].appendleft(self._send_buffers[key][0])
        self.tilts_y_deque[beam.ID.number].appendleft(self._send_buffers[key][1])
        self.tilts_age_deque[beam.ID.number].appendleft(-group_delay_in_seconds)

        if self.linear_kick:
            sine = self.crab_k*beam.zeta
            cosine = np.ones_like(beam.zeta)
        else:
            sine = np.sin(self.crab_k*beam.zeta)
            cosine = np.cos(self.crab_k*beam.zeta)

        for i in range(len(self.tilts_x_deque[beam.ID.number])):
            scale = np.exp(-self.crab_omega*np.abs(self.tilts_age_deque[beam.ID.number][i])/(2*self.crab_Q))
            if self.gain_x != 0.0 and scale != 0.0:
                #print('PyPLINEDCrabCavityAmplitudeFeedback',beam.ID.name,beam.period,self.gain_x,self.tilts_x_deque[beam.ID.number][i],scale)
                kickAmpl = self.gain_x*self.tilts_x_deque[beam.ID.number][i]*scale # correspond to qV/E
                beam.px -= kickAmpl*sine # https://twiki.cern.ch/twiki/bin/viewauth/LHCAtHome/SixTrackDocRFElements
                if self.energy_kick:
                    beam.delta -= kickAmpl*beam.x*self.crab_k*cosine
            if self.gain_y != 0.0 and scale != 0.0:
                kickAmpl = self.gain_y*self.tilts_y_deque[beam.ID.number][i]*scale
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

