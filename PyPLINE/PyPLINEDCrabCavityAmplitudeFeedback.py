import pickle,copy
from collections import deque
import numpy as np

from scipy import constants

import xfields as xf
from PyPLINE.PyPLINEDElement import PyPLINEDElement

class PyPLINEDCrabCavityAmplitudeFeedback(PyPLINEDElement):

    def __init__(self,name,number,slicer,n_turns=1,gain_x=0.0,gain_y=0.0,crab_Q=1,crab_frequency = 400.6E6, demodulation_frequency=400.6E6):
        PyPLINEDElement.__init__(self,name,number)

        self.n_turns = n_turns
        self.slicer = slicer
        self.crab_omega = 2.0*np.pi*crab_frequency
        self.crab_k = self.crab_omega/constants.c
        self.demod_k = 2.0*np.pi*demodulation_frequency/constants.c
        self.gain_x = gain_x
        self.gain_y = gain_y
        self.crab_Q = crab_Q

        self.tilts_x_deque = {}
        self.tilts_y_deque = {}
        self.tilts_age_deque = {}

        self._buffer_size = 3
        self._recv_buffer = np.ones(self._buffer_size,dtype=np.float64)
        self._send_buffers = {}

    def _slice_set_to_buffer(self,slice_set,delay,key):
        if key not in self._send_buffers.keys():
            self._send_buffers[key] = np.zeros(self._buffer_size,dtype=np.float64)
        tilt_x,tilt_y = self.get_tilts(slice_set)
        self._send_buffers[key][0] = tilt_x
        self._send_buffers[key][1] = tilt_y
        self._send_buffers[key][2] = delay

        return slice_set,self._recv_buffer[-1]

    def send_messages(self,beam, partners_IDs):
        if len(partners_IDs)>0:
            key = self.get_message_key(beam.ID,partners_IDs[0])
            request_is_pending = False
            if key in self._pending_requests.keys():
                if beam.period == self._pending_requests[key]:
                    #print('There is a pending request with key',key,'at turn',beam.period,flush=True)
                    request_is_pending = True
            if not request_is_pending:
                if not beam.is_real:
                    print('Cannot compute slices on fake beam')
                slice_set = beam.get_slices(self.slicer,statistics=self._statistics)
                self._pending_requests[key] = beam.period
                for partner_ID in partners_IDs:
                    tag = self.get_message_tag(beam.ID,partner_ID)
                    key = self.get_message_key(beam.ID,partner_ID)
                    #print(beam.ID.name,'sending slice set to',partner_ID.name,'with tag',tag,flush=True)
                    self._slice_set_to_buffer(slice_set,beam.delay,key)
                    self._comm.Isend(self._send_buffers[key],dest=partner_ID.rank,tag=tag)

    def messages_are_ready(self,beam_ID, partners_IDs):
        for partner_ID in partners_IDs:
            if not self._comm.Iprobe(source=partner_ID.rank, tag=self.get_message_tag(partner_ID,beam_ID)):
                return False
        return True

    def track(self, beam, partners_IDs):
        if beam.ID.number not in self.slice_set_deque.keys():
            self.tilts_x_deque[beam.ID.number] = deque([], maxlen=self.n_turns*(1+len(partners_IDs)))
            self.tilts_y_deque[beam.ID.number] = deque([], maxlen=self.n_turns*(1+len(partners_IDs)))
            self.tilts_age_deque[beam.ID.number] = deque([], maxlen=self.n_turns*(1+len(partners_IDs)))
        # delaying past signals
        for i in range(len(self.slice_set_age_deque[beam.ID.number])):
            self.tilts_age_deque[beam.ID.number][i] += (beam.circumference / (beam.beta * constants.c))
        # adding signal from other bunches
        for partner_ID in partners_IDs:
            tag = self.get_message_tag(partner_ID,beam.ID)
            #print(beam.ID.name,'receiving slice set from',partner_ID.name,'with tag',tag,flush=True)
            self._comm.Recv(self._recv_buffer,source=partner_ID.rank,tag=tag)
            self.tilts_x_deque[beam.ID.number].appendleft(self._recv_buffer[0])
            self.tilts_y_deque[beam.ID.number].appendleft(self._recv_buffer[1])
            self.tilts_age_deque[beam.ID.number].appendleft(beam.delay-self._recv_buffer[2])
        # adding my own signal
        key = self.get_message_key(beam.ID,partners_IDs[0])
        self.tilts_x_deque[beam.ID.number].appendleft(self._send_buffers[key][0])
        self.tilts_y_deque[beam.ID.number].appendleft(self._send_buffers[key][1])
        self.tilts_age_deque[beam.ID.number].appendleft(0.0)

        slice_set = beam.get_slices(self.slicer,statistics=[])
        sine = np.sin(self.crab_k*slice_set.z_centers)
        cosine = np.cos(self.crab_k*slice_set.z_centers)

        for i in range(len(self.tilts_x_deque[beam.ID.number])):
            scale = np.exp(-self.crab_omega*self.tilts_age_deque[beam.ID.number][i]/(2*self.crab_Q))
            if self.gain_x != 0.0:
                kickAmpl = self.gain_x*self.tilts_x_deque[beam.ID.number][i]*scale # correspond to qV/E
                beam.px -= kickAmpl*sine
                beam.delta -= kickAmpl*beam.x*self.crab_k*cosine
            if self.gain_y != 0.0:
                kickAmpl = self.gain_y*self.tilts_y_deque[beam.ID.number][i]*scale
                beam.py -= kickAmpl*sine
                beam.delta -= kickAmpl*beam.y*self.crab_k*cosine

    def get_tilts(self,slice_set):
        phases = self.demod_k*slice_set.z_centers
        cosine = np.cos(phases)
        sine = np.sin(phases)
        Isum = np.sum(cosine*slice_set.n_macroparticles_per_slice)
        Qsum = np.sum(sine*slice_set.n_macroparticles_per_slice)
        signal_X = slice_set.mean_px*slice_set.n_macroparticles_per_slice
        Idiff_X = np.sum(cosine*signalX)
        Qdiff_X = np.sum(sine*signalX)
        signal_Y = slice_set.mean_py*slice_set.n_macroparticles_per_slice
        Idiff_Y = np.sum(cosine*signal_Y)
        Qdiff_Y = np.sum(sine*signal_Y)    
        norm = Isum**2+Qsum**2
        tilt_x = (Qdiff_X*Isum-Idiff_X*Qsum)/norm
        tilt_y = (Qdiff_Y*Isum-Idiff_Y*Qsum)/norm
        return tilt_x,tilt_y

