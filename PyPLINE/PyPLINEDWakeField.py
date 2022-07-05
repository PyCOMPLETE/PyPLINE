import pickle,copy,time
from collections import deque
import numpy as np

from scipy import constants

import xfields as xf
from PyPLINE.PyPLINEDElement import PyPLINEDElement
from PyHEADTAIL.impedances.wakes import WakeField
from PyHEADTAIL.particles.slicing import SliceSet

class PyPLINEDWakeField(WakeField,PyPLINEDElement):

    def __init__(self,name,number,n_turns_wake,slicer,*wake_sources):
        PyPLINEDElement.__init__(self,name,number)
        WakeField.__init__(self,slicer,*wake_sources)

        self.n_turns_wake = n_turns_wake
        self.slice_set_deque = {}
        self.slice_set_age_deque = {}

        self._statistics = ['mean_x','mean_y']

        self._attributes_to_buffer = ['n_macroparticles_per_slice']
        self._attributes_to_buffer.extend(self._statistics)
        self._buffer_size = self.slicer.n_slices*len(self._attributes_to_buffer)+2
        self._recv_buffer = np.zeros(self._buffer_size,dtype=np.float64)
        self._send_buffers = {}

    def _slice_set_to_buffer(self,slice_set,delay,key):
        if key not in self._send_buffers.keys():
            self._send_buffers[key] = np.zeros(self._buffer_size,dtype=np.float64)
        k = 0
        for i in range(len(self._attributes_to_buffer)):
            self._send_buffers[key][k:k+len(getattr(slice_set,self._attributes_to_buffer[i]))] = getattr(slice_set,self._attributes_to_buffer[i])
            k += self.slicer.n_slices
        self._send_buffers[key][-2] = slice_set.beta
        self._send_buffers[key][-1] = delay

    def _slice_set_from_buffer(self,slice_set0):
        slice_set_kwargs = {'z_bins':slice_set0.z_bins,'mode':slice_set0.mode}
        k = 0
        arrays_recvd = {}
        for i in range(len(self._attributes_to_buffer)):
            arrays_recvd[self._attributes_to_buffer[i]] = np.copy(self._recv_buffer[k:k+self.slicer.n_slices])
            k += self.slicer.n_slices
        slice_set_kwargs['n_macroparticles_per_slice'] = arrays_recvd['n_macroparticles_per_slice'].astype(int)
        slice_set_kwargs['slice_index_of_particle'] = None
        slice_set_kwargs['beam_parameters'] = {'beta':self._recv_buffer[-2]}
        slice_set = SliceSet(**slice_set_kwargs)
        for statistic in self._statistics:
            setattr(slice_set,statistic,arrays_recvd[statistic])

        return slice_set,self._recv_buffer[-1]

    def send_messages(self,beam, partners):
        if beam.number not in self._pending_requests.keys():
            self._pending_requests[beam.number] = {}
            self._last_requests_turn[beam.number] = {}
        if len(partners)>0:
            request_is_pending = False
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
                slice_set = beam.get_slices(self.slicer,statistics=self._statistics)
                for partner in partners:
                    if beam.period > 0 or partner.delay > beam.delay: # on the first turn, only send message to partners that are behind
                        tag = self.get_message_tag(beam,partner)
                        key = self.get_message_key(beam,partner)
                        #print(beam.name,'sending slice set to',partner.name,'with tag',tag,flush=True)
                        self._slice_set_to_buffer(slice_set,beam.delay,key)
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
        if beam.number not in self.slice_set_deque.keys():
            self.slice_set_deque[beam.number] = deque([], maxlen=self.n_turns_wake*(1+len(partners)))
            self.slice_set_age_deque[beam.number] = deque([], maxlen=self.n_turns_wake*(1+len(partners)))
        for i in range(len(self.slice_set_age_deque[beam.number])):
            self.slice_set_age_deque[beam.number][i] += (beam.circumference / (beam.beta * constants.c))
        # retrieving my own slice set (it was already computed in 'sendMessages') 
        slice_set = beam.get_slices(self.slicer,statistics=['mean_x', 'mean_y'])
        # adding slice sets from other bunches
        for partner in partners:
            if beam.period > 0 or partner.delay < beam.delay: # on the first turn, get only messages from bunches ahead
                if partner.delay < beam.delay:
                    turn_delay = 0.0 # partner is ahead, the last passage was in the same turn
                else:
                    turn_delay = revolution_time  # partner is behind, the last passage was in the previous turn
                tag = self.get_message_tag(partner,beam)
                #print(beam.name,'receiving slice set from',partner.name,'with tag',tag,flush=True)
                self._comm.Recv(self._recv_buffer,source=partner.rank,tag=tag)
                partner_slice_set,partner_delay = self._slice_set_from_buffer(slice_set)
                self.slice_set_deque[beam.number].appendleft(partner_slice_set)
                self.slice_set_age_deque[beam.number].appendleft(beam.delay-partner_delay+turn_delay)
        # adding my own slice set (note: my own slice set needs to be the first in the list since PyHEADTAIL uses its 'slice_index_of_particle')
        self.slice_set_deque[beam.number].appendleft(slice_set)
        self.slice_set_age_deque[beam.number].appendleft(0.0)
        # applying all kicks
        for kick in self.wake_kicks:
            kick.apply(beam, self.slice_set_deque[beam.number], self.slice_set_age_deque[beam.number])

