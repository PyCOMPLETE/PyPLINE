import pickle,copy
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
        buffer_size = self.slicer.n_slices*len(self._attributes_to_buffer)+2
        self._send_buffer = np.zeros(buffer_size,dtype=float)
        self._recv_buffer = np.zeros(buffer_size,dtype=float)

    def _slice_set_to_buffer(self,slice_set,delay):
        k = 0
        for i in range(len(self._attributes_to_buffer)):
            self._send_buffer[k:k+len(getattr(slice_set,self._attributes_to_buffer[i]))] = getattr(slice_set,self._attributes_to_buffer[i])
            k += self.slicer.n_slices
        self._send_buffer[-2] = slice_set.beta
        self._send_buffer[-1] = delay

        self.tmp = copy.deepcopy(slice_set.slice_index_of_particle)

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
                    #print(beam.ID.name,'sending slice set to',partner_ID.name,'with tag',tag,flush=True)
                    self._slice_set_to_buffer(slice_set,beam.delay)
                    self._comm.Isend(self._send_buffer,dest=partner_ID.rank,tag=tag)

    def messages_are_ready(self,beam_ID, partners_IDs):
        for partner_ID in partners_IDs:
            if not self._comm.Iprobe(source=partner_ID.rank, tag=self.get_message_tag(partner_ID,beam_ID)):
                return False
        return True

    def track(self, beam, partners_IDs):
        if beam.ID.number not in self.slice_set_deque.keys():
            self.slice_set_deque[beam.ID.number] = deque([], maxlen=self.n_turns_wake*(1+len(partners_IDs)))
            self.slice_set_age_deque[beam.ID.number] = deque([], maxlen=self.n_turns_wake*(1+len(partners_IDs)))
        # delaying past slice sets
        for i in range(len(self.slice_set_age_deque[beam.ID.number])):
            self.slice_set_age_deque[beam.ID.number][i] += (beam.circumference / (beam.beta * constants.c))
        # retrieving my own slice set (it was already computed in 'sendMessages') 
        slice_set = beam.get_slices(self.slicer,statistics=['mean_x', 'mean_y'])
        # adding slice sets from other bunches
        for partner_ID in partners_IDs:
            tag = self.get_message_tag(partner_ID,beam.ID)
            #print(beam.ID.name,'receiving slice set from',partner_ID.name,'with tag',tag,flush=True)
            self._comm.Recv(self._recv_buffer,source=partner_ID.rank,tag=tag)
            partner_slice_set,partner_delay = self._slice_set_from_buffer(slice_set)
            self.slice_set_deque[beam.ID.number].appendleft(partner_slice_set)
            self.slice_set_age_deque[beam.ID.number].appendleft(beam.delay-partner_delay)
        # adding my own slice set (note: my own slice set needs to be the first in the list since PyHEADTAIL uses its 'slice_index_of_particle')
        self.slice_set_deque[beam.ID.number].appendleft(slice_set)
        self.slice_set_age_deque[beam.ID.number].appendleft(0.0)
        # applying all kicks
        for kick in self.wake_kicks:
            kick.apply(beam, self.slice_set_deque[beam.ID.number], self.slice_set_age_deque[beam.ID.number])

