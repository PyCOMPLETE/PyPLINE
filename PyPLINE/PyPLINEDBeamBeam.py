import numpy as np

import xfields as xf
from PyPLINE.PyPLINEDElement import PyPLINEDElement

class PyPLINEDBeamBeam(PyPLINEDElement):

    def __init__(self,name,number,_context=None,_buffer=None,_offset=None,min_sigma_diff=1e-10):
        PyPLINEDElement.__init__(self,name,number)
        self.tracker = xf.BeamBeamBiGaussian2D(_context=_context,_buffer=_buffer,_offset=_offset,min_sigma_diff=min_sigma_diff,beta0=0.9)

    def setQ0(self,q0):
        self.tracker.update(q0=q0)

    def setBeta0(self,beta0):
        self.tracker.update(beta0=beta0)

    def send_messages(self,beam, partners_IDs):
        tag = self.get_message_tag(beam.ID,partners_IDs[0])
        key = self.get_message_key(beam.ID,partners_IDs[0])
        request_is_pending = False
        if key in self._pending_requests.keys():
            if beam.period == self._pending_requests[key]:
                #print('There is a pending request with key',key,'at turn',beam.period)
                request_is_pending = True
        if not request_is_pending:
            #print(beam.ID.name,'sending avg to',partners_IDs[0].name,'with tag',tag)
            if not beam.is_real:
                print('Cannot compute avg on fake beam')
            mean_x, sigma_x = xf.mean_and_std(beam.x)
            mean_y, sigma_y = xf.mean_and_std(beam.y)
            params = np.array([mean_x,mean_y,sigma_x,sigma_y,beam.num_particles])
            self._comm.Isend(params,dest=partners_IDs[0].rank,tag=tag)
            self._pending_requests[key] = beam.period

    def messages_are_ready(self,beam_ID, partners_IDs):
        return self._comm.Iprobe(source=partners_IDs[0].rank, tag=self.get_message_tag(partners_IDs[0],beam_ID))

    def track(self, beam, partners_IDs):
        tag = self.get_message_tag(partners_IDs[0],beam.ID)
        params =  np.empty(5, dtype=np.float64)
        self._comm.Recv(params,source=partners_IDs[0].rank,tag=tag)
        #print(beam.ID.name,'revieved avg from',partners_IDs[0].name,'with tag',tag,params)
        self.tracker.update(mean_x=params[0],mean_y=params[1],sigma_x=params[2],sigma_y=params[3],n_particles=params[4])
        self.tracker.track(beam)
    

