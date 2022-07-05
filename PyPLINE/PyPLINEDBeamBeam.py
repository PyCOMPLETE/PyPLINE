import numpy as np

import xfields as xf
from PyPLINE.PyPLINEDElement import PyPLINEDElement

class PyPLINEDBeamBeam(PyPLINEDElement):

    def __init__(self,name,number,_context=None,_buffer=None,_offset=None,min_sigma_diff=1e-10):
        PyPLINEDElement.__init__(self,name,number)
        self.tracker = xf.BeamBeamBiGaussian2D(_context=_context,_buffer=_buffer,_offset=_offset,min_sigma_diff=min_sigma_diff,beta0=0.9)
        self._recv_buffer = np.zeros(5,dtype=np.float64)

    def set_q0(self,q0):
        self.tracker.update(q0=q0)

    def set_beta0(self,beta0):
        self.tracker.update(beta0=beta0)

    def send_messages(self,beam, partners):
        if beam.number not in self._pending_requests.keys():
            self._pending_requests[beam.number] = {}
            self._last_requests_turn[beam.number] = {}
        tag = self.get_message_tag(beam,partners[0])
        key = self.get_message_key(beam,partners[0])
        request_is_pending = False
        if key in self._pending_requests[beam.number].keys():
            if beam.period <= self._last_requests_turn[beam.number][key]:
                #print(self.name,': a request with key',key,'was already sent at turn',self._last_requests_turn[beam.number][key],'(present turn',beam.period,')',flush=True)
                request_is_pending = True
            else:
                #print(self.name,': testing for request with key',key,'at turn',beam.period,flush=True)
                if not self._pending_requests[beam.number][key].Test():
                    #print(self.name,': There is a pending request with key',key,'at turn',beam.period,flush=True)
                    request_is_pending = True
        if not request_is_pending:
            #print(beam.name,'sending avg to',partners[0].name,'with tag',tag)
            if not beam.is_real:
                print('Cannot compute avg on fake beam')
            if np.any(np.isnan(beam.x)):
                print(beam.name,'turn',beam.period,'nan in x in BeamBeam')
            mean_x, sigma_x = xf.mean_and_std(beam.x)
            mean_y, sigma_y = xf.mean_and_std(beam.y)
            params = np.array([mean_x,mean_y,sigma_x,sigma_y,beam.macroparticlenumber*beam.particlenumber_per_mp],dtype=np.float64)
            #print(self.name,':',beam.name,'turn',beam.period,'sending',params,' to',partners[0].name,'with tag',tag,flush=True)
            request = self._comm.Issend(params,dest=partners[0].rank,tag=tag)
            self._pending_requests[beam.number][key] = request
            self._last_requests_turn[beam.number][key] = beam.period

    def messages_are_ready(self,beam, partners):
        return self._comm.Iprobe(source=partners[0].rank, tag=self.get_message_tag(partners[0],beam))

    def track(self, beam, partners):
        tag = self.get_message_tag(partners[0],beam)
        self._comm.Recv(self._recv_buffer,source=partners[0].rank,tag=tag)
        #print(self.name,':',beam.name,'turn',beam.period,'recieved',self._recv_buffer,' from',partners[0].name,'with tag',tag,flush=True)
        self.tracker.update(mean_x=self._recv_buffer[0],mean_y=self._recv_buffer[1],sigma_x=self._recv_buffer[2],sigma_y=self._recv_buffer[3],n_particles=self._recv_buffer[4])
        #if np.any(np.isnan(beam.px)):
        #    print(beam.name,'turn',beam.period,'nan in px before track in beambeam')
        #else:
        #    print(beam.name,'turn',beam.period,'no nan in px before track in beambeam')
        self.tracker.track(beam)
        #if np.any(np.isnan(beam.px)):
        #    print(beam.name,'turn',beam.period,'nan in px after track in beambeam')
        #else:
        #    print(beam.name,'turn',beam.period,'no nan in px after track in beambeam')


    

