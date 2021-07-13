import numpy as np

from xtrack.dress_element import MetaBeamElement
import xfields as xf
from PyPLINEDElement import PyPLINEDElement

class PyPLINEDBeamBeam(PyPLINEDElement):

    def __init__(self,name,number,_context=None,_buffer=None,_offset=None,min_sigma_diff=1e-10):
        PyPLINEDElement.__init__(self,name,number)
        self.tracker = xf.BeamBeamBiGaussian2D(_context=_context,_buffer=_buffer,_offset=_offset,min_sigma_diff=min_sigma_diff)

    def setQ0(self,q0):
        self.tracker.update(q0=q0)

    def setBeta0(self,beta0):
        self.tracker.update(beta0=beta0)

    def sendMessages(self,beam, partnerIDs):
        tag = self.getMessageTag(beam.ID,partnerIDs[0])
        key = self.getMessageKey(beam.ID,partnerIDs[0])
        requestIsPending = False
        if key in self._pendingRequests.keys():
            if beam.period == self._pendingRequests[key]:
                #print('There is a pending request with key',key,'at turn',beam.period)
                requestIsPending = True
        if not requestIsPending:
            #print(beam.ID.name,'sending avg to',partnerIDs[0].name,'with tag',tag)
            if not beam.isReal:
                print('Cannot compute avg on fake beam')
            mean_x, sigma_x = xf.mean_and_std(beam.x)
            mean_y, sigma_y = xf.mean_and_std(beam.y)
            params = np.array([mean_x,mean_y,sigma_x,sigma_y,beam.num_particles])
            self._comm.Isend(params,dest=partnerIDs[0].rank,tag=tag)
            self._pendingRequests[key] = beam.period

    def messagesAreReady(self,beamID, partnerIDs):
        return self._comm.Iprobe(source=partnerIDs[0].rank, tag=self.getMessageTag(partnerIDs[0],beamID))

    def track(self, beam, partnerIDs):
        tag = self.getMessageTag(partnerIDs[0],beam.ID)
        params =  np.empty(5, dtype=np.float64)
        self._comm.Irecv(params,source=partnerIDs[0].rank,tag=tag)
        #print(beam.ID.name,'revieved avg from',partnerIDs[0].name,'with tag',tag,params)
        self.tracker.update(mean_x=params[0],mean_y=params[1],sigma_x=params[2],sigma_y=params[3],n_particles=params[4])
        self.tracker.track(beam)
    

