import numpy as np
from scipy import constants as cst
from PyPLINEDElement import PyPLINEDElement

class RoundBeamBeam4D(PyPLINEDElement):

    def sendMessages(self,beam, partnerIDs):
        tag = self.getMessageTag(beam.ID,partnerIDs[0])
        key = self.getMessageKey(beam.ID,partnerIDs[0])
        requestIsPending = False
        if key in self._pendingRequests.keys():
            print('There is a pending request with key',key)
            if not self._pendingRequests[key].Test():
                requestIsPending = True
        if not requestIsPending:
            print(beam.ID.name,'sending avg to',partnerIDs[0].name,'with tag',tag)
            if not beam.isReal:
                print('Cannot compute avg on fake beam')
            params = np.array([beam.mean_x(),beam.mean_y(),beam.sigma_x(),beam.sigma_y(),beam.intensity])
            self._pendingRequests[key] = self._comm.Isend(params,dest=partnerIDs[0].rank,tag=tag)

    def messagesAreReady(self,beamID, partnerIDs):
        return self._comm.Iprobe(source=partnerIDs[0].rank, tag=self.getMessageTag(partnerIDs[0],beamID))

    def track(self, beam, partnerIDs):
        tag = self.getMessageTag(partnerIDs[0],beam.ID)
        params =  np.empty(5, dtype=np.float64)
        self._comm.Irecv(params,source=partnerIDs[0].rank,tag=tag)
        print(beam.ID.name,'revieved avg from',partnerIDs[0].name,'with tag',tag,params)
        sigma = (params[2]+params[3])/2
        myX = beam.mean_x()
        myY = beam.mean_y()
        r2 = (params[0]-myX)**2 + (params[1]-myY)**2
        r0 = beam.charge**2/(4*np.pi*cst.epsilon_0*beam.mass*cst.c**2) #TODO lift assumption that both beams have the particle type
        common = -2*r0*params[4]*(1-np.exp(-0.5*r2/sigma**2))/beam.gamma #TODO lift assumption that both beams have the same gamma
        beam.xp += common*(params[0]-myX)/r2
        beam.yp += common*(params[1]-myY)/r2
