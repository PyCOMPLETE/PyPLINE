from mpi4py import MPI
from abc import abstractmethod

from PyHEADTAIL.general.element import Element

# Each element must have a different number. The name is there for convenience.
class PyPLINEDElement(Element):

    def __init__(self,name,number,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.isPyPLINED = True
        self.name = name
        self.number = number
        self._comm = MPI.COMM_WORLD
        self.rank = self._comm.Get_rank()

        #TODO update according to input?
        self.maxNumberOfElement = 1000
        self.maxNumberOfBunchesPerCore = 100

        self._pendingRequests = {}

    def getMessageTag(self,senderID,recieverID):
        return self.number+self.maxNumberOfElement*senderID.number+self.maxNumberOfElement*self.maxNumberOfBunchesPerCore*recieverID.number

    def getMessageKey(self,senderID,recieverID):
        return f'{senderID.number}_{recieverID.number}'
    
    @abstractmethod
    def sendMessages(self, beam, partnerIDs):
        '''
        Attempts to send a non-blocking message to partners. Does nothing is the message was aleady send
        '''
        pass

    @abstractmethod
    def messagesAreReady(self, beam, partnerIDs):
        '''
        check whether the partners have send messages.
        '''
        pass

    @abstractmethod
    def track(self, beam, partnerIDs):
        '''
        Collect messages from partners (blocking) and perform tracking of beam through this Element.
        '''
        pass
