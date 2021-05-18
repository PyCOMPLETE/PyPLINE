from mpi4py import MPI
from abc import abstractmethod

#import PyHEADTAIL.particles 
from PyHEADTAIL.particles.particles import Particles

class PyPLINEDParticlesID:
    def __init__(self,name,number,rank):
        self.name = name
        self.number = number
        self.rank = rank

# The bunches in the same core must have a different number. The name is there for convenience.
class PyPLINEDParticles(Particles):

    def __init__(self,name,rank,number,*args, **kwargs):
        self._comm = MPI.COMM_WORLD
        self.isReal = self._comm.Get_rank() == rank
        if self.isReal:
            super().__init__(*args, **kwargs)
        self._pipeline = []
        self._partnerIDs = []
        self._positionInPipeLine = 0
        self.ID = PyPLINEDParticlesID(name,number,rank)
        self.period = 0

        if self.isReal:
            print('I m bunch',name,'on rank',self._comm.Get_rank(),'and I m real')
        else:
            print('I m bunch',name,'on rank',self._comm.Get_rank(),'and I m fake')

    def addElementToPipeline(self,element,partnerIDs=[]):
        self._pipeline.append(element)
        self._partnerIDs.append(partnerIDs)

    def _increment(self):
        self._positionInPipeLine += 1
        if self._positionInPipeLine >= len(self._pipeline):
            self._positionInPipeLine = 0
            self.period += 1

    def step(self):
        if hasattr(self._pipeline[self._positionInPipeLine], 'isPyPLINED'):
            #print('Bunch',self.ID.name,'on rank',self.ID.rank,'turn',self.period,'at pipeline position',self._positionInPipeLine,self._pipeline[self._positionInPipeLine].name)
            self._pipeline[self._positionInPipeLine].sendMessages(self,self._partnerIDs[self._positionInPipeLine])
            if self._pipeline[self._positionInPipeLine].messagesAreReady(self.ID,self._partnerIDs[self._positionInPipeLine]):
                self._pipeline[self._positionInPipeLine].track(self,self._partnerIDs[self._positionInPipeLine])
                self._increment()
        else:
            #print('Bunch',self.ID.name,'on rank',self.ID.rank,'turn',self.period,'at pipeline position',self._positionInPipeLine)
            if hasattr(self._pipeline[self._positionInPipeLine],'track'):
                self._pipeline[self._positionInPipeLine].track(self)
            if hasattr(self._pipeline[self._positionInPipeLine],'dump'):
                self._pipeline[self._positionInPipeLine].dump(self)
            self._increment()

