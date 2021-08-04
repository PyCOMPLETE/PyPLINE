#import time
from mpi4py import MPI
import xtrack as xt

class PyPLINEDParticlesID:
    def __init__(self,name,number,rank):
        self.name = name
        self.number = number
        self.rank = rank

# The bunches in the same core must have a different number. The name is there for convenience.
class PyPLINEDParticles(xt.Particles):

    def __init__(self,name,rank,number,delay=0.0,*args, **kwargs):
        super(PyPLINEDParticles, self).__init__(*args, **kwargs)
        self._comm = MPI.COMM_WORLD
        self.is_real = self._comm.Get_rank() == rank
        if self.is_real:
            super().__init__(*args, **kwargs)
        self._pipeline = []
        self._partners_IDs = []
        self.delay = delay
        self._position_in_pipeline = 0
        self.ID = PyPLINEDParticlesID(name,number,rank)
        self.period = 0
        #self.waitingTime = 0.0

        if self.is_real:
            print('I m bunch',name,'on rank',self._comm.Get_rank(),'and I m real')
        else:
            print('I m bunch',name,'on rank',self._comm.Get_rank(),'and I m fake')

    def add_element_to_pipeline(self,element,partners_IDs=[]):
        self._pipeline.append(element)
        self._partners_IDs.append(partners_IDs)

    def _increment(self):
        #print(f'Bunch {self.ID.name} waited {self.waitingTime}s',flush=True)
        #self.waitingTime = 0
        self._position_in_pipeline += 1
        if self._position_in_pipeline >= len(self._pipeline):
            self._position_in_pipeline = 0
            self.period += 1

    def step(self):
        #print('Bunch',self.ID.name,'on rank',self.ID.rank,'turn',self.period,'at pipeline position',self._position_in_pipeline,'is PyPLINED',hasattr(self._pipeline[self._position_in_pipeline], 'is_PyPLINED'))
        if hasattr(self._pipeline[self._position_in_pipeline], 'is_PyPLINED'):
            #print('Bunch',self.ID.name,'on rank',self.ID.rank,'turn',self.period,'at pipeline position',self._position_in_pipeline,self._pipeline[self._position_in_pipeline].name)
            self._pipeline[self._position_in_pipeline].send_messages(self,self._partners_IDs[self._position_in_pipeline])
            #time0 = time.time()
            if self._pipeline[self._position_in_pipeline].messages_are_ready(self.ID,self._partners_IDs[self._position_in_pipeline]):
                self._pipeline[self._position_in_pipeline].track(self,self._partners_IDs[self._position_in_pipeline])
                #print(f'Time for tracking {self.ID.name} through {self._pipeline[self._position_in_pipeline].name} {time.time()-time0}s',flush=True)
                self._increment()
            #else:
            #    self.waitingTime += time.time()-time0
        else:
            #time0 = time.time()
            #print('Bunch',self.ID.name,'on rank',self.ID.rank,'turn',self.period,'at pipeline position',self._position_in_pipeline)
            if hasattr(self._pipeline[self._position_in_pipeline],'track'):
                self._pipeline[self._position_in_pipeline].track(self)
            if hasattr(self._pipeline[self._position_in_pipeline],'dump'):
                self._pipeline[self._position_in_pipeline].dump(self)
            #print(f'Time for tracking {self.ID.name} through non-PyPLINED element {time.time()-time0}s',flush=True)
            self._increment()

