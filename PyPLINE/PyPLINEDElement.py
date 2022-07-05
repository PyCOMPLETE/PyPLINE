from mpi4py import MPI
from abc import abstractmethod

# Each element must have a different number. The name is there for convenience.
class PyPLINEDElement:

    def __init__(self,name='',number=-1,*args, **kwargs):
        #print(f'New PyPLINED element {name}: {number}')
        self.is_PyPLINED = True
        self.name = name
        self.number = number
        self._comm = MPI.COMM_WORLD
        self.rank = self._comm.Get_rank()
        self._max_tag = self._comm.Get_attr(MPI.TAG_UB) # 8388607 with OpenMPI on HPC photon

        #TODO update according to input?
        self.max_n_elements = 1000
        self.max_bunch_per_rank = 100

        self._pending_requests = {}
        self._last_requests_turn = {}

    def get_message_tag(self,sender,reciever):
        tag = self.number+self.max_n_elements*sender.number+self.max_n_elements*self.max_bunch_per_rank*reciever.number
        if tag > self._max_tag:
            print(f'PyPLINEDElement WARNING {self.name}: MPI message tag {tag} is larger than max ({self._max_tag})')
        return tag

    def get_message_key(self,sender,reciever):
        #return f'{sender.name}_{reciever.name}'
        return f'{sender.number+self.max_bunch_per_rank*sender.rank}_{reciever.number+self.max_bunch_per_rank*reciever.rank}'
    
    @abstractmethod
    def send_messages(self, beam, partners):
        '''
        Attempts to send a non-blocking message to partners. Does nothing if the message was aleady sent with the same beam.period or if a message with the same tag was sent and is not received.
        '''
        pass

    @abstractmethod
    def messages_are_ready(self, beam, partners):
        '''
        check whether the partners have send messages.
        '''
        pass

    @abstractmethod
    def track(self, beam, partners):
        '''
        Collect messages from partners (blocking) and perform tracking of beam through this Element.
        '''
        pass
