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

        #TODO update according to input?
        self.max_n_elements = 1000
        self.max_bunch_per_rank = 100

        self._pending_requests = {}

    def get_message_tag(self,sender_ID,reciever_ID):
        return self.number+self.max_n_elements*sender_ID.number+self.max_n_elements*self.max_bunch_per_rank*reciever_ID.number

    def get_message_key(self,sender_ID,reciever_ID):
        return f'{sender_ID.number}_{reciever_ID.number}'
    
    @abstractmethod
    def send_messages(self, beam, partners_IDs):
        '''
        Attempts to send a non-blocking message to partners. Does nothing is the message was aleady send
        '''
        pass

    @abstractmethod
    def messages_are_ready(self, beam_ID, partners_IDs):
        '''
        check whether the partners have send messages.
        '''
        pass

    @abstractmethod
    def track(self, beam, partners_IDs):
        '''
        Collect messages from partners (blocking) and perform tracking of beam through this Element.
        '''
        pass
