import sys
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Logger(object):
    def __init__(self, filename: str):
        self.f = open(filename, mode='w')

    def write(self, message):
        self.f.write(message)
        print(message)  
    
    def close(self):
        self.f.close()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

