from mpi4py import MPI
from tqdm import tqdm
import time

OPERATION_TIME = 1
NUM_OPERATIONS = 100

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def perform_operation():
    """
    Simulate a time-consuming operation.
    """
    time.sleep(OPERATION_TIME)


for i in tqdm(range(NUM_OPERATIONS), desc=f"Process {rank}", position=rank, leave=True):
    perform_operation()
