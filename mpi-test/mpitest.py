from mpi4py import MPI
from tqdm import tqdm
import time
import numpy as np
from datetime import datetime
import sys

OPERATION_TIME = 1
NUM_OPERATIONS = 10

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Check for file path argument
if len(sys.argv) != 2:
    if rank == 0:
        print("Usage: mpirun -n <num_processes> python mpitest.py <output_file_path>")
    sys.exit(1)

output_file_path = sys.argv[1]


def perform_operation():
    """
    Simulate a time-consuming operation.
    """
    time.sleep(OPERATION_TIME / (rank + 1))
    return np.random.rand()  # Simulate some computation result


best_result = -1

for i in tqdm(range(NUM_OPERATIONS), desc=f"Process {rank}", position=rank, leave=True):
    result = perform_operation()
    if result > best_result:
        best_result = result

all_results = comm.gather(best_result, root=0)  # Blocking

if rank == 0:
    global_best = max(all_results)
    best_rank = all_results.index(global_best)
    print(f"Global best result: {global_best:.4f} found by process {best_rank}")

    for i in range(size):
        if i == best_rank:
            comm.send(True, dest=i, tag=33)
        else:
            comm.send(False, dest=i, tag=33)

else:
    should_save = comm.recv(source=0, tag=33)  # Blocking
    if should_save:
        print(f"Process {rank} will save its model with result {best_result:.4f}")

if rank == 0 and best_rank == 0:
    print(f"Process {rank} will save its model with result {global_best:.4f}")
    with open(output_file_path, "w") as f:
        f.write(f"Best result: {best_result:.4f} from process {rank}\n")

elif rank != 0 and should_save:
    with open(output_file_path, "w") as f:
        f.write(f"Best result: {best_result:.4f} from process {rank}\n")
