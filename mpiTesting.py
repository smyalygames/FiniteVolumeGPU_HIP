# -*- coding: utf-8 -*-

"""
This python module implements MPI simulations for benchmarking

Copyright (C) 2018  SINTEF ICT

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import gc
import time
import json
import logging
import os

#GPU-aware MPI
"""
from os import environ
if environ.get("MPICH_GPU_SUPPORT_ENABLED", False):
    from ctypes import CDLL, RTLD_GLOBAL
    CDLL(f"{environ.get('CRAY_MPICH_ROOTDIR')}/gtl/lib/libmpi_gtl_hsa.so", mode=RTLD_GLOBAL)
"""

# MPI
from mpi4py import MPI

# CUDA
#import pycuda.driver as cuda
from hip import hip,hiprtc

# Simulator engine etc
from GPUSimulators import MPISimulator, Common, CudaContext
from GPUSimulators import EE2D_KP07_dimsplit
from GPUSimulators.helpers import InitialConditions as IC
from GPUSimulators.Simulator import BoundaryCondition as BC

import argparse
parser = argparse.ArgumentParser(description='Strong and weak scaling experiments.')
parser.add_argument('-nx', type=int, default=128)
parser.add_argument('-ny', type=int, default=128)
parser.add_argument('--profile', action='store_true') # default: False

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    elif (
        isinstance(err, hiprtc.hiprtcResult)
        and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS
    ):
        raise RuntimeError(str(err))
    return result

args = parser.parse_args()

if(args.profile):
    profiling_data = {}
    # profiling: total run time
    t_total_start = time.time()
    t_init_start = time.time()


# Get MPI COMM to use
comm = MPI.COMM_WORLD


####
# Initialize logging
####
log_level_console = 20
log_level_file = 10
log_filename = 'mpi_' + str(comm.rank) + '.log'
logger = logging.getLogger('GPUSimulators')
logger.setLevel(min(log_level_console, log_level_file))

ch = logging.StreamHandler()
ch.setLevel(log_level_console)
logger.addHandler(ch)
logger.info("Console logger using level %s",
            logging.getLevelName(log_level_console))

fh = logging.FileHandler(log_filename)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s: %(message)s')
fh.setFormatter(formatter)
fh.setLevel(log_level_file)
logger.addHandler(fh)
logger.info("File logger using level %s to %s",
            logging.getLevelName(log_level_file), log_filename)


####
# Initialize MPI grid etc
####
logger.info("Creating MPI grid")
grid = MPISimulator.MPIGrid(MPI.COMM_WORLD)

"""
job_id = int(os.environ["SLURM_JOB_ID"])
allocated_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
allocated_gpus = int(os.environ["ROCR_VISIBLE_DEVICES"].count(",") + 1)

print("job_id:", job_id)
print("allocated_nodes", allocated_nodes)
print("allocated_gpus", allocated_gpus)
"""

####
# Initialize CUDA
####
#cuda.init(flags=0)
#logger.info("Initializing CUDA")
local_rank = grid.getLocalRank()
#num_cuda_devices = cuda.Device.count()
num_cuda_devices = hip_check(hip.hipGetDeviceCount())
cuda_device = local_rank % num_cuda_devices
logger.info("Process %s using CUDA device %s", str(local_rank), str(cuda_device))
cuda_context = CudaContext.CudaContext(device=cuda_device, autotuning=False)

####
# Set initial conditions
####

# DEBUGGING - setting random seed
np.random.seed(42)

logger.info("Generating initial conditions")
nx = args.nx
ny = args.ny

dt = 0.000001

gamma = 1.4
#save_times = np.linspace(0, 0.000009, 2)
#save_times = np.linspace(0, 0.000099, 11)
#save_times = np.linspace(0, 0.000099, 2)
save_times = np.linspace(0, 0.0000999, 2)
outfile = "mpi_out_" + str(MPI.COMM_WORLD.rank) + ".nc"
save_var_names = ['rho', 'rho_u', 'rho_v', 'E']

arguments = IC.genKelvinHelmholtz(nx, ny, gamma, grid=grid)
arguments['context'] = cuda_context
arguments['theta'] = 1.2
arguments['grid'] = grid

if(args.profile):
    t_init_end = time.time()
    t_init = t_init_end - t_init_start
    profiling_data["t_init"] = t_init

####
# Run simulation
####
logger.info("Running simulation")
# Helper function to create MPI simulator


def genSim(grid, **kwargs):
    local_sim = EE2D_KP07_dimsplit.EE2D_KP07_dimsplit(**kwargs)
    sim = MPISimulator.MPISimulator(local_sim, grid)
    return sim


(outfile, sim_runner_profiling_data, sim_profiling_data) = Common.runSimulation(
    genSim, arguments, outfile, save_times, save_var_names, dt)

if(args.profile):
    t_total_end = time.time()
    t_total = t_total_end - t_total_start
    profiling_data["t_total"] = t_total
    print("Total run time on rank " + str(MPI.COMM_WORLD.rank) + " is " + str(t_total) + " s")

# write profiling to json file
if(args.profile and MPI.COMM_WORLD.rank == 0):
    job_id = ""
    if "SLURM_JOB_ID" in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"])
        allocated_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        allocated_gpus = int(os.environ["ROCR_VISIBLE_DEVICES"].count(",") + 1)
     #   allocated_gpus = int(os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1)
        profiling_file = "MPI_jobid_" + \
            str(job_id) + "_" + str(allocated_nodes) + "_nodes_and_" + str(allocated_gpus) + "_GPUs_profiling.json"
        profiling_data["outfile"] = outfile
    else:
        profiling_file = "MPI_" + str(MPI.COMM_WORLD.size) + "_procs_and_" + str(num_cuda_devices) + "_GPUs_profiling.json"

    for stage in sim_runner_profiling_data["start"].keys():
        profiling_data[stage] = sim_runner_profiling_data["end"][stage] - sim_runner_profiling_data["start"][stage]

    for stage in sim_profiling_data["start"].keys():
        profiling_data[stage] = sim_profiling_data["end"][stage] - sim_profiling_data["start"][stage]

    profiling_data["nx"] = nx
    profiling_data["ny"] = ny
    profiling_data["dt"] = dt
    profiling_data["n_time_steps"] = sim_profiling_data["n_time_steps"]

    profiling_data["slurm_job_id"] = job_id
    profiling_data["n_cuda_devices"] = str(num_cuda_devices)
    profiling_data["n_processes"] = str(MPI.COMM_WORLD.size)
    profiling_data["git_hash"] = Common.getGitHash()
    profiling_data["git_status"] = Common.getGitStatus()

    with open(profiling_file, "w") as write_file:
        json.dump(profiling_data, write_file)

####
# Clean shutdown
####
sim = None
local_sim = None
cuda_context = None
arguments = None
logging.shutdown()
gc.collect()



####
# Print completion and exit
####
print("Completed!")
exit(0)
