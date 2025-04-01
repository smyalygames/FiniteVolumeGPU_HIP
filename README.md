# FiniteVolumeGPU

This is a HIP version of the [FiniteVolume code](https://github.com/babrodtk/FiniteVolumeGPU). It is a Python software package that implements several finite volume discretizations on Cartesian grids for the shallow water equations and the Euler equations. 

## Setup on LUMI-G
Here is a step-by-step guide on installing packages on LUMI-G

### Step 1: run conda-container
Installation via conda can be done as:
```shell
ml LUMI/24.03 partition/G
ml lumi-container-wrapper
```
```shell
conda-containerize new --prefix MyCondaEnv conda_environment_lumi.yml
```
where the file `conda_environment_lumi.yml` contains packages to be installed.

### Step 1 alternative: Convert to a singularity container with cotainr
Load the required modules first
```shell
ml CrayEnv
ml cotainr
```

Then build the Singularity/Apptainer container 
```shell
cotainr build my_container.sif --system=lumi-g --conda-env=conda_environment_lumi.yml
```

### Step 2: Modify Slurm Job file
Depending on your build method, update [`Jobs/job_lumi.slurm`](Jobs/job_lumi.slurm) if `conda-containerize` was used, or [`Jobs/job_apptainer_lumi.slurm`](Jobs/job_apptainer_lumi.slurm) if `containr` was used.

In the job file, the required changes is to match your project allocation,
and the directories of where the simulator and container is stored.

### Step 3: Run the Slurm Job
If `conda-containerize` was used for building:
```shell
sbatch Jobs/job_lumi.slurm
```

Otherwise, if `containr` was used for building:
```shell
sbatch Jobs/job_apptainer_lumi.slurm
```

### Troubleshooting

#### Error when running MPI.
```
`MPI startup(): PMI server not found. Please set I_MPI_PMI_LIBRARY variable if it is not a singleton case.
```
This can be resolved by exporting this:
```
export I_MPI_PMI_LIBRARY=/opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/lib/libmpi.so
```