# -*- coding: utf-8 -*-

"""
This python module implements the FORCE flux
for the shallow water equations

Copyright (C) 2016  SINTEF ICT

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

#Import packages we need
from GPUSimulators import Simulator, Common
from GPUSimulators.Simulator import BaseSimulator, BoundaryCondition
import numpy as np
import ctypes

#from pycuda import gpuarray
from hip import hip,hiprtc
from hip import hipblas

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

"""
Class that solves the SW equations 
"""
class HLL2 (Simulator.BaseSimulator):

    """
    Initialization routine
    h0: Water depth incl ghost cells, (nx+1)*(ny+1) cells
    hu0: Initial momentum along x-axis incl ghost cells, (nx+1)*(ny+1) cells
    hv0: Initial momentum along y-axis incl ghost cells, (nx+1)*(ny+1) cells
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis (20 000 m)
    dy: Grid cell spacing along y-axis (20 000 m)
    dt: Size of each timestep (90 s)
    g: Gravitational accelleration (9.81 m/s^2)
    """

    def __init__(self, 
                 context, 
                 h0, hu0, hv0, 
                 nx, ny, 
                 dx, dy, 
                 g, 
                 theta=1.8,
                 cfl_scale=0.9,
                 boundary_conditions=BoundaryCondition(), 
                 block_width=16, block_height=16):
                 
        # Call super constructor
        super().__init__(context, 
            nx, ny, 
            dx, dy, 
            boundary_conditions,
            cfl_scale,
            2,
            block_width, block_height)
        self.g = np.float32(g) 
        self.theta = np.float32(theta)

        #Get cuda kernels
        """
        module = context.get_module("cuda/SWE2D_HLL2.cu",
                                        defines={
                                            'BLOCK_WIDTH': self.block_size[0], 
                                            'BLOCK_HEIGHT': self.block_size[1]
                                        }, 
                                        compile_args={
                                            'no_extern_c': True,
                                            'options': ["--use_fast_math"], 
                                        }, 
                                        jit_compile_args={})
        self.kernel = module.get_function("HLL2Kernel")
        self.kernel.prepare("iiffffiPiPiPiPiPiPiP")
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Specify the relative path to the "cuda" directory
        cuda_dir = os.path.join(current_dir, 'cuda')

        #kernel source
        kernel_file_path = os.path.abspath(os.path.join(cuda_dir, 'SWE2D_HLL2.cu.hip'))
        with open(kernel_file_path, 'r') as file:
            kernel_source = file.read()

        #headers
        #common.h
        header_file_path = os.path.abspath(os.path.join(cuda_dir, 'common.h'))
        with open(header_file_path, 'r') as file:
            header_common = file.read()

        #SWECommon.h
        header_file_path = os.path.abspath(os.path.join(cuda_dir, 'SWECommon.h'))
        with open(header_file_path, 'r') as file:
            header_EulerCommon = file.read()
        
        #limiters.h
        header_file_path = os.path.abspath(os.path.join(cuda_dir, 'limiters.h'))
        with open(header_file_path, 'r') as file:
            header_limiters = file.read()

        #hip.hiprtc.hiprtcCreateProgram(const char *src, const char *name, int numHeaders, headers, includeNames)
        prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_source.encode(), b"HLL2Kernel", 3, [header_common.encode(),header_EulerCommon.encode(),header_limiters.encode()], [b"common.h",b"SWECommon.h",b"limiters.h"]))

        # Check if the program is created successfully
        if prog is not None:
            print("--This is <SWE2D_HLL2.cu.hip>")
            print("--HIPRTC program created successfully")
            print()
        else:
            print("--Failed to create HIPRTC program")
            print("--I stop:", err)
            exit()

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName

        print(f"Compiling kernel .HLL2Kernel. for {arch}")

        cflags = [b"--offload-arch="+arch, b"-O2", b"-D BLOCK_WIDTH="+ str(self.block_size[0]).encode(), b"-D BLOCK_HEIGHT=" + str(self.block_size[1]).encode()]

        err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        # Check if the program is compiled successfully
        if err is not None:
            print("--Compilation:", err)
            print("--The program is compiled successfully")
        else:
            print("--Compilation:", err)
            print("--Failed to compile the program")
            print("--I stop:", err)

        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError(log.decode())

        code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(prog, code))

        #Load the code as a module
        self.module = hip_check(hip.hipModuleLoadData(code))

        #Get the device kernel named named "FORCEKernel"
        self.kernel = hip_check(hip.hipModuleGetFunction(self.module, b"HLL2Kernel"))

        print()
        print("--Get the device kernel *HLL2Kernel* is created successfully--")
        print("--kernel", self.kernel)
        print()

        #Create data by uploading to device
        self.u0 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [h0, hu0, hv0])
        self.u1 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [None, None, None])
        #self.cfl_data = gpuarray.GPUArray(self.grid_size, dtype=np.float32)

        dt_x = np.min(self.dx / (np.abs(hu0/h0) + np.sqrt(g*h0)))
        dt_y = np.min(self.dy / (np.abs(hv0/h0) + np.sqrt(g*h0)))
        dt = min(dt_x, dt_y)
        #in HIP, the "DeviceArray" object doesn't have a 'fill' attribute
        #self.cfl_data.fill(self.dt, stream=self.stream)
        grid_dim_x, grid_dim_y, grid_dim_z = self.grid_size

        data_h = np.zeros((grid_dim_x, grid_dim_y), dtype=np.float32)
        num_bytes = data_h.size * data_h.itemsize
        data_h.fill(self.dt)

        self.cfl_data = hip_check(hip.hipMalloc(num_bytes)).configure(
                 typestr="float32",shape=(grid_dim_x, grid_dim_y))

        hip_check(hip.hipMemcpyAsync(self.cfl_data,data_h,num_bytes,hip.hipMemcpyKind.hipMemcpyHostToDevice,self.stream))
        #sets the memory region pointed to by x_d to zero asynchronously
        #initiates the memset operation asynchronously
        #hip_check(hip.hipMemsetAsync(self.cfl_data,0,num_bytes,self.stream))

    def substep(self, dt, step_number):
        self.substepDimsplit(dt*0.5, step_number)

    def substepDimsplit(self, dt, substep):        
        #Cuda
        """
        self.kernel.prepared_async_call(self.grid_size, self.block_size, self.stream, 
                self.nx, self.ny, 
                self.dx, self.dy, dt, 
                self.g, 
                self.theta,
                substep,
                self.boundary_conditions, 
                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
                self.u1[2].data.gpudata, self.u1[2].data.strides[0],
                self.cfl_data.gpudata)
        self.u0, self.u1 = self.u1, self.u0
        """

        u00_strides0 = self.u0[0].data.shape[0]*np.float32().itemsize
        u01_strides0 = self.u0[1].data.shape[0]*np.float32().itemsize
        u02_strides0 = self.u0[2].data.shape[0]*np.float32().itemsize

        u10_strides0 = self.u1[0].data.shape[0]*np.float32().itemsize
        u11_strides0 = self.u1[1].data.shape[0]*np.float32().itemsize
        u12_strides0 = self.u1[2].data.shape[0]*np.float32().itemsize

        #launch kernel
        hip_check(
                hip.hipModuleLaunchKernel(
                    self.kernel,
                    *self.grid_size, #grid
                    *self.block_size, #block
                    sharedMemBytes=0, #65536,
                    stream=self.stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(dt),
                        ctypes.c_float(self.g),
                        ctypes.c_float(self.theta),
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        self.u0[0].data, ctypes.c_int(u00_strides0),
                        self.u0[1].data, ctypes.c_int(u01_strides0),
                        self.u0[2].data, ctypes.c_int(u02_strides0),
                        self.u1[0].data, ctypes.c_int(u10_strides0),
                        self.u1[1].data, ctypes.c_int(u11_strides0),
                        self.u1[2].data, ctypes.c_int(u12_strides0),
                        self.cfl_data,
                    )
                )
            )

        self.u0, self.u1 = self.u1, self.u0
            
        #print("--Launching Kernel .HLL2Kernel. is ok")

    def getOutput(self):
        return self.u0
        
    def check(self):
        self.u0.check()
        self.u1.check()
    
    # computing min with hipblas: the output is an index
    def min_hipblas(self, num_elements, cfl_data, stream):
        num_bytes = num_elements * np.dtype(np.float32).itemsize
        num_bytes_i = np.dtype(np.int32).itemsize
        indx_d = hip_check(hip.hipMalloc(num_bytes_i))
        indx_h = np.zeros(1, dtype=np.int32)
        x_temp = np.zeros(num_elements, dtype=np.float32)

        #print("--size.data:", cfl_data.size)
        handle = hip_check(hipblas.hipblasCreate())

        #hip_check(hipblas.hipblasGetStream(handle, stream))
        #"incx" [int] specifies the increment for the elements of x. incx must be > 0.
        hip_check(hipblas.hipblasIsamin(handle, num_elements, cfl_data, 1, indx_d))

        # destruction of handle
        hip_check(hipblas.hipblasDestroy(handle))

        # copy result (stored in indx_d) back to the host (store in indx_h)
        hip_check(hip.hipMemcpyAsync(indx_h,indx_d,num_bytes_i,hip.hipMemcpyKind.hipMemcpyDeviceToHost,stream))
        hip_check(hip.hipMemcpyAsync(x_temp,cfl_data,num_bytes,hip.hipMemcpyKind.hipMemcpyDeviceToHost,stream))
        #hip_check(hip.hipMemsetAsync(cfl_data,0,num_bytes,self.stream))
        hip_check(hip.hipStreamSynchronize(stream))

        min_value = x_temp.flatten()[indx_h[0]-1]

        # clean up
        hip_check(hip.hipStreamDestroy(stream))
        hip_check(hip.hipFree(cfl_data))
        return min_value

    def computeDt(self):
        #max_dt = gpuarray.min(self.cfl_data, stream=self.stream).get();
        max_dt = self.min_hipblas(self.cfl_data.size, self.cfl_data, self.stream)
        return max_dt*0.5
