# -*- coding: utf-8 -*-

"""
This python module implements the classical Lax-Friedrichs numerical
scheme for the shallow water equations

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

#from pycuda import gpuarray
from hip import hip,hiprtc





"""
Class that solves the SW equations using the Lax Friedrichs scheme
"""
class LxF (Simulator.BaseSimulator):

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
                 cfl_scale=0.9,
                 boundary_conditions=BoundaryCondition(),
                 block_width=16, block_height=16):
                 
        # Call super constructor
        super().__init__(context, 
            nx, ny, 
            dx, dy, 
            boundary_conditions,
            cfl_scale,
            1,
            block_width, block_height);
        self.g = np.float32(g) 

        # Get kernels
#        module = context.get_module("cuda/SWE2D_LxF.cu", 
#                                        defines={
#                                            'BLOCK_WIDTH': self.block_size[0], 
#                                            'BLOCK_HEIGHT': self.block_size[1]
#                                        }, 
#                                        compile_args={
#                                            'no_extern_c': True,
#                                            'options': ["--use_fast_math"], 
#                                        }, 
#                                        jit_compile_args={})
#        self.kernel = module.get_function("LxFKernel")
#        self.kernel.prepare("iiffffiPiPiPiPiPiPiP")

        kernel_file_path = os.path.abspath(os.path.join('cuda', 'SWE2D_LxF.cu.hip'))
        with open(kernel_file_path, 'r') as file:
            kernel_source = file.read()

        prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_source.encode(), b"LxFKernel", 0, [], []))

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName

        print(f"Compiling kernel .LxFKernel. for {arch}")

        cflags = [b"--offload-arch="+arch]
        err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError(log.decode())
        code_size = hip_check(hiprtc.hiprtcGetCodeSize(prog))
        code = bytearray(code_size)
        hip_check(hiprtc.hiprtcGetCode(prog, code))
        module = hip_check(hip.hipModuleLoadData(code))

        kernel = hip_check(hip.hipModuleGetFunction(module, b"LxFKernel"))

        #Create data by uploading to device
        self.u0 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        1, 1, 
                        [h0, hu0, hv0])
        self.u1 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        1, 1, 
                        [None, None, None])
        #self.cfl_data = gpuarray.GPUArray(self.grid_size, dtype=np.float32)
        data_h = np.empty(self.grid_size, dtype=np.float32)
        num_bytes = data_h.size * data_h.itemsize
        self.cfl_data = hip_check(hip.hipMalloc(num_bytes)).configure(
                 typestr="float32",shape=self.grid_size)

        dt_x = np.min(self.dx / (np.abs(hu0/h0) + np.sqrt(g*h0)))
        dt_y = np.min(self.dy / (np.abs(hv0/h0) + np.sqrt(g*h0)))
        dt = min(dt_x, dt_y)
        self.cfl_data.fill(dt, stream=self.stream)
        
    def substep(self, dt, step_number):
#        self.kernel.prepared_async_call(self.grid_size, self.block_size, self.stream, 
#                self.nx, self.ny, 
#                self.dx, self.dy, dt, 
#                self.g, 
#                self.boundary_conditions, 
#                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
#                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
#                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
#                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
#                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
#                self.u1[2].data.gpudata, self.u1[2].data.strides[0],
#                self.cfl_data.gpudata)

        #launch kernel
        hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *self.grid_size,
                    *self.block_size,
                    sharedMemBytes=0,
                    stream=self.stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(self.dt),
                        ctypes.c_float(self.g),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        self.cfl_data
                        )
                    )
                )

        hip_check(hip.hipDeviceSynchronize())

        self.u0, self.u1 = self.u1, self.u0
 
        hip_check(hip.hipModuleUnload(module))

        hip_check(hip.hipFree(cfl_data))

        print("--Launching Kernel .LxFKernel. is ok")

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
