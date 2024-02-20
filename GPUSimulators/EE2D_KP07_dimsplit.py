# -*- coding: utf-8 -*-

"""
This python module implements the 2nd order HLL flux

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

        
        
        
        
        


"""
Class that solves the SW equations using the Forward-Backward linear scheme
"""
class EE2D_KP07_dimsplit (BaseSimulator):

    """
    Initialization routine
    rho: Density
    rho_u: Momentum along x-axis
    rho_v: Momentum along y-axis
    E: energy
    nx: Number of cells along x-axis
    ny: Number of cells along y-axis
    dx: Grid cell spacing along x-axis
    dy: Grid cell spacing along y-axis
    dt: Size of each timestep 
    g: Gravitational constant
    gamma: Gas constant
    p: pressure
    """

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

    def __init__(self, 
                 context, 
                 rho, rho_u, rho_v, E, 
                 nx, ny, 
                 dx, dy,  
                 g, 
                 gamma, 
                 theta=1.3, 
                 cfl_scale=0.9,
                 boundary_conditions=BoundaryCondition(), 
                 block_width=16, block_height=8):
                 
        # Call super constructor
        super().__init__(context, 
            nx, ny, 
            dx, dy, 
            boundary_conditions,
            cfl_scale, 
            2, 
            block_width, block_height)
        self.g = np.float32(g)
        self.gamma = np.float32(gamma)
        self.theta = np.float32(theta) 

        #Get kernels
        #module = context.get_module("cuda/EE2D_KP07_dimsplit.cu", 
        #                                defines={
        #                                    'BLOCK_WIDTH': self.block_size[0], 
        #                                    'BLOCK_HEIGHT': self.block_size[1]
        #                                }, 
        #                                compile_args={
        #                                    'no_extern_c': True,
        #                                    'options': ["--use_fast_math"], 
        #                                }, 
        #                                jit_compile_args={})
        #self.kernel = module.get_function("KP07DimsplitKernel")
        #self.kernel.prepare("iiffffffiiPiPiPiPiPiPiPiPiPiiii")
        #
        kernel_file_path = os.path.abspath(os.path.join('cuda', 'EE2D_KP07_dimsplit.cu'))
        with open(kernel_file_path, 'r') as file:
            kernel_source = file.read()

        prog = hip_check(hiprtc.hiprtcCreateProgram(kernel_source.encode(), b"KP07DimsplitKernel", 0, [], []))    

        props = hip.hipDeviceProp_t()
        hip_check(hip.hipGetDeviceProperties(props,0))
        arch = props.gcnArchName

        print(f"Compiling kernel for {arch}")

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

        kernel = hip_check(hip.hipModuleGetFunction(module, b"KP07DimsplitKernel"))

        #Create data by uploading to device
        self.u0 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [rho, rho_u, rho_v, E])
        self.u1 = Common.ArakawaA2D(self.stream, 
                        nx, ny, 
                        2, 2, 
                        [None, None, None, None])
        #self.cfl_data = gpuarray.GPUArray(self.grid_size, dtype=np.float32)
        # init device array cfl_data
        data_h = np.empty(self.grid_size, dtype=np.float32)
        num_bytes = data_h.size * data_h.itemsize
        self.cfl_data = hip_check(hip.hipMalloc(num_bytes)).configure(
                 typestr="float32",shape=self.grid_size)

        dt_x = np.min(self.dx / (np.abs(rho_u/rho) + np.sqrt(gamma*rho)))
        dt_y = np.min(self.dy / (np.abs(rho_v/rho) + np.sqrt(gamma*rho)))
        self.dt = min(dt_x, dt_y)
        self.cfl_data.fill(self.dt, stream=self.stream)
                        
    
    def substep(self, dt, step_number, external=True, internal=True):
            self.substepDimsplit(0.5*dt, step_number, external, internal)
    
    def substepDimsplit(self, dt, substep, external, internal):
        if external and internal:
            #print("COMPLETE DOMAIN (dt=" + str(dt) + ")")

#            self.kernel.prepared_async_call(self.grid_size, self.block_size, self.stream, 
#                self.nx, self.ny, 
#                self.dx, self.dy, dt, 
#                self.g, 
#                self.gamma, 
#                self.theta, 
#                substep,
#                self.boundary_conditions, 
#                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
#                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
#                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
#                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
#                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
#                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
#                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
#                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
#                self.cfl_data.gpudata,
#                0, 0, 
#                self.nx, self.ny)
            
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
                        ctypes.c_float(self.gamma),
                        ctypes.c_float(self.theta), 
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),    
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),   
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u0[3].data), ctypes.c_float(self.u0[3].data.strides[0]),   
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),   
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        ctypes.c_float(self.u1[3].data), ctypes.c_float(self.u1[3].data.strides[0]),   
                        self.cfl_data, 
                        0, 0,
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny)   
                        )
                    )
                )

            hip_check(hip.hipDeviceSynchronize())
            hip_check(hip.hipModuleUnload(module))

            hip_check(hip.hipFree(cfl_data))

            print("--External & Internal: Launching Kernel is ok")

            return
        
        if external and not internal:
            ###################################
            # XXX: Corners are treated twice! #
            ###################################

            ns_grid_size = (self.grid_size[0], 1)

            # NORTH
            # (x0, y0) x (x1, y1)
            #  (0, ny-y_halo) x (nx, ny)
#            self.kernel.prepared_async_call(ns_grid_size, self.block_size, self.stream, 
#                self.nx, self.ny,
#                self.dx, self.dy, dt, 
#                self.g, 
#                self.gamma, 
#                self.theta, 
#                substep,
#                self.boundary_conditions, 
#                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
#                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
#                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
#                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
#                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
#                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
#                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
#                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
#                self.cfl_data.gpudata,
#                0, self.ny - int(self.u0[0].y_halo),
#                self.nx, self.ny)

            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *ns_grid_size,
                    *self.block_size,
                    sharedMemBytes=0,
                    stream=self.stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(self.dt),
                        ctypes.c_float(self.g),
                        ctypes.c_float(self.gamma),
                        ctypes.c_float(self.theta),
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u0[3].data), ctypes.c_float(self.u0[3].data.strides[0]),
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        ctypes.c_float(self.u1[3].data), ctypes.c_float(self.u1[3].data.strides[0]),
                        self.cfl_data,
                        0, ctypes.c_int(self.ny) - ctypes.c_int(self.u0[0].y_halo),
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny)
                        )
                    )
                )


            # SOUTH
            # (x0, y0) x (x1, y1)
            #   (0, 0) x (nx, y_halo)
#            self.kernel.prepared_async_call(ns_grid_size, self.block_size, self.stream, 
#                self.nx, self.ny,
#                self.dx, self.dy, dt, 
#                self.g, 
#                self.gamma, 
#                self.theta, 
#                substep,
#                self.boundary_conditions, 
#                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
#                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
#                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
#                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
#                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
#                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
#                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
#                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
#                self.cfl_data.gpudata,
#                0, 0,
#                self.nx, int(self.u0[0].y_halo))
 
            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *ns_grid_size,
                    *self.block_size,
                    sharedMemBytes=0,
                    stream=self.stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(self.dt),
                        ctypes.c_float(self.g),
                        ctypes.c_float(self.gamma),
                        ctypes.c_float(self.theta),
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u0[3].data), ctypes.c_float(self.u0[3].data.strides[0]),
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        ctypes.c_float(self.u1[3].data), ctypes.c_float(self.u1[3].data.strides[0]),
                        self.cfl_data,
                        0, 0,
                        ctypes.c_int(self.nx), ctypes.c_int(self.u0[0].y_halo)
                        )
                    )
                )


            we_grid_size = (1, self.grid_size[1])
            
            # WEST
            # (x0, y0) x (x1, y1)
            #  (0, 0) x (x_halo, ny)
#            self.kernel.prepared_async_call(we_grid_size, self.block_size, self.stream, 
#                self.nx, self.ny,
#                self.dx, self.dy, dt, 
#                self.g, 
#                self.gamma, 
#                self.theta, 
#                substep,
#                self.boundary_conditions, 
#                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
#                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
#                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
#                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
#                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
#                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
#                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
#                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
#                self.cfl_data.gpudata,
#                0, 0,
#                int(self.u0[0].x_halo), self.ny)

            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *we_grid_size,
                    *self.block_size,
                    sharedMemBytes=0,
                    stream=self.stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(self.dt),
                        ctypes.c_float(self.g),
                        ctypes.c_float(self.gamma),
                        ctypes.c_float(self.theta),
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u0[3].data), ctypes.c_float(self.u0[3].data.strides[0]),
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        ctypes.c_float(self.u1[3].data), ctypes.c_float(self.u1[3].data.strides[0]),
                        self.cfl_data,
                        0, 0,
                        ctypes.c_int(self.u0[0].x_halo), ctypes.c_int(self.ny)
                        )
                    )
                )


            # EAST
            # (x0, y0) x (x1, y1)
            #   (nx-x_halo, 0) x (nx, ny)
#            self.kernel.prepared_async_call(we_grid_size, self.block_size, self.stream, 
#                self.nx, self.ny,
#                self.dx, self.dy, dt, 
#                self.g, 
#                self.gamma, 
#                self.theta, 
#                substep,
#                self.boundary_conditions, 
#                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
#                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
#                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
#                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
#                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
#                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
#                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
#                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
#                self.cfl_data.gpudata,
#                self.nx - int(self.u0[0].x_halo), 0,
#                self.nx, self.ny)

            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *we_grid_size,
                    *self.block_size,
                    sharedMemBytes=0,
                    stream=self.stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(self.dt),
                        ctypes.c_float(self.g),
                        ctypes.c_float(self.gamma),
                        ctypes.c_float(self.theta),
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u0[3].data), ctypes.c_float(self.u0[3].data.strides[0]),
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        ctypes.c_float(self.u1[3].data), ctypes.c_float(self.u1[3].data.strides[0]),
                        self.cfl_data,
                        ctypes.c_int(self.nx) - ctypes.c_int(self.u0[0].x_halo), 0,
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny)
                        )
                    )
                )

            hip_check(hip.hipDeviceSynchronize())
            hip_check(hip.hipModuleUnload(module))

            hip_check(hip.hipFree(cfl_data))

            print("--External and not Internal: Launching Kernel is ok")

            return

        if internal and not external:
            
            # INTERNAL DOMAIN
            #         (x0, y0) x (x1, y1)
            # (x_halo, y_halo) x (nx - x_halo, ny - y_halo)
            self.kernel.prepared_async_call(self.grid_size, self.block_size, self.internal_stream, 
                self.nx, self.ny, 
                self.dx, self.dy, dt, 
                self.g, 
                self.gamma, 
                self.theta, 
                substep,
                self.boundary_conditions, 
                self.u0[0].data.gpudata, self.u0[0].data.strides[0], 
                self.u0[1].data.gpudata, self.u0[1].data.strides[0], 
                self.u0[2].data.gpudata, self.u0[2].data.strides[0], 
                self.u0[3].data.gpudata, self.u0[3].data.strides[0], 
                self.u1[0].data.gpudata, self.u1[0].data.strides[0], 
                self.u1[1].data.gpudata, self.u1[1].data.strides[0], 
                self.u1[2].data.gpudata, self.u1[2].data.strides[0], 
                self.u1[3].data.gpudata, self.u1[3].data.strides[0],
                self.cfl_data.gpudata,
                int(self.u0[0].x_halo), int(self.u0[0].y_halo),
                self.nx - int(self.u0[0].x_halo), self.ny - int(self.u0[0].y_halo))


            hip_check(
                hip.hipModuleLaunchKernel(
                    kernel,
                    *self.grid_size,
                    *self.block_size,
                    sharedMemBytes=0,
                    stream=self.internal_stream,
                    kernelParams=None,
                    extra=(     # pass kernel's arguments
                        ctypes.c_int(self.nx), ctypes.c_int(self.ny),
                        ctypes.c_float(self.dx), ctypes.c_float(self.dy), ctypes.c_float(self.dt),
                        ctypes.c_float(self.g),
                        ctypes.c_float(self.gamma),
                        ctypes.c_float(self.theta),
                        ctypes.c_int(substep),
                        ctypes.c_int(self.boundary_conditions),
                        ctypes.c_float(self.u0[0].data), ctypes.c_float(self.u0[0].data.strides[0]),
                        ctypes.c_float(self.u0[1].data), ctypes.c_float(self.u0[1].data.strides[0]),
                        ctypes.c_float(self.u0[2].data), ctypes.c_float(self.u0[2].data.strides[0]),
                        ctypes.c_float(self.u0[3].data), ctypes.c_float(self.u0[3].data.strides[0]),
                        ctypes.c_float(self.u1[0].data), ctypes.c_float(self.u1[0].data.strides[0]),
                        ctypes.c_float(self.u1[1].data), ctypes.c_float(self.u1[1].data.strides[0]),
                        ctypes.c_float(self.u1[2].data), ctypes.c_float(self.u1[2].data.strides[0]),
                        ctypes.c_float(self.u1[3].data), ctypes.c_float(self.u1[3].data.strides[0]),
                        self.cfl_data,
                        ctypes.c_int(self.u0[0].x_halo), ctypes.c_int(self.u0[0].y_halo),
                        ctypes.c_int(self.nx) - ctypes.c_int(self.u0[0].x_halo), ctypes.c_int(self.ny) - ctypes.c_int(self.u0[0].y_halo)
                        )
                    )
                )

            hip_check(hip.hipDeviceSynchronize())
            hip_check(hip.hipModuleUnload(module))

            hip_check(hip.hipFree(cfl_data))

            print("--Internal and not External: Launching Kernel is ok")
            return

    def swapBuffers(self):
        self.u0, self.u1 = self.u1, self.u0
        return
        
    def getOutput(self):
        return self.u0

    def check(self):
        self.u0.check()
        self.u1.check()
        return
        
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
