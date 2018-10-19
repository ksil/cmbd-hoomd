/*! \file CollectiveModeGPU.cuh
    \brief Declares GPU kernel code for Collective Mode Brownian Dynamics
*/

#ifndef __COLLECTIVE_MODE_GPU_CUH__
#define __COLLECTIVE_MODE_GPU_CUH__

#include "hoomd/ParticleData.cuh"
#include "hoomd/HOOMDMath.h"
#include <cublas_v2.h>

//! Kernel driver for the first part of the Brownian update called by TwoStepBDGPU
cudaError_t gpu_collective(unsigned int timestep,
                            unsigned int seed,
                            unsigned int wave_seed,
                            Scalar4* d_pos,
                            int3* d_image,
                            const BoxDim &box,
                            const unsigned int* d_index_array,
                            const unsigned int* d_tag,
                            const Scalar alpha,
                            const unsigned int N,
                            const unsigned int Nk,
                            const Scalar4* d_net_force,
                            const Scalar dt,
                            const unsigned int D,
                            const Scalar T,
                            Scalar* d_dft_cos,
                            Scalar* d_dft_sin,
                            Scalar* d_B_T_F_cos,
                            Scalar* d_B_T_F_sin,
                            const Scalar* d_ks_mat,
                            const Scalar* d_ks_norm_mat,
                            const Scalar* d_A_mat,
                            const Scalar* d_A_half_mat);

#endif //__COLLECTIVE_MODE_GPU_CUH__
