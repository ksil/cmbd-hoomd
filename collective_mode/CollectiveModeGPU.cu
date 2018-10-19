#include "CollectiveModeGPU.cuh"
#include "hoomd/VectorMath.h"
#include "hoomd/HOOMDMath.h"
#include <stdio.h>

#include "hoomd/Saru.h"
using namespace hoomd;

#include <assert.h>

// definitions to call the correct cublas functions depending on specified precision
#ifdef SINGLE_PRECISION

#define SINCOSPI(...) sincospif(__VA_ARGS__)

#else

#define SINCOSPI(...) sincospi(__VA_ARGS__)

#endif

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

extern "C" __global__
void print_matrix(Scalar* d_mat, int rows, int cols, bool row_major)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (row_major) printf("%f ", d_mat[cols*i + j]);
            else printf("%f ", d_mat[j*rows + i]);
        }
        printf("\n");
    }
}

/*! \file CollectiveModeGPU.cu
    \brief Defines GPU kernel code for Collective Mode Brownian Dynamics.
*/

// ======================================== KERNELS =========================================

/* adds a column-major displacement to the Scalar 4 array of positions

d_net_force
    HOOMD array of forces for each particle
d_B_T_F_cos and d_B_T_F_sin
    row-major Nk x 3 matrices containing the cosine and sine
    transforms of particle positions times the forces
d_A_mat
    a column-major 3x3 matrix containing the self part of the mobility
d_A_half_mat
    a column-major 3xe matrix containing the Cholesky decomposition of
    matrix A, the self-part
d_dft_cos and d_dft_sin
    row-major N x Nk matrices containing the cosine and sine
    transforms of particle positions
d_pos
    HOOMD array of positions
d_index_array
    HOOMD array of index mapping for particles
    (matters when integrating a subset of particles)
d_tag
    HOOMD array of particle tags
    necessary for generation of independent random numbers for different groups
d_image
    HOOMD array containing the index of the image in which 
    each particle resides
box
    HOOMD object that contains data on the box size
alpha
    factor that describes how much to excite collective modes
    alpha = 0 corresponds to typical Brownian dynamics
N
    number of particles
Nk
    number of excited wave vectors
timestep
    the current timestep
seed
    a random seed
    MUST BE DIFFERENT than the seed for the self part above
T
    temperature
dt
    time step

*/
extern "C" __global__
void add_to_pos(const Scalar4* d_net_force,
                const Scalar* d_B_T_F_cos,
                const Scalar* d_B_T_F_sin,
                const Scalar* d_A_mat,
                const Scalar* d_A_half_mat,
                const Scalar* d_dft_cos,
                const Scalar* d_dft_sin,
                Scalar4* d_pos,
                const unsigned int* d_index_array,
                const unsigned int* d_tag,
                int3* d_image,
                const BoxDim box,
                const Scalar alpha,
                const unsigned int N,
                const unsigned int Nk,
                const unsigned int timestep,
                const unsigned int seed,
                const Scalar T,
                const Scalar dt)
{
    extern __shared__ Scalar buf[];
    Scalar* A = buf;
    Scalar* A_half = &buf[9];
    Scalar* C_T  = &buf[18];
    Scalar* S_T = &buf[18 + 3*Nk];

    int idx = threadIdx.x;

    // copy A and B_T_F matrices to shared memory
    while (idx < max(3*Nk, 9))
    {
        if (idx < 9) {
            A[idx] = d_A_mat[idx];
            A_half[idx] = d_A_half_mat[idx];
        }

        C_T[idx] = d_B_T_F_cos[idx];
        S_T[idx] = d_B_T_F_sin[idx];

        idx += BLOCK_SIZE;
    }

    __syncthreads();

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        unsigned int p_idx = d_index_array[idx];
        Scalar4 p = d_pos[p_idx];
        Scalar4 F = d_net_force[p_idx];
        int3 image = d_image[p_idx];
        unsigned int ptag = d_tag[p_idx];

        // --------------- self part -------------------

        // dt * F*A
        p.x += dt*(F.x*A[0] + F.y*A[1] + F.z*A[2]);
        p.y += dt*(F.x*A[3] + F.y*A[4] + F.z*A[5]);
        p.z += dt*(F.x*A[6] + F.y*A[7] + F.z*A[8]);

        // Brownian part
        detail::Saru saru(ptag, timestep, seed);
        Scalar sx = saru.s<Scalar>(-1,1);
        Scalar sy = saru.s<Scalar>(-1,1);
        Scalar sz = saru.s<Scalar>(-1,1);

        Scalar coeff = fast::sqrt(Scalar(3.0)*Scalar(2.0)*T*dt);    // sqrt(3) because not Gaussian

        p.x += coeff*(sx*A_half[0]);                                // A_half is upper triangular
        p.y += coeff*(sx*A_half[3] + sy*A_half[4]);
        p.z += coeff*(sx*A_half[6] + sy*A_half[7] + sz*A_half[8]);

        // ------------ collective part ----------------
        
        Scalar c, s;
        for (int i = 0; i < Nk; i++) {
            c = d_dft_cos[Nk*idx + i];
            s = d_dft_sin[Nk*idx + i];

            p.x += c * C_T[3*i];
            p.x += s * S_T[3*i];
            p.y += c * C_T[3*i + 1];
            p.y += s * S_T[3*i + 1];
            p.z += c * C_T[3*i + 2];
            p.z += s * S_T[3*i + 2];
        }
        
        // wrap particles in case displacements pushed particles outside box
        box.wrap(p, image);

        // write out data
        d_pos[p_idx] = p;
        d_image[p_idx] = image;
    }
}

/*
Calculates the nonuniform Fourier transform for given k vectors

d_pos
    HOOMD array of positions
d_index_array
    HOOMD array of index mapping for particles
    (matters when integrating a subset of particles)
d_net_force
    HOOMD array of forces for each particle
d_B_T_F_cos and d_B_T_F_sin
    row-major Nk x 3 matrices containing the cosine and sine
    transforms of particle positions
d_ks_mat
    row-major Nk x 3 matrix containing the excited wave vectors
box
    HOOMD object that contains data on the box size
N
    number of particles
D
    number of dimensions
Nk
    number of excited wave vectors
dt
    time step
T
    temperature
alpha
    factor that describes how much to excite collective modes
    alpha = 0 corresponds to typical Brownian dynamics
timestep
    the current timestep
seed
    a random seed
    MUST BE DIFFERENT than the seed for the self part above

*/
extern "C" __global__
void calculate_dft_and_reduce(const Scalar4* d_pos,
                    const unsigned int* d_index_array,
                    const Scalar4* d_net_force,
                    Scalar* d_B_T_F_cos,
                    Scalar* d_B_T_F_sin,
                    Scalar* d_dft_cos,
                    Scalar* d_dft_sin,
                    const Scalar* d_ks_mat,
                    const BoxDim box,
                    const unsigned int N,
                    const unsigned int D,
                    const unsigned int Nk,
                    const Scalar dt,
                    const Scalar T,
                    const Scalar alpha,
                    const int timestep,
                    const unsigned int seed)
{
    extern __shared__ Scalar ks[];

    // copy d_ks_mat (row-major) to shared memory
    int idx = threadIdx.x;
    while (idx < Nk)
    {
        ks[3*idx] = d_ks_mat[3*idx];
        ks[3*idx + 1] = d_ks_mat[3*idx + 1];
        if (D == 3) ks[3*idx + 2] = d_ks_mat[3*idx + 2];
            else ks[3*idx + 2] = 0.0;

        idx += BLOCK_SIZE;
    }

    __syncthreads();

    // calculate discrete fourier transform
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    Scalar4 p, F;
    unsigned int p_idx;

    Scalar3 L = box.getL();

    if (idx < N)
    {
        p_idx = d_index_array[idx];
        p = d_pos[p_idx];
        F = d_net_force[p_idx];
    }
    else
    {
        p = make_scalar4(0.0, 0.0, 0.0, 0.0);
        F = make_scalar4(0.0, 0.0, 0.0, 0.0);
    }


    if (D == 2)
    {
        L.z = 1.0;
        p.z = 0.0;
        F.z = 0.0;
    }

    // calculate cos and sin transforms and do reduction
    Scalar dft_cos;
    Scalar dft_sin;
    Scalar cf_x, cf_y, cf_z, sf_x, sf_y, sf_z;

    for (int kidx = 0; kidx < Nk; kidx++)
    {
        // stagger the k vector each block is operating on
        int i = (blockIdx.x + kidx) % Nk;

        // argument is 2 * pi * dot(k, x/L)
        Scalar arg = 2*(ks[3*i]*p.x/L.x + ks[3*i+1]*p.y/L.y + ks[3*i+2]*p.z/L.z);
        SINCOSPI(arg, &dft_sin, &dft_cos);

        // save values to d_dft arrays
        if (idx < N)
        {
            d_dft_cos[Nk*idx + i] = dft_cos;
            d_dft_sin[Nk*idx + i] = dft_sin;
        }

        // calculate action on forces
        cf_x = dt*alpha/Nk * dft_cos * F.x;
        cf_y = dt*alpha/Nk * dft_cos * F.y;
        cf_z = dt*alpha/Nk * dft_cos * F.z;
        sf_x = dt*alpha/Nk * dft_sin * F.x;
        sf_y = dt*alpha/Nk * dft_sin * F.y;
        sf_z = dt*alpha/Nk * dft_sin * F.z;

        // warp reduce
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            cf_x += __shfl_down_sync(FULL_MASK, cf_x, offset);
            cf_y += __shfl_down_sync(FULL_MASK, cf_y, offset);
            cf_z += __shfl_down_sync(FULL_MASK, cf_z, offset);
            sf_x += __shfl_down_sync(FULL_MASK, sf_x, offset);
            sf_y += __shfl_down_sync(FULL_MASK, sf_y, offset);
            sf_z += __shfl_down_sync(FULL_MASK, sf_z, offset);
        }

        // device-wide atomic reduction
        if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
        {
            atomicAdd(&d_B_T_F_cos[3*i], cf_x);
            atomicAdd(&d_B_T_F_cos[3*i+1], cf_y);
            atomicAdd(&d_B_T_F_cos[3*i+2], cf_z);
            atomicAdd(&d_B_T_F_sin[3*i], sf_x);
            atomicAdd(&d_B_T_F_sin[3*i+1], sf_y);
            atomicAdd(&d_B_T_F_sin[3*i+2], sf_z);
        }

        // add random wave-space displacement
        if (idx < 6)
        {
            detail::Saru saru(6*i + idx, timestep, seed);
            Scalar rand_disp = fast::sqrt(Scalar(3.0)*Scalar(2.0)*T*dt*alpha/Nk) * saru.s<Scalar>(-1,1);

            switch(idx)
            {
                case 0:
                    atomicAdd(&d_B_T_F_cos[3*i], rand_disp);
                    break;
                case 1:
                    atomicAdd(&d_B_T_F_cos[3*i+1], rand_disp);
                    break;
                case 2:
                    atomicAdd(&d_B_T_F_cos[3*i+2], rand_disp);
                    break;
                case 3:
                    atomicAdd(&d_B_T_F_sin[3*i], -rand_disp);
                    break;
                case 4:
                    atomicAdd(&d_B_T_F_sin[3*i+1], -rand_disp);
                    break;
                case 5:
                    atomicAdd(&d_B_T_F_sin[3*i+2], -rand_disp);
                    break;
            }
        }
    }

}

/*
Takes the cos and sin transform matrices and performs the orthogonal wavespace
projection (i.e., multiplication by (I-kk) for each k)

d_B_T_F_cos and d_B_T_F_sin
    both row-major Nk x 3 matrices
    each row corresponds to a different wave vector
d_ks_norm_mat
    a row-major Nk x 3 matrix
    each row contains a normalized wave vector
Nk
    the number of wave vectors

Assumes kernel will be launched with 1 block consisting of # threads = Nk
*/
extern "C" __global__
void project_ks(Scalar* d_B_T_F_cos, Scalar* d_B_T_F_sin, const Scalar* d_ks_norm_mat, const unsigned int Nk)
{
    int idx = threadIdx.x;

    if (idx < Nk)
    {
        Scalar kx = d_ks_norm_mat[3*idx];
        Scalar ky = d_ks_norm_mat[3*idx + 1];
        Scalar kz = d_ks_norm_mat[3*idx + 2];

        Scalar cx = d_B_T_F_cos[3*idx];
        Scalar cy = d_B_T_F_cos[3*idx + 1];
        Scalar cz = d_B_T_F_cos[3*idx + 2];

        Scalar sx = d_B_T_F_sin[3*idx];
        Scalar sy = d_B_T_F_sin[3*idx + 1];
        Scalar sz = d_B_T_F_sin[3*idx + 2];

        // do dot product
        Scalar dotc = kx*cx + ky*cy + kz*cz;
        Scalar dots = kx*sx + ky*sy + kz*sz;

        // I - kk
        d_B_T_F_cos[3*idx] -= dotc * kx;
        d_B_T_F_cos[3*idx + 1] -= dotc * ky;
        d_B_T_F_cos[3*idx + 2] -= dotc * kz;
        d_B_T_F_sin[3*idx] -= dots * kx;
        d_B_T_F_sin[3*idx + 1] -= dots * ky;
        d_B_T_F_sin[3*idx + 2] -= dots * kz;
    }
}

// ============================== MAIN FXN =============================================

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
                        const Scalar* d_A_half_mat)
{
    // zero out reduction matrices
    cudaMemset(d_B_T_F_cos, 0, Nk*3*sizeof(Scalar));
    cudaMemset(d_B_T_F_sin, 0, Nk*3*sizeof(Scalar));

    calculate_dft_and_reduce<<< N/BLOCK_SIZE + 1, BLOCK_SIZE, 3*Nk*sizeof(Scalar) >>>(d_pos,
                    d_index_array,
                    d_net_force,
                    d_B_T_F_cos,
                    d_B_T_F_sin,
                    d_dft_cos,
                    d_dft_sin,
                    d_ks_mat,
                    box,
                    N,
                    D,
                    Nk,
                    dt,
                    T,
                    alpha,
                    timestep,
                    wave_seed);
    
    project_ks<<< 1, Nk >>>(d_B_T_F_cos, d_B_T_F_sin, d_ks_norm_mat, Nk);

    add_to_pos<<< N/BLOCK_SIZE + 1, BLOCK_SIZE, (18 + 6*Nk)*sizeof(Scalar) >>>(d_net_force, d_B_T_F_cos, d_B_T_F_sin, d_A_mat, d_A_half_mat, 
                                                    d_dft_cos, d_dft_sin, d_pos, 
                                                    d_index_array, d_tag, d_image, box, 
                                                    alpha, N, Nk, timestep, seed, T, dt);


    return cudaSuccess;
}
