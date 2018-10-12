#include "CollectiveModeGPU.cuh"
#include "hoomd/VectorMath.h"
#include "hoomd/HOOMDMath.h"
#include <stdio.h>

#include "hoomd/Saru.h"
using namespace hoomd;

#include <assert.h>

// definitions to call the correct cublas functions depending on specified precision
#ifdef SINGLE_PRECISION

#define GEMV(...) cublasSgemv(__VA_ARGS__)
#define SINCOSPI(...) sincospif(__VA_ARGS__)

#else

#define GEMV(...) cublasDgemv(__VA_ARGS__)
#define SINCOSPI(...) sincospi(__VA_ARGS__)

#endif

#define BLOCK_SIZE 256

/*! \file CollectiveModeGPU.cu
    \brief Defines GPU kernel code for Collective Mode Brownian Dynamics.
*/

extern "C" __global__
void print_matrix(const Scalar* mat, const int m, const int n, bool rowmajor)
{
    printf("(%p)\n", (void*)mat);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (rowmajor) printf("%f ", mat[i*n + j]);
            else printf("%f ", mat[j*m + i]);
        }
        printf("\n");
    }
}

// ------------------------------ HELPER FXNS ---------------------------------

// adds a column-major displacement to the Scalar 4 array of positions
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
                int3* d_image,
                const BoxDim box,
                const Scalar alpha,
                const unsigned int N,
                const unsigned int Nk,
                const unsigned int D,
                const unsigned int timestep,
                const unsigned int seed,
                const Scalar T,
                const Scalar dt)
{
    extern __shared__ Scalar buf[];
    Scalar* A = buf;
    Scalar* A_half = &buf[9];
    Scalar* C_T  = &buf[18];
    Scalar* S_T = &buf[18 + Nk*D];

    int idx = threadIdx.x;

    // copy A and B_T_F matrices to shared memory
    while (idx < max(D*Nk, 9))
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

        // --------------- self part -------------------

        // dt * F*A
        p.x += dt*(F.x*A[0] + F.y*A[1] + F.z*A[2]);
        p.y += dt*(F.x*A[3] + F.y*A[4] + F.z*A[5]);
        p.z += dt*(F.x*A[6] + F.y*A[7] + F.z*A[8]);

        // Brownian part
        detail::Saru saru(idx, timestep, seed);
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
            p.x += c * C_T[i];
            p.x += s * S_T[i];
            p.y += c * C_T[i + Nk];
            p.y += s * S_T[i + Nk];
            if (D == 3)
            {
                p.z += c * C_T[i + 2*Nk];
                p.z += s * S_T[i + 2*Nk];
            }
        }

        // wrap particles in case displacements pushed particles outside box
        box.wrap(p, image);

        // write out data
        d_pos[p_idx] = p;
        d_image[p_idx] = image;
    }
}

// calculates the nonuniform Fourier transform for given k vectors
extern "C" __global__
void calculate_dft(const Scalar4* d_pos,
                    const unsigned int* d_index_array,
                    Scalar* d_dft_cos,
                    Scalar* d_dft_sin,
                    const Scalar* d_ks_mat,
                    const BoxDim box,
                    const unsigned int N,
                    const unsigned int D,
                    const unsigned int Nk)
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

    if (idx < N)
    {
        unsigned int p_idx = d_index_array[idx];
        Scalar4 pos = d_pos[p_idx];
        Scalar3 L = box.getL();

        for (int i = 0; i < Nk; i++)
        {
            // argument is 2 * pi * dot(k, x/L)
            Scalar arg = 2*(ks[3*i]*pos.x/L.x + ks[3*i+1]*pos.y/L.y);
            if (D == 3) arg += 2*ks[3*i+2]*pos.z/L.z;
            SINCOSPI(arg, &d_dft_sin[idx*Nk + i], &d_dft_cos[idx*Nk + i]);
        }
    }
}

// takes a matrix d_mat and performs the orthogonal projection (I - kk) * d_mat
// assumes kernel will be launched with 1 block consisting of # threads = Nk
extern "C" __global__
void project_ks(Scalar* d_mat, const Scalar* d_ks_norm_mat, const unsigned int Nk, const unsigned int D)
{
    extern __shared__ Scalar ks_norm[];

    int idx = threadIdx.x;

    if (idx < Nk)
    {
        // copy ks_norm_mat (column-major) to shared memory
        ks_norm[idx] = d_ks_norm_mat[idx];
        ks_norm[idx + Nk] = d_ks_norm_mat[idx + Nk];
        if (D == 3) ks_norm[idx + 2*Nk] = d_ks_norm_mat[idx + 2*Nk];

        // do dot product
        Scalar dot = ks_norm[idx] * d_mat[idx];
        dot += ks_norm[idx + Nk] * d_mat[idx + Nk];
        if (D == 3) dot += ks_norm[idx + 2*Nk] * d_mat[idx + 2*Nk];

        // I - kk
        d_mat[idx] -= dot * ks_norm[idx];
        d_mat[idx + Nk] -= dot * ks_norm[idx + Nk];
        if (D == 3) d_mat[idx + 2*Nk] -= dot * ks_norm[idx + 2*Nk];
    }
}


// copies the components of each force for each particle in the group from
// a Scalar4 array to a NxD array of scalars, d_F
extern "C" __global__
void copy_forces(const Scalar4* d_net_force,
                    Scalar* d_F,
                    const unsigned int* d_index_array,
                    const unsigned int N,
                    const unsigned int D)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        unsigned int p_idx = d_index_array[idx];
        Scalar4 net_force = d_net_force[p_idx];

        // column-major ordering for d_F
        d_F[idx] = net_force.x;
        d_F[idx + N] = net_force.y;
        if (D == 3) d_F[idx + 2*N] = net_force.z;
    }
}

// calculates a random displacement for a given temperature
extern "C" __global__
void fill_random_psi(Scalar* psi1,
                    const unsigned int timestep,
                    const unsigned int seed,
                    const unsigned int size,
                    const Scalar T,
                    const Scalar dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        detail::Saru saru(idx, timestep, seed);
        Scalar s = saru.s<Scalar>(-1,1);

        psi1[idx] = s * fast::sqrt(Scalar(3.0)*Scalar(2.0)*T*dt); // sqrt(3) because not Gaussian
    }
}

// ----------------------------- MAIN FXN ------------------------------------

/*! \param d_pos array of particle positions and types
    \param d_image array of particle images
    \param box simulation box
    \param d_tag array of particle tags
    \param d_index_array Device array listing the indicies of the mebers of the group to integrate
    \param group_size Number of members in the group
    \param d_net_force Net force on each particle
    \param d_gamma_r List of per-type gamma_rs (rotational drag coeff.)
    \param d_orientation Device array of orientation quaternion
    \param d_torque Device array of net torque on each particle
    \param d_inertia Device array of moment of inertial of each particle
    \param d_angmom Device array of transformed angular momentum quaternion of each particle (see online documentation)
    \param langevin_args Collected arguments for gpu_brownian_step_one_kernel()
    \param aniso If set true, the system would go through rigid body updates for its orientation
    \param deltaT Amount of real time to step forward in one time step
    \param D Dimensionality of the system

    This is just a driver for gpu_brownian_step_one_kernel(), see it for details.
*/
cudaError_t gpu_collective(unsigned int timestep,
                        unsigned int seed,
                        Scalar4 *d_pos,
                        int3 *d_image,
                        const BoxDim &box,
                        const unsigned int *d_index_array,
                        const Scalar alpha,
                        const unsigned int N,
                        const unsigned int Nk,
                        const Scalar4 *d_net_force,
                        const Scalar dt,
                        const unsigned int D,
                        const Scalar T,
                        Scalar *d_F,
                        Scalar *d_dft_cos,
                        Scalar *d_dft_sin,
                        Scalar *d_B_T_F_cos,
                        Scalar *d_B_T_F_sin,
                        const Scalar *d_ks_mat,
                        const Scalar *d_ks_norm_mat,
                        const Scalar *d_A_mat,
                        const Scalar *d_A_half_mat,
                        cublasHandle_t handle)
{

    // setup the grid to run the kernel
    Scalar a, b;
    int m, n, lda;

    // cublasSgemm(handle, transa, transb, m, n, k, *alpha, *A, lda, *B, ldb, *beta, *C, ldc)
    //printf("Copying forces...\n");
    // copy forces to matrix with row-major order
    copy_forces<<< N/BLOCK_SIZE + 1, BLOCK_SIZE >>>(d_net_force, d_F, d_index_array, N, D);

    // ----------------------------- B part ------------------------------------

    // calculate dft - d_dft_cos and d_dft_sin stored as row-major matrices
    //printf("Calculating dft...\n");
    calculate_dft<<< N/BLOCK_SIZE + 1, BLOCK_SIZE, 3*Nk*sizeof(Scalar) >>>(d_pos, d_index_array, d_dft_cos, d_dft_sin, d_ks_mat, box, N, D, Nk);

    // --------- cos part -------------

    // find stochastic part
    //printf("Fill random psi...\n");
    fill_random_psi<<< (Nk*D)/BLOCK_SIZE + 1, BLOCK_SIZE >>>(d_B_T_F_cos, timestep, seed+1, Nk*D, T, dt);

    // do Fourier transform of forces (cos) - leads to a Nk x D matrix stored in psi1
    // b factor comes from 2*alpha/Nk * d_dft_cos*psi / sqrt(2) for variance adjustment
    //printf("Doing GEMV...\n");
    m = Nk; n = N; lda = m; a = dt*alpha/Nk; b = sqrt(2*alpha/Nk);
    GEMV(handle, CUBLAS_OP_N, m, n, &a, d_dft_cos, lda, d_F, 1, &b, d_B_T_F_cos, 1);
    GEMV(handle, CUBLAS_OP_N, m, n, &a, d_dft_cos, lda, &d_F[N], 1, &b, &d_B_T_F_cos[Nk], 1);
    if (D == 3) GEMV(handle, CUBLAS_OP_N, m, n, &a, d_dft_cos, lda, &d_F[2*N], 1, &b, &d_B_T_F_cos[2*Nk], 1);

    // apply projection I - kk
    //printf("Projecting ks...\n");
    project_ks<<< 1, Nk, Nk*D*sizeof(Scalar) >>>(d_B_T_F_cos, d_ks_norm_mat, Nk, D);

    // --------- sin part -------------

    // find stochastic part
    fill_random_psi<<< (Nk*D)/BLOCK_SIZE + 1, BLOCK_SIZE >>>(d_B_T_F_sin, timestep, seed+2, Nk*D, T, dt);

    // do Fourier transform of forces (sin) - leads to a Nk x D matrix stored in psi1
    // b factor comes from -2*alpha/Nk * d_dft_sin*psi / sqrt(2) for variance adjustment
    m = Nk; n = N; lda = m; a = dt*alpha/Nk; b = -sqrt(2*alpha/Nk);
    GEMV(handle, CUBLAS_OP_N, m, n, &a, d_dft_sin, lda, d_F, 1, &b, d_B_T_F_sin, 1);
    GEMV(handle, CUBLAS_OP_N, m, n, &a, d_dft_sin, lda, &d_F[N], 1, &b, &d_B_T_F_sin[Nk], 1);
    if (D == 3) GEMV(handle, CUBLAS_OP_N, m, n, &a, d_dft_sin, lda, &d_F[2*N], 1, &b, &d_B_T_F_sin[2*Nk], 1);

    // apply projection I - kk
    project_ks<<< 1, Nk, Nk*D*sizeof(Scalar) >>>(d_B_T_F_sin, d_ks_norm_mat, Nk, D);

    // ------------------- add to pos and wrap box -----------------------------

    //printf("Add to pos...\n");
    add_to_pos<<< N/BLOCK_SIZE + 1, BLOCK_SIZE, (18 + 2*Nk*D)*sizeof(Scalar) >>>(d_net_force, d_B_T_F_cos, d_B_T_F_sin, d_A_mat, d_A_half_mat, 
                                                    d_dft_cos, d_dft_sin, d_pos, 
                                                    d_index_array, d_image, box, 
                                                    alpha, N, Nk, D, timestep, seed, T, dt);

    //printf("Done...\n");


    return cudaSuccess;
}
