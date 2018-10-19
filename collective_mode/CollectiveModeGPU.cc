#include "CollectiveModeGPU.h"
#include "CollectiveModeGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;

using namespace std;

/*! \file CollectiveModeGPU.cc
    \brief Contains code for the GPU implementation of CollectiveMode
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param ks A numpy array containing k vectors to excite (one per row)
    \param alpha Coefficient determining how much collective modes are excited
*/
CollectiveModeGPU::CollectiveModeGPU(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    pybind11::array_t<Scalar> ks,
                    Scalar alpha)
    : CollectiveMode(sysdef, group, T, seed, ks, alpha)
{
    if (!m_exec_conf->isCUDAEnabled())
    {
        m_exec_conf->msg->error() << "Creating a CollectiveModeGPU while CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing CollectiveModeGPU");
    }

    initializeMatrices();
}

CollectiveModeGPU::CollectiveModeGPU(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    Eigen::MatrixXf& ks,
                    Scalar alpha)
    : CollectiveMode(sysdef, group, T, seed, ks, alpha)
{
    if (!m_exec_conf->isCUDAEnabled())
    {
        m_exec_conf->msg->error() << "Creating a CollectiveModeGPU while CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing CollectiveModeGPU");
    }

    initializeMatrices();
}

CollectiveModeGPU::~CollectiveModeGPU()
{
    m_exec_conf->msg->notice(5) << "Destroying CollectiveModeGPU" << endl;

    freeMatrices();
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void CollectiveModeGPU::initializeMatrices()
{
    cudaMalloc((void **)&d_dft_cos, N*Nk * sizeof(Scalar));
    cudaMalloc((void **)&d_dft_sin, N*Nk * sizeof(Scalar));
    cudaMalloc((void **)&d_ks_mat, Nk*3 * sizeof(Scalar));
    cudaMalloc((void **)&d_ks_norm_mat, Nk*3 * sizeof(Scalar));
    cudaMalloc((void **)&d_A_mat, 9 * sizeof(Scalar));
    cudaMalloc((void **)&d_A_half_mat, 9 * sizeof(Scalar));
    cudaMalloc((void **)&d_B_T_F_cos, Nk*3 * sizeof(Scalar));
    cudaMalloc((void **)&d_B_T_F_sin, Nk*3 * sizeof(Scalar));

    // copy A and ks matrix to device
    cudaMemcpy(d_ks_mat, ks_mat.data(), Nk*3 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks_norm_mat, ks_norm_mat.data(), Nk*3 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_mat, A_mat.data(), 9 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_half_mat, A_half_mat.data(), 9 * sizeof(Scalar), cudaMemcpyHostToDevice);
}

void CollectiveModeGPU::freeMatrices()
{
    cudaFree(d_dft_cos);
    cudaFree(d_dft_sin);
    cudaFree(d_ks_mat);
    cudaFree(d_ks_norm_mat);
    cudaFree(d_A_mat);
    cudaFree(d_A_half_mat);
    cudaFree(d_B_T_F_cos);
    cudaFree(d_B_T_F_sin);
}

void CollectiveModeGPU::set_ks(pybind11::array_t<Scalar> ks)
{
    unsigned int old_Nk = Nk;
    CollectiveMode::set_ks(ks);

    // if the number of wave vectors changes, reallocate memory on the GPU
    if (old_Nk != Nk)
    {
        cudaFree(d_ks_mat);
        cudaFree(d_ks_norm_mat);
        cudaMalloc((void **)&d_ks_mat, Nk*3 * sizeof(Scalar));
        cudaMalloc((void **)&d_ks_norm_mat, Nk*3 * sizeof(Scalar));
    }

    // copy the new A and ks matrices to the device
    cudaMemcpy(d_ks_mat, ks_mat.data(), Nk*3 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ks_norm_mat, ks_norm_mat.data(), Nk*3 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_mat, A_mat.data(), 9 * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_half_mat, A_half_mat.data(), 9 * sizeof(Scalar), cudaMemcpyHostToDevice);
}

/*! \param timestep Current time step
    \post Particle positions are moved forward a full time step and velocities are redrawn from the proper distrubtion.
*/
void CollectiveModeGPU::integrateStepOne(unsigned int timestep)
{
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "BD step 1");

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // perform the update on the GPU
    gpu_collective(timestep,
                m_seed,
                m_wave_seed,
                d_pos.data,
                d_image.data,
                box,
                d_index_array.data,
                d_tag.data,
                m_alpha,
                N,
                Nk,
                d_net_force.data,
                m_deltaT,
                D,
                m_T->getValue(timestep),
                d_dft_cos,
                d_dft_sin,
                d_B_T_F_cos,
                d_B_T_F_sin,
                d_ks_mat,
                d_ks_norm_mat,
                d_A_mat,
                d_A_half_mat);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
}

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void CollectiveModeGPU::integrateStepTwo(unsigned int timestep)
{
    // there is no step 2
}

void export_CollectiveModeGPU(py::module& m)
{
    py::class_<CollectiveModeGPU, std::shared_ptr<CollectiveModeGPU> >(m, "CollectiveModeGPU", py::base<CollectiveMode>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            unsigned int,
                            pybind11::array_t<Scalar>,
                            Scalar>())
        ;
}
