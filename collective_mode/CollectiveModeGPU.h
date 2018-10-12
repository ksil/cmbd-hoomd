#ifndef __COLLECTIVE_MODE_GPU__
#define __COLLECTIVE_MODE_GPU__

/*! \file CollectiveModeGPU.h
    \brief Declares the CollectiveModeGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CollectiveMode.h"
#include <hoomd/md/IntegrationMethodTwoStep.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <cublas_v2.h>

//! Integrates part of the system forward with Collective Mode Brownian Dynamics
/*! Implements the integrator.

    Collective Mode Brownian Dynamics is similar to Brownian dynamics in that it integrates
    the overdamped Langevin equation (i.e., particle acceleration set to 0), but particle motion
    is correlated via a mobility matrix that induces collective motion at specified length scales. 

    \ingroup updaters
*/
class PYBIND11_EXPORT CollectiveModeGPU : public CollectiveMode
{
    public:
        //! Constructs the integration method and associates it with the system
        CollectiveModeGPU(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    pybind11::array_t<Scalar> ks,
                    Scalar alpha
                    );

        virtual ~CollectiveModeGPU();

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        Scalar* d_dft_cos;
        Scalar* d_dft_sin;
        Scalar* d_B_T_F_cos;
        Scalar* d_B_T_F_sin;
        Scalar* d_F;

        Scalar* d_ks_mat;
        Scalar* d_ks_norm_mat;
        Scalar* d_A_mat;
        Scalar* d_A_half_mat;

        cublasHandle_t handle;

        virtual void initializeMatrices();
        virtual void freeMatrices();
};

//! Exports the CollectiveModeGPU class to python
void export_CollectiveModeGPU(pybind11::module& m);

#endif // #ifndef __COLLECTIVE_MODE_GPU__
