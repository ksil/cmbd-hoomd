#ifndef __COLLECTIVE_MODE_H__
#define __COLLECTIVE_MODE_H__

/*! \file CollectiveMode.h
    \brief Declares the CollectiveMode class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/Variant.h>
#include <hoomd/md/IntegrationMethodTwoStep.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/numpy.h>
#include "hoomd/extern/Eigen/Eigen/Dense"

using namespace Eigen;

//! Integrates part of the system forward with Collective Mode Brownian Dynamics
/*! Implements the integrator.

    Collective Mode Brownian Dynamics is similar to Brownian dynamics in that it integrates
    the overdamped Langevin equation (i.e., particle acceleration set to 0), but particle motion
    is correlated via a mobility matrix that induces collective motion at specified length scales. 

    \ingroup updaters
*/
class PYBIND11_EXPORT CollectiveMode : public IntegrationMethodTwoStep
{
    public:
        //! Constructs the integration method and associates it with the system
        CollectiveMode(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    pybind11::array_t<Scalar> ks,
        		    Scalar alpha
                    );

        CollectiveMode(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    Eigen::MatrixXf& ks,
                    Scalar alpha
                    );

        virtual ~CollectiveMode();

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        // set the excited wave vectors
        virtual void set_ks(pybind11::array_t<Scalar> ks);

        void set_alpha(Scalar alpha);

    protected:
        Eigen::Matrix<Scalar,3,3,RowMajor> A_mat;
        Eigen::Matrix<Scalar,3,3> A_half_mat;
        Eigen::Matrix<Scalar,Dynamic,3,RowMajor> ks_mat;
        Eigen::Matrix<Scalar,Dynamic,3,RowMajor> ks_norm_mat;


        unsigned int Nk;
        unsigned int N;
        unsigned int D;

        std::shared_ptr<Variant> m_T;
        Scalar m_alpha;
        unsigned int m_seed;
        unsigned int m_wave_seed;

        void calculateA();
};

//! Exports the CollectiveMode class to python
void export_CollectiveMode(pybind11::module& m);

#endif // #ifndef __COLLECTIVE_MODE_H__
