#include "CollectiveMode.h"
#include <hoomd/VectorMath.h>
#include <hoomd/HOOMDMath.h>
#include <hoomd/md/IntegrationMethodTwoStep.h>
#include <random>

#include <hoomd/Saru.h>
using namespace hoomd;


#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
using namespace std;
using Eigen::Matrix;

/*! \file CollectiveMode.cc
    \brief Contains code for the CollectiveMode class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param ks A numpy array containing k vectors to excite (one per row)
    \param alpha Coefficient determining how much collective modes are excited
*/
CollectiveMode::CollectiveMode(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    pybind11::array_t<Scalar> ks,
                    Scalar alpha
                    )
                    : IntegrationMethodTwoStep(sysdef, group),
                    m_T(T), m_alpha(alpha), m_seed(seed)
{
    m_exec_conf->msg->notice(5) << "Constructing CollectiveMode" << endl;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;
    std::mt19937 mt_rand(m_seed);
    m_wave_seed = mt_rand();
    
    // set constants
    D = Scalar(sysdef->getNDimensions());
    N = group->getNumMembers();

    Scalar3 box_dims = m_pdata->getBox().getL();
    Scalar L[3] = {box_dims.x, box_dims.y, box_dims.z};

    // copy numpy matrix to Eigen matrix
    Nk = ks.shape(0);
    auto r = ks.unchecked<2>();
    if (ks.shape(1) != D)
    {
        throw std::runtime_error("ks is not a Nk x D array, where D is the number of dimensions");
    }

    ks_mat.resize(Nk, 3);
    ks_mat.setZero();

    for (unsigned int j = 0; j < D; j++)
    {
        for (unsigned int i = 0; i < Nk; i++)
        {
            ks_mat(i,j) = 2*M_PI*r(i,j) / L[j];
        }
    }

    ks_norm_mat = ks_mat.rowwise().normalized();

    // calculate the self part of the mobility matrix
    calculateA();

}

CollectiveMode::CollectiveMode(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    Eigen::MatrixXf& ks,
                    Scalar alpha
                    )
                    : IntegrationMethodTwoStep(sysdef, group),
                    m_T(T), m_alpha(alpha), m_seed(seed)
{
    m_exec_conf->msg->notice(5) << "Constructing CollectiveMode" << endl;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;
    std::mt19937 mt_rand(m_seed);
    m_wave_seed = mt_rand();

    printf("seed:%u wave_seed:%u\n",m_seed,m_wave_seed);
    
    // set constants
    D = Scalar(sysdef->getNDimensions());
    N = group->getNumMembers();

    Scalar3 box_dims = m_pdata->getBox().getL();
    Scalar L[3] = {box_dims.x, box_dims.y, box_dims.z};

    // copy numpy matrix to Eigen matrix
    Nk = ks.rows();
    if (ks.cols() != D)
    {
        throw std::runtime_error("ks is not a Nk x D array, where D is the number of dimensions");
    }

    ks_mat.resize(Nk, 3);
    ks_mat.setZero();

    ks_mat.block(0,0,Nk,D) = ks;

    ks_mat.col(0) *= 2*M_PI/L[0];
    ks_mat.col(1) *= 2*M_PI/L[1];
    if (D == 3) ks_mat.col(2) *= 2*M_PI/L[2];

    ks_norm_mat = ks_mat.rowwise().normalized();

    // calculate the self part of the mobility matrix
    calculateA();
}

CollectiveMode::~CollectiveMode()
{
    m_exec_conf->msg->notice(5) << "Destroying CollectiveMode" << endl;
}

void CollectiveMode::set_ks(pybind11::array_t<Scalar> ks)
{
    // get box size
    Scalar3 box_dims = m_pdata->getBox().getL();
    Scalar L[3] = {box_dims.x, box_dims.y, box_dims.z};

    // copy numpy matrix to Eigen matrix
    Nk = ks.shape(0);
    auto r = ks.unchecked<2>();
    if (ks.shape(1) != D)
    {
        throw std::runtime_error("ks is not a Nk x D array, where D is the number of dimensions");
    }

    ks_mat.resize(Nk, 3);
    ks_mat.setZero();

    for (unsigned int j = 0; j < D; j++)
    {
        for (unsigned int i = 0; i < Nk; i++)
        {
            ks_mat(i,j) = 2*M_PI*r(i,j) / L[j];
        }
    }

    ks_norm_mat = ks_mat.rowwise().normalized();

    // calculate the self part of the mobility matrix
    calculateA();
}

void CollectiveMode::set_alpha(Scalar alpha)
{
    m_alpha = alpha;
}

void CollectiveMode::calculateA()
{
    A_mat.setZero();
    A_mat.topLeftCorner(D,D).setIdentity();
    A_mat.topLeftCorner(D,D) *= (1 - m_alpha);

    A_mat.topLeftCorner(D,D) += (m_alpha/Nk) * (ks_norm_mat.transpose() * ks_norm_mat);
    
    // do Cholesky decomposition
    A_half_mat.setZero();
    A_half_mat.topLeftCorner(D,D) = A_mat.topLeftCorner(D,D).llt().matrixU();
}

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1
*/
void CollectiveMode::integrateStepOne(unsigned int timestep)
{
    // unsigned int group_size = m_group->getNumMembers();

    // // profile this step
    // if (m_prof)
    //     m_prof->push("CollectiveBD step 1");

    // // grab some initial variables
    // const Scalar currentTemp = m_T->getValue(timestep);
    // const unsigned int D = Scalar(m_sysdef->getNDimensions());

    // const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    // ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    // ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    // ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    // ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    // ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    // ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);

    // ArrayHandle<Scalar> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);
    // ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    // ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);

    // ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::readwrite);
    // ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);

    // const BoxDim& box = m_pdata->getBox();

    // // perform the first half step
    // // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // // v(t+deltaT) = random distribution consistent with T
    // for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
    // {
    //     unsigned int j = m_group->getMemberIndex(group_idx);
    //     unsigned int ptag = h_tag.data[j];

    //     // Initialize the RNG
    //     detail::Saru saru(ptag, timestep, m_seed);

    //     // compute the random force
    //     Scalar rx = saru.s<Scalar>(-1,1);
    //     Scalar ry = saru.s<Scalar>(-1,1);
    //     Scalar rz = saru.s<Scalar>(-1,1);

    //     Scalar gamma;
    //     if (m_use_lambda)
    //         gamma = m_lambda*h_diameter.data[j];
    //     else
    //     {
    //         unsigned int type = __scalar_as_int(h_pos.data[j].w);
    //         gamma = h_gamma.data[type];
    //     }

    //     // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
    //     // it is not the dimensionality of the system
    //     Scalar coeff = fast::sqrt(Scalar(3.0)*Scalar(2.0)*gamma*currentTemp/m_deltaT);
    //     if (m_noiseless_t)
    //         coeff = Scalar(0.0);
    //     Scalar Fr_x = rx*coeff;
    //     Scalar Fr_y = ry*coeff;
    //     Scalar Fr_z = rz*coeff;

    //     if (D < 3)
    //         Fr_z = Scalar(0.0);

    //     // update position
    //     h_pos.data[j].x += (h_net_force.data[j].x + Fr_x) * m_deltaT / gamma;
    //     h_pos.data[j].y += (h_net_force.data[j].y + Fr_y) * m_deltaT / gamma;
    //     h_pos.data[j].z += (h_net_force.data[j].z + Fr_z) * m_deltaT / gamma;

    //     // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    //     box.wrap(h_pos.data[j], h_image.data[j]);

    //     // draw a new random velocity for particle j
    //     Scalar mass =  h_vel.data[j].w;
    //     Scalar sigma = fast::sqrt(currentTemp/mass);
    //     h_vel.data[j].x = gaussian_rng(saru, sigma);
    //     h_vel.data[j].y = gaussian_rng(saru, sigma);
    //     if (D > 2)
    //         h_vel.data[j].z = gaussian_rng(saru, sigma);
    //     else
    //         h_vel.data[j].z = 0;

    //     // rotational random force and orientation quaternion updates
    //     if (m_aniso)
    //     {
    //         unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
    //         Scalar gamma_r = h_gamma_r.data[type_r];
    //         if (gamma_r > 0)
    //         {
    //             vec3<Scalar> p_vec;
    //             quat<Scalar> q(h_orientation.data[j]);
    //             vec3<Scalar> t(h_torque.data[j]);
    //             vec3<Scalar> I(h_inertia.data[j]);

    //             bool x_zero, y_zero, z_zero;
    //             x_zero = (I.x < EPSILON); y_zero = (I.y < EPSILON); z_zero = (I.z < EPSILON);

    //             Scalar sigma_r = fast::sqrt(Scalar(2.0)*gamma_r*currentTemp/m_deltaT);
    //             if (m_noiseless_r)
    //                 sigma_r = Scalar(0.0);

    //             // original Gaussian random torque
    //             // Gaussian random distribution is preferred in terms of preserving the exact math
    //             vec3<Scalar> bf_torque;
    //             bf_torque.x = gaussian_rng(saru, sigma_r);
    //             bf_torque.y = gaussian_rng(saru, sigma_r);
    //             bf_torque.z = gaussian_rng(saru, sigma_r);

    //             if (x_zero) bf_torque.x = 0;
    //             if (y_zero) bf_torque.y = 0;
    //             if (z_zero) bf_torque.z = 0;

    //             // use the damping by gamma_r and rotate back to lab frame
    //             // Notes For the Future: take special care when have anisotropic gamma_r
    //             // if aniso gamma_r, first rotate the torque into particle frame and divide the different gamma_r
    //             // and then rotate the "angular velocity" back to lab frame and integrate
    //             bf_torque = rotate(q, bf_torque);
    //             if (D < 3)
    //             {
    //                 bf_torque.x = 0;
    //                 bf_torque.y = 0;
    //                 t.x = 0;
    //                 t.y = 0;
    //             }

    //             // do the integration for quaternion
    //             q += Scalar(0.5) * m_deltaT * ((t + bf_torque) / gamma_r) * q ;
    //             q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));
    //             h_orientation.data[j] = quat_to_scalar4(q);

    //             // draw a new random ang_mom for particle j in body frame
    //             p_vec.x = gaussian_rng(saru, fast::sqrt(currentTemp * I.x));
    //             p_vec.y = gaussian_rng(saru, fast::sqrt(currentTemp * I.y));
    //             p_vec.z = gaussian_rng(saru, fast::sqrt(currentTemp * I.z));
    //             if (x_zero) p_vec.x = 0;
    //             if (y_zero) p_vec.y = 0;
    //             if (z_zero) p_vec.z = 0;

    //             // !! Note this isn't well-behaving in 2D,
    //             // !! because may have effective non-zero ang_mom in x,y

    //             // store ang_mom quaternion
    //             quat<Scalar> p = Scalar(2.0) * q * p_vec;
    //             h_angmom.data[j] = quat_to_scalar4(p);
    //         }
    //     }
    // }

    // done profiling
    if (m_prof)
        m_prof->pop();
}


/*! \param timestep Current time step
*/
void CollectiveMode::integrateStepTwo(unsigned int timestep)
{
    // there is no step 2 in Brownian dynamics.
}

void export_CollectiveMode(py::module& m)
{
    py::class_<CollectiveMode, std::shared_ptr<CollectiveMode> >(m, "CollectiveMode", py::base<IntegrationMethodTwoStep>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            unsigned int,
                            pybind11::array_t<Scalar>,
                            Scalar>())
        ;
}
