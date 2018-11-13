#include <hoomd/ExecutionConfiguration.h>

#include <iostream>
#include <stdio.h>

#include "CollectiveMode.h"
#ifdef ENABLE_CUDA
#include "CollectiveModeGPU.h"
#endif


#include <hoomd/Initializers.h>
#include <hoomd/md/IntegratorTwoStep.h>
#include <hoomd/Saru.h>

#include <math.h>
#include <hoomd/extern/upp11/upp11.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/numpy.h>

#include "hoomd/extern/Eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;
using namespace hoomd;
namespace py = pybind11;

typedef std::complex<float> cfloat;
#define PI 3.141592653589793f

#define CHECK_CLOSE(a,b,c) UP_ASSERT((std::abs((a)-(b)) <= ((c) * std::abs(a))) && (std::abs((a)-(b)) <= ((c) * std::abs(b))))
#define CHECK_SMALL(a,c) UP_ASSERT(std::abs(a) < c)

int main(int argc, char **argv)
{
    return upp11::TestMain().main(argc, argv);
}

// --------------------------------------------- HELPER FUNCTIONS ---------------------------------------------------

void set_positions_from_matrix(std::shared_ptr<ParticleData> pdata, Eigen::MatrixXf& pos)
{
    for (int i = 0; i < pos.rows(); i++)
    {
        pdata->setPosition(i, make_scalar3(pos(i,0), pos(i,1), pos(i,2)));
    }
}

inline float wrap_box(float dx, float L)
{
    return dx - L*(dx > L/2) + L*(dx < -L/2);
}

void calculate_mobility(MatrixXf& pos, MatrixXf& A, MatrixXcf& B, MatrixXcf& dft_coeffs, MatrixXf& ks, int N, int Nk, float alpha)
{
    A = MatrixXf::Identity(3*N, 3*N);
    B = MatrixXf::Zero(3*N, 3*Nk);
    dft_coeffs = MatrixXf::Zero(N, Nk);
    cfloat im(0,1);

    for (int j = 0; j < Nk; j++)
    {
        Matrix3f I_kk = Matrix3f::Identity() - ks.row(j).normalized().transpose() * ks.row(j).normalized();

        for (int i = 0; i < N; i++)
        {
            A.block(3*i,3*i,3,3) -= alpha/Nk * I_kk;

            B.block(3*i,3*j,3,3) = exp(im*(pos.row(i) * ks.row(j).transpose())(0)) * I_kk;
            dft_coeffs(i,j) = exp(im*(pos.row(i) * ks.row(j).transpose())(0));
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------

template <class CollectiveMode>
void collective_mode_mobility_gpu(std::shared_ptr<ExecutionConfiguration> exec_conf, float alpha)
{
    int N = 300;
    int Nk = 8;
    Vector3f L(3,5,4.2);

    MatrixXf ks_part(Nk/2, 3);
    MatrixXf ks(Nk, 3);

    ks_part <<   3, 0, 0,
            0, 2, 0,
            0, 0, 1,
            0, -1, 4;

    ks << ks_part, -ks_part;

    ks = (2*PI*ks).array().rowwise() / L.array().transpose();

    MatrixXf pos = MatrixXf::Random(N, 3).array().rowwise() * (L.array().transpose()/2);
    MatrixXf emp_M = MatrixXf::Zero(3*N, 3*N);
    MatrixXf M(3*N, 3*N);

    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(L(0), L(1), L(2)), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    set_positions_from_matrix(pdata, pos);

    // ------------ calculate real mobility matrix analytically -----------------------
    
    MatrixXf A = MatrixXf::Identity(3*N, 3*N);
    MatrixXcf B = MatrixXf::Zero(3*N, 3*Nk);
    MatrixXcf dft_coeffs = MatrixXf::Zero(N, Nk);

    calculate_mobility(pos, A, B, dft_coeffs, ks, N, Nk, alpha);

    MatrixXcf BBT = B * B.adjoint();

    M = A + alpha/Nk * BBT.real();

    // ----------------- construct simulation ---------------------------------

    Scalar dt = Scalar(1.0); // dt needs to be 1 in order to extract mobility matrix
    shared_ptr<VariantConst> T(new VariantConst(0.0));

    shared_ptr<CollectiveModeGPU> cmbd(new CollectiveModeGPU(sysdef, group_all, T, 124, ks_part, alpha));
    shared_ptr<IntegratorTwoStep> cmbd_up(new IntegratorTwoStep(sysdef, dt));
    cmbd_up->addIntegrationMethod(cmbd);
    cmbd_up->prepRun(0);

    // construct empirical mobility matrix column by column with unit vector forces
    for (int i = 0; i < 3*N; i++)
    {
		ArrayHandle<Scalar4> h_net_force(pdata->getNetForce(), access_location::host, access_mode::readwrite);

    	// reset positions
    	set_positions_from_matrix(pdata, pos);

    	// set unit vector force
    	for (int ii = 0; ii < N; ii++)
    	{
    		h_net_force.data[ii] = make_scalar4(0,0,0,0);

    		if (ii == i/3)
    		{
    			h_net_force.data[ii] = make_scalar4((i % 3)==0, (i % 3)==1, (i % 3)==2, 0);
    		}
    	}

    	cmbd_up->update(0);

    	for (int ii = 0; ii < N; ii++)
    	{
    		Scalar3 newpos = pdata->getPosition(ii);

            
    		emp_M(3*ii, i) = wrap_box(newpos.x - pos(ii, 0), L(0));
    		emp_M(3*ii+1, i) = wrap_box(newpos.y - pos(ii, 1), L(1));
    		emp_M(3*ii+2, i) = wrap_box(newpos.z - pos(ii, 2), L(2));
    	}
    }

    UP_ASSERT((emp_M - M).array().abs().maxCoeff() < 1e-5);
}

template <class CollectiveMode>
void collective_mode_finite_temp(std::shared_ptr<ExecutionConfiguration> exec_conf, float alpha, float temp)
{
    int N = 275; // just a little more than one block of particles
    int Nk = 8;
    Vector3f L(3,5,4.2);
    Scalar T_val = temp;
    int num_steps = 1000;

    // calculate seed
    unsigned int seed = 234;
    // unsigned int seed_tmp = seed_tmp*0x12345677 + 0x12345 ; seed_tmp^=(seed_tmp>>16); seed_tmp*= 0x45679;
    // std::mt19937 mt_rand(seed_tmp);
    unsigned int wave_seed = 4147318334;

    MatrixXf ks_part(Nk/2, 3);
    MatrixXf ks(Nk, 3);

    ks_part <<   3, 0, 0,
            0, 2, 0,
            0, 0, 1,
            0, -1, 4;

    ks << ks_part, -ks_part;

    ks = (2*PI*ks).array().rowwise() / L.array().transpose();

    MatrixXf pos = MatrixXf::Random(N, 3).array().rowwise() * (L.array().transpose()/2);
    MatrixXf emp_M = MatrixXf::Zero(3*N, 3*N);
    MatrixXf M(3*N, 3*N);

    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(L(0), L(1), L(2)), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    // ------------ calculate real mobility matrix analytically -----------------------
    MatrixXf A = MatrixXf::Identity(3*N, 3*N);
    MatrixXf A_half(3*N, 3*N);
    MatrixXcf B = MatrixXf::Zero(3*N, 3*Nk);
    MatrixXcf dft_coeffs = MatrixXf::Zero(N, Nk);
    cfloat im(0,1);

    calculate_mobility(pos, A, B, dft_coeffs, ks, N, Nk, alpha);
    A_half = A.llt().matrixL();

    MatrixXcf BBT = B * B.adjoint();

    M = A + alpha/Nk * BBT.real();

    // --------------------------- construct simulation ----------------------------------

    set_positions_from_matrix(pdata, pos);

    Scalar dt = Scalar(0.1);
    shared_ptr<VariantConst> T(new VariantConst(T_val));

    shared_ptr<CollectiveModeGPU> cmbd(new CollectiveModeGPU(sysdef, group_all, T, seed, ks_part, alpha));
    shared_ptr<IntegratorTwoStep> cmbd_up(new IntegratorTwoStep(sysdef, dt));
    cmbd_up->addIntegrationMethod(cmbd);
    cmbd_up->prepRun(0);

    VectorXf disp(3*N);

    float avg_trace = 0.0;
    VectorXf psi1(3*N);
    VectorXcf psi2(3*Nk);
    MatrixXf prev_pos = pos;
    seed = 1895357627;                          // necessary to get the same displacements from Saru due to hashing

    // step forward in time and calculate mean square displacement
    for (int i = 0; i < num_steps; i++)
    {
        cmbd_up->update(i);

    	for (int ii = 0; ii < N; ii++)
    	{
    		Scalar3 newpos = pdata->getPosition(ii);
            
            // calculate displacements
    		disp(3*ii) = wrap_box(newpos.x - prev_pos(ii, 0), L(0));
    		disp(3*ii+1) = wrap_box(newpos.y - prev_pos(ii,1), L(1));
    		disp(3*ii+2) = wrap_box(newpos.z - prev_pos(ii,2), L(2));

            //update prev_pos
            prev_pos(ii, 0) = newpos.x;
            prev_pos(ii, 1) = newpos.y;
            prev_pos(ii, 2) = newpos.z;

            detail::Saru saru(ii, i, seed);
            psi1(3*ii) = sqrt(3*2*T_val*dt) * saru.s<Scalar>(-1,1);
            psi1(3*ii+1) = sqrt(3*2*T_val*dt) * saru.s<Scalar>(-1,1);
            psi1(3*ii+2) = sqrt(3*2*T_val*dt) * saru.s<Scalar>(-1,1);
    	}

        for (int j = 0; j < 3; j++)
        {
            detail::Saru saru(j, i, wave_seed);
            detail::Saru saru2(j+3, i, wave_seed);

            for (int k = 0; k < Nk/2; k++)
            {
                cfloat new_rand(saru.s<Scalar>(-1,1), saru2.s<Scalar>(-1,1));
                psi2(3*k + j) = sqrt(3*T_val*dt*alpha/Nk) * new_rand; // divide by sqrt(2) for variance
                psi2(3*(k + Nk/2) + j) = conj(psi2(3*k + j));
            }
        }

        // ensure each step is taken correctly
        UP_ASSERT((A_half*psi1 + (B*psi2).real() - disp).array().abs().maxCoeff() < 0.001);

        // update mobility
        calculate_mobility(prev_pos, A, B, dft_coeffs, ks, N, Nk, alpha);
        A_half = A.llt().matrixL();

    	avg_trace += disp.squaredNorm() / num_steps;
    }

    // assert that the trace of the mobility matrix is 6NT*dt
    UP_ASSERT(fabs(avg_trace / (6*T_val*dt*N) - 1.0) < 0.01);
}

#ifdef ENABLE_CUDA
UP_TEST( CMBD_Explicit_Mobility_Self_GPU )
{
    collective_mode_mobility_gpu<CollectiveModeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)), 0.0);
}

UP_TEST( CMBD_Explicit_Mobility_Full_GPU )
{
    collective_mode_mobility_gpu<CollectiveModeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)), 0.95);
}

UP_TEST( CMBD_Finite_Temperature_Full_GPU )
{
    collective_mode_finite_temp<CollectiveModeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)), 0.95, 0.7);
}
#endif
