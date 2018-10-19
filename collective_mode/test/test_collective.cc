#include <hoomd/ExecutionConfiguration.h>

#include <iostream>
#include <stdio.h>

#include "CollectiveMode.h"
#ifdef ENABLE_CUDA
#include "CollectiveModeGPU.h"
#endif


#include <hoomd/Initializers.h>
#include <hoomd/md/IntegratorTwoStep.h>

#include <math.h>
#include <hoomd/extern/upp11/upp11.h>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#include <hoomd/extern/pybind/include/pybind11/numpy.h>

#include "hoomd/extern/Eigen/Eigen/Dense"

using namespace std;
using namespace Eigen;
namespace py = pybind11;

typedef std::complex<float> cfloat;
#define PI 3.141592653589793f

#define CHECK_CLOSE(a,b,c) UP_ASSERT((std::abs((a)-(b)) <= ((c) * std::abs(a))) && (std::abs((a)-(b)) <= ((c) * std::abs(b))))
#define CHECK_SMALL(a,c) UP_ASSERT(std::abs(a) < c)

int main(int argc, char **argv)
{
    return upp11::TestMain().main(argc, argv);
}

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

//! Apply the thermostat to 1000 particles in an ideal gas
template <class CollectiveMode>
void collective_mode_mobility_gpu(std::shared_ptr<ExecutionConfiguration> exec_conf, float alpha)
{
    int N = 400;
    int Nk = 8;
    float L = 3;

    MatrixXf ks_part(Nk/2, 3);
    MatrixXf ks(Nk, 3);

    ks_part <<   3, 0, 0,
            0, 2, 0,
            0, 0, 1,
            0, -1, 4;

    ks << ks_part, -ks_part;

    MatrixXf pos = (L/2) * MatrixXf::Random(N, 3);
    MatrixXf emp_M = MatrixXf::Zero(3*N, 3*N);
    MatrixXf M(3*N, 3*N);

    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(L), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    set_positions_from_matrix(pdata, pos);

    // ------------ calculate real mobility matrix analytically -----------------------
    
    MatrixXf A = MatrixXf::Identity(3*N, 3*N);
    MatrixXcf B = MatrixXf::Zero(3*N, 3*Nk);
    MatrixXcf dft_coeffs = MatrixXf::Zero(N, Nk);
    cfloat im(0,1);

    for (int j = 0; j < Nk; j++)
    {
        Matrix3f I_kk = Matrix3f::Identity() - ks.row(j).normalized().transpose() * ks.row(j).normalized();

        for (int i = 0; i < N; i++)
        {
            A.block(3*i,3*i,3,3) -= alpha/Nk * I_kk;

            B.block(3*i,3*j,3,3) = exp(2*PI*im*(pos.row(i) * ks.row(j).transpose())(0)/L) * I_kk;
            dft_coeffs(i,j) = exp(2*PI*im*(pos.row(i) * ks.row(j).transpose())(0)/L);
        }
    }

    MatrixXcf BBT = B * B.adjoint();

    M = A + alpha/Nk * BBT.real();

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

            
    		emp_M(3*ii, i) = wrap_box(newpos.x - pos(ii, 0), L);
    		emp_M(3*ii+1, i) = wrap_box(newpos.y - pos(ii, 1), L);
    		emp_M(3*ii+2, i) = wrap_box(newpos.z - pos(ii, 2), L);
    	}
    }

    UP_ASSERT((emp_M - M).array().abs().maxCoeff() < 1e-5);
}

template <class CollectiveMode>
void collective_mode_random_trace(std::shared_ptr<ExecutionConfiguration> exec_conf, float alpha)
{
    int N = 500;
    int Nk = 8;
    float L = 3;
    int num_steps = 1000;
    float T_val = 0.7;

    MatrixXf ks_part(Nk/2, 3);
    MatrixXf ks(Nk, 3);

    ks_part <<   3, 0, 0,
            0, 2, 0,
            0, 0, 1,
            0, -1, 4;

    ks << ks_part, -ks_part;

    MatrixXf pos = (L/2) * MatrixXf::Random(N, 3);

    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(L), 1, 0, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));

    set_positions_from_matrix(pdata, pos);

    Scalar dt = Scalar(0.01); // dt needs to be 1 in order to extract mobility matrix
    shared_ptr<VariantConst> T(new VariantConst(T_val));

    shared_ptr<CollectiveModeGPU> cmbd(new CollectiveModeGPU(sysdef, group_all, T, 234, ks_part, alpha));
    shared_ptr<IntegratorTwoStep> cmbd_up(new IntegratorTwoStep(sysdef, dt));
    cmbd_up->addIntegrationMethod(cmbd);
    cmbd_up->prepRun(0);

    VectorXf disp(3*N);
    VectorXf prevx(3*N);

    float avg_trace = 0.0;
    MatrixXf cov = MatrixXf::Zero(3*N,3*N);

    // fill prevx with initial values
    for (int ii = 0; ii < N; ii++)
	{
		prevx(3*ii) = pos(ii, 0);
		prevx(3*ii+1) = pos(ii, 1);
		prevx(3*ii+2) = pos(ii, 2);
	}

    // step forward in time and calculate mean square displacement
    for (int i = 0; i < num_steps; i++)
    {
    	cmbd_up->update(i);

    	for (int ii = 0; ii < N; ii++)
    	{
    		Scalar3 newpos = pdata->getPosition(ii);
            
            // calculate displacements
    		disp(3*ii) = wrap_box(newpos.x - prevx(3*ii), L);
    		disp(3*ii+1) = wrap_box(newpos.y - prevx(3*ii+1), L);
    		disp(3*ii+2) = wrap_box(newpos.z - prevx(3*ii+2), L);

    		// update prevx with current positions
    		prevx(3*ii) = newpos.x;
			prevx(3*ii+1) = newpos.y;
			prevx(3*ii+2) = newpos.z;
    	}

    	avg_trace += disp.squaredNorm() / num_steps;

    	cov += (disp * disp.transpose()) / num_steps;
    }

    UP_ASSERT(fabs(avg_trace / (6*T_val*dt*N) - 1.0) < 0.01);

    if (alpha == 0.0)
    {
    	UP_ASSERT((cov/(2*T_val*dt) - MatrixXf::Identity(3*N, 3*N)).array().abs().maxCoeff() < 0.5);
    }
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

UP_TEST( CMBD_Finite_Temperature_Self_GPU )
{
    collective_mode_random_trace<CollectiveModeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)), 0.0);
}

UP_TEST( CMBD_Finite_Temperature_Full_GPU )
{
    collective_mode_random_trace<CollectiveModeGPU>(std::shared_ptr<ExecutionConfiguration>(new ExecutionConfiguration(ExecutionConfiguration::GPU)), 0.95);
}
#endif
