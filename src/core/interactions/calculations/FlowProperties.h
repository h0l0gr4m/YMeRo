#pragma once

#include <core/datatypes.h>
#include <core/utils/cuda_rng.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <random>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>

class CellList;
class LocalParticleVector;

#ifndef __NVCC__
float fastPower(float x, float a)
{
    return pow(x, a);
}
#else
#include <core/utils/cuda_common.h>
#endif



inline __device__ float der_eta_kernel(float r)
{
	float q = 0.574*(-8*fastPower(r,3)+8*fastPower(r,7));
	return q;
}

inline __device__ float eta_kernel(float r)
{
	float q = 0.574*(-2*fastPower(r,4)+fastPower(r,8));
	return q;
}


inline __device__ float symmetry_function(float r, float eta , float R_s)
{
  float x = fastPower((r-R_s),2);
	float q = -eta*x;
  float y = exp(q);
	return y;
}






class FlowProperties
{
public:
    FlowProperties(std::string fp_name = "fp_name") :
        fp_name(fp_name)
    {}




    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {

      pv1Aprox_Density = lpv1->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->devPtr();
      pv2Aprox_Density = lpv2->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->devPtr();

      pv1Vorticity = lpv1->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();
      pv2Vorticity = lpv2->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();

      pv1Velocity_Gradient = lpv1->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
      pv2Velocity_Gradient = lpv2->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
    }

    __D__ inline void operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
        const float3 dr = dst.r - src.r;
        const float rij = length(dr);
        const float invrij = 1.0/rij;
        const float3 dstu = dst.u;
        const float3 srcu = src.u;
      	const float dx = dr.x;
      	const float dy = dr.y;
      	const float dz = dr.z;
      	// float dux = dstu.x-srcu.x;
      	// float duy = dstu.y-srcu.y;
        // float duz = dstu.z-srcu.z;

        float dux = srcu.x;
        float duy = srcu.y;
        float duz = srcu.z;

      	const float Vj  = 1.0/6.0;
        const float q = invrij*Vj*der_eta_kernel(rij);

        //calculate velocity gradient matrix

        const float dstVelocity_Gradientxx = - q*dux*dx;
        const float dstVelocity_Gradientxy = - q*dux*dy;
        const float dstVelocity_Gradientxz = - q*dux*dz;
        const float dstVelocity_Gradientyy = - q*duy*dy;
        const float dstVelocity_Gradientyx = - q*duy*dx;
        const float dstVelocity_Gradientyz = - q*duy*dz;
        const float dstVelocity_Gradientzx = - q*duz*dx;
        const float dstVelocity_Gradientzy = - q*duz*dy;
        const float dstVelocity_Gradientzz = - q*duz*dz;


        atomicAdd(&pv1Velocity_Gradient[dstId].xx ,dstVelocity_Gradientxx);
        atomicAdd(&pv1Velocity_Gradient[dstId].xy ,dstVelocity_Gradientxy);
        atomicAdd(&pv1Velocity_Gradient[dstId].xz ,dstVelocity_Gradientxz);
        atomicAdd(&pv1Velocity_Gradient[dstId].yx ,dstVelocity_Gradientyx);
        atomicAdd(&pv1Velocity_Gradient[dstId].yy ,dstVelocity_Gradientyy);
        atomicAdd(&pv1Velocity_Gradient[dstId].yz ,dstVelocity_Gradientyz);
        atomicAdd(&pv1Velocity_Gradient[dstId].zx ,dstVelocity_Gradientzx);
        atomicAdd(&pv1Velocity_Gradient[dstId].zy ,dstVelocity_Gradientzy);
        atomicAdd(&pv1Velocity_Gradient[dstId].zz ,dstVelocity_Gradientzz);

        //caluclate vorticity vector
        const float dstVorticityx = -q*(duz*dy-duy*dz);
        const float dstVorticityy = -q*(dux*dz-duz*dx);
        const float dstVorticityz = -q*(duy*dx-dux*dz);

        atomicAdd(&pv1Vorticity[dstId].x,dstVorticityx);
        atomicAdd(&pv1Vorticity[dstId].y,dstVorticityy);
        atomicAdd(&pv1Vorticity[dstId].z,dstVorticityz);

        // calculate density (similar to density) via symmetry functions
        const float d_particle = symmetry_function(rij,2,0.1)*eta_kernel(rij);
        atomicAdd(&pv1Aprox_Density[dstId].x,d_particle);

   }
private:
      Aprox_Density *pv1Aprox_Density, *pv2Aprox_Density;
      Vorticity *pv1Vorticity, *pv2Vorticity;
      Velocity_Gradient *pv1Velocity_Gradient, *pv2Velocity_Gradient;
      std::string fp_name;
};
