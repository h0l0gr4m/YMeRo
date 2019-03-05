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
  const float x = fastPower((r-R_s),2);
	const float q = -eta*x;
  const float y = exp(q);
	return y;
}





template<typename BasicPairwiseForce>
class FlowProperties
{
public:
    FlowProperties(std::string fp_name,BasicPairwiseForce basicForce) :
        fp_name(fp_name),basicForce(basicForce)
    {}


    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {
      basicForce.setup(lpv1, lpv2, cl1, cl2, t);

      pv1Aprox_Density = lpv1->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->devPtr();
      pv2Aprox_Density = lpv2->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->devPtr();

      pv1Vorticity = lpv1->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();
      pv2Vorticity = lpv2->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();

      pv1Velocity_Gradient = lpv1->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
      pv2Velocity_Gradient = lpv2->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();

      pv1Stress = lpv1->extraPerParticle.getData<Stress>("stressDPD")->devPtr();
      pv2Stress = lpv2->extraPerParticle.getData<Stress>("stressDPD")->devPtr();

    }

    __D__ inline float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
        float3 f = basicForce(dst, dstId, src, srcId);

        const float3 dr = dst.r - src.r;
        const float rij = length(dr);
        const float invrij = 1.0/rij;
        const float3 du = dst.u-src.u;
      	const float Vj  = 1.0/6.0;
        const float q = invrij*Vj*der_eta_kernel(rij);

        // calculate velocity gradient matrix

        const float dstVelocity_Gradientxx = - q*du.x*dr.x;
        const float dstVelocity_Gradientxy = - q*du.x*dr.y;
        const float dstVelocity_Gradientxz = - q*du.x*dr.z;
        const float dstVelocity_Gradientyy = - q*du.y*dr.y;
        const float dstVelocity_Gradientyx = - q*du.y*dr.x;
        const float dstVelocity_Gradientyz = - q*du.y*dr.z;
        const float dstVelocity_Gradientzx = - q*du.z*dr.x;
        const float dstVelocity_Gradientzy = - q*du.z*dr.y;
        const float dstVelocity_Gradientzz = - q*du.z*dr.z;

        atomicAdd(&pv1Velocity_Gradient[dstId].xx ,dstVelocity_Gradientxx);
        atomicAdd(&pv1Velocity_Gradient[dstId].xy ,dstVelocity_Gradientxy);
        atomicAdd(&pv1Velocity_Gradient[dstId].xz ,dstVelocity_Gradientxz);
        atomicAdd(&pv1Velocity_Gradient[dstId].yx ,dstVelocity_Gradientyx);
        atomicAdd(&pv1Velocity_Gradient[dstId].yy ,dstVelocity_Gradientyy);
        atomicAdd(&pv1Velocity_Gradient[dstId].yz ,dstVelocity_Gradientyz);
        atomicAdd(&pv1Velocity_Gradient[dstId].zx ,dstVelocity_Gradientzx);
        atomicAdd(&pv1Velocity_Gradient[dstId].zy ,dstVelocity_Gradientzy);
        atomicAdd(&pv1Velocity_Gradient[dstId].zz ,dstVelocity_Gradientzz);


        atomicAdd(&pv2Velocity_Gradient[srcId].xx ,dstVelocity_Gradientxx);
        atomicAdd(&pv2Velocity_Gradient[srcId].xy ,dstVelocity_Gradientxy);
        atomicAdd(&pv2Velocity_Gradient[srcId].xz ,dstVelocity_Gradientxz);
        atomicAdd(&pv2Velocity_Gradient[srcId].yx ,dstVelocity_Gradientyx);
        atomicAdd(&pv2Velocity_Gradient[srcId].yy ,dstVelocity_Gradientyy);
        atomicAdd(&pv2Velocity_Gradient[srcId].yz ,dstVelocity_Gradientyz);
        atomicAdd(&pv2Velocity_Gradient[srcId].zx ,dstVelocity_Gradientzx);
        atomicAdd(&pv2Velocity_Gradient[srcId].zy ,dstVelocity_Gradientzy);
        atomicAdd(&pv2Velocity_Gradient[srcId].zz ,dstVelocity_Gradientzz);

        //caluclate vorcicity vector
        const float dstVorticityx = -q*(du.z*dr.y-du.y*dr.z);
        const float dstVorticityy = -q*(du.x*dr.z-du.z*dr.x);
        const float dstVorticityz = -q*(du.y*dr.x-du.x*dr.z);



        atomicAdd(&pv1Vorticity[dstId].x,dstVorticityx);
        atomicAdd(&pv1Vorticity[dstId].y,dstVorticityy);
        atomicAdd(&pv1Vorticity[dstId].z,dstVorticityz);

        atomicAdd(&pv2Vorticity[srcId].x,dstVorticityx);
        atomicAdd(&pv2Vorticity[srcId].y,dstVorticityy);
        atomicAdd(&pv2Vorticity[srcId].z,dstVorticityz);
        // calculate density (similar to density) via symmetry functions
        float3 d_particle;
        d_particle.x = symmetry_function(rij,0.5,1)*eta_kernel(rij);
        d_particle.y = symmetry_function(rij,0.1,1)*eta_kernel(rij);
        d_particle.z = symmetry_function(rij,0.9,0.5)*eta_kernel(rij);
        atomicAdd(&pv1Aprox_Density[dstId].x,d_particle.x);
        atomicAdd(&pv1Aprox_Density[dstId].y,d_particle.y);
        atomicAdd(&pv1Aprox_Density[dstId].z,d_particle.z);

        atomicAdd(&pv2Aprox_Density[srcId].x,d_particle.x);
        atomicAdd(&pv2Aprox_Density[srcId].y,d_particle.y);
        atomicAdd(&pv2Aprox_Density[srcId].z,d_particle.z);




        return f;

   }
private:
      std::string fp_name;
      BasicPairwiseForce basicForce;

      Aprox_Density *pv1Aprox_Density, *pv2Aprox_Density;
      Vorticity *pv1Vorticity, *pv2Vorticity;
      Velocity_Gradient *pv1Velocity_Gradient, *pv2Velocity_Gradient;
      Stress *pv1Stress, *pv2Stress;

};
