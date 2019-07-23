//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

#include <core/datatypes.h>
#include <core/interactions/accumulators/accFlowProperties.h>
#include <core/utils/common.h>

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


class LocalParticleVector;
class CellList;


class PairwiseFlowProperties : public ParticleFetcherWithVelocityandFlowProperties , public PairwiseKernel
{
public:
    using ViewType     = PVviewWithFlowProperties;
    using ParticleType = ParticleWithFlowProperties;
    using HandlerType  = PairwiseFlowProperties;


    PairwiseFlowProperties(float rc) :
        ParticleFetcherWithVelocityandFlowProperties(rc),rc(rc)
    {}

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, const YmrState *state)
    {}

    __device__ inline flowproperties operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        float3 dr =dst.p.r - src.p.r;
        float3 du = dst.p.u - src.p.u;
        const float rij2 = dot(dr, dr);
        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float Vj  = 1.0/10.0;
        const float q = invrij*Vj*der_eta_kernel(rij);

        Vorticity vorticity;
        Aprox_Density aprox_density;
        Velocity_Gradient velocity_gradient;
        Density_Gradient density_gradient;

        // calculate velocity gradient matrix

        velocity_gradient.xx = - q*du.x*dr.x;
        velocity_gradient.xy = - q*du.x*dr.y;
        velocity_gradient.xz = - q*du.x*dr.z;
        velocity_gradient.yy = - q*du.y*dr.y;
        velocity_gradient.yx = - q*du.y*dr.x;
        velocity_gradient.yz = - q*du.y*dr.z;
        velocity_gradient.zx = - q*du.z*dr.x;
        velocity_gradient.zy = - q*du.z*dr.y;
        velocity_gradient.zz = - q*du.z*dr.z;


        //caluclate vorcicity vector
        vorticity.x = -q*(du.z*dr.y-du.y*dr.z);
        vorticity.y = -q*(du.x*dr.z-du.z*dr.x);
        vorticity.z = -q*(du.y*dr.x-du.x*dr.z);

        // calculate aprox_densities via symmetry functions
        aprox_density.x = symmetry_function(rij,0.5,1)*eta_kernel(rij);
        aprox_density.y = symmetry_function(rij,0.1,1)*eta_kernel(rij);
        aprox_density.z = symmetry_function(rij,0.9,0.5)*eta_kernel(rij);

        //calculate (weighted) density gradient
        density_gradient.x = q*dr.x;
        density_gradient.y = q*dr.y;
        density_gradient.z = q*dr.z;




        return {aprox_density,vorticity,velocity_gradient,density_gradient};
    }
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }


    __D__ inline FlowPropertyAccumulator getZeroedAccumulator() const {return FlowPropertyAccumulator();}

private:

    float rc;
};
