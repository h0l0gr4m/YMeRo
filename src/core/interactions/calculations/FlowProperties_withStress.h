//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#pragma once

#include <core/interactions/calculations/FlowProperties.h>
#include <core/datatypes.h>
#include <core/interactions/accumulators/forceStressFlowProperties.h>
#include <core/utils/common.h>

#ifndef __NVCC__
float fastPower(float x, float a)
{
    return pow(x, a);
}
#else
#include <core/utils/cuda_common.h>
#endif



class LocalParticleVector;
class CellList;

template<typename BasicPairwiseForce>
class FlowProperties_withStress
{
public:

    using BasicViewType = typename BasicPairwiseForce::ViewType;
    using ViewType      = PVviewWithStressFlowProperties<BasicViewType>;
    using ParticleType  = typename BasicPairwiseForce::ParticleType;

    FlowProperties_withStress(BasicPairwiseForce basicForce) :
        basicForce(basicForce)
    {}

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2, CellList *cl1, CellList *cl2, float t)
    {
        basicForce.setup(lpv1, lpv2, cl1, cl2, t);
    }

    __D__ inline ParticleType read(const ViewType& view, int id) const                     { return        basicForce.read(view, id); }
    __D__ inline ParticleType readNoCache(const ViewType& view, int id) const              { return basicForce.readNoCache(view, id); }
    __D__ inline void readCoordinates(ParticleType& p, const ViewType& view, int id) const { basicForce.readCoordinates(p, view, id); }
    __D__ inline void readExtraData  (ParticleType& p, const ViewType& view, int id) const { basicForce.readExtraData  (p, view, id); }
    __D__ inline bool withinCutoff(const ParticleType& src, const ParticleType& dst) const { return basicForce.withinCutoff(src, dst);}
    __D__ inline float3 getPosition(const ParticleType& p) const {return basicForce.getPosition(p);}

    __device__ inline ForceStressFlowProperty operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        float3 dr = getPosition(dst) - getPosition(src);
        float3 du = dst.u - src.u;
        float3 f  = basicForce(dst, dstId, src, srcId);
        const float rij2 = dot(dr, dr);
        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float Vj  = 1.0/6.0;
        const float q = invrij*Vj*der_eta_kernel(rij);

        Stress s;
        Vorticity vorticity;
        Aprox_Density aprox_density;
        Velocity_Gradient velocity_gradient;

        s.xx = 0.5f * dr.x * f.x;
        s.xy = 0.5f * dr.x * f.y;
        s.xz = 0.5f * dr.x * f.z;
        s.yy = 0.5f * dr.y * f.y;
        s.yz = 0.5f * dr.y * f.z;
        s.zz = 0.5f * dr.z * f.z;

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

        aprox_density.x = symmetry_function(rij,0.5,1)*eta_kernel(rij);
        aprox_density.y = symmetry_function(rij,0.1,1)*eta_kernel(rij);
        aprox_density.z = symmetry_function(rij,0.9,0.5)*eta_kernel(rij);


        return {f,s,aprox_density,vorticity,velocity_gradient};
    }

    __D__ inline ForceStressFlowPropertyAccumulator<BasicViewType> getZeroedAccumulator() const {return ForceStressFlowPropertyAccumulator<BasicViewType>();}

private:

    BasicPairwiseForce basicForce;
};
