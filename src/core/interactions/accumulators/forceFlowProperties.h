#pragma once

#include <core/datatypes.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

struct ForceFlowProperty
{
    float3 force;
    Aprox_Density aprox_density;
    Vorticity vorticity;
    Velocity_Gradient velocity_gradient;
};

template <typename BasicView>
class ForceFlowPropertyAccumulator
{
public:

    __D__ inline ForceFlowPropertyAccumulator() :
        frcFlowProperty({{0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f,0.f, 0.f, 0.f,0.f, 0.f, 0.f}})
    {}

    __D__ inline void atomicAddToDst(const ForceFlowProperty& fsfp, PVviewWithFlowProperties<BasicView>& view, int id) const
    {
        atomicAdd(      view.forces   + id, fsfp.force );
        atomicAddAprox_Density(view.aprox_densities + id, fsfp.aprox_density);
        atomicAddVorticity(view.vorticities + id, fsfp.vorticity);
        atomicAddVelocity_Gradient(view.velocity_gradients + id , fsfp.velocity_gradient);
    }

    __D__ inline void atomicAddToSrc(const ForceFlowProperty& fsfp, PVviewWithFlowProperties<BasicView>& view, int id) const
    {
        atomicAdd( view.forces   + id, -fsfp.force );
        atomicAddAprox_Density(view.aprox_densities + id, fsfp.aprox_density);
        atomicAddVorticity(view.vorticities + id, fsfp.vorticity);
        atomicAddVelocity_Gradient(view.velocity_gradients + id , fsfp.velocity_gradient);
    }

    __D__ inline ForceFlowProperty get() const {return frcFlowProperty;}

    __D__ inline void add(const ForceFlowProperty& fsfp)
    {
        frcFlowProperty.force += fsfp.force;

        frcFlowProperty.aprox_density.x += fsfp.aprox_density.x;
        frcFlowProperty.aprox_density.y += fsfp.aprox_density.y;
        frcFlowProperty.aprox_density.z += fsfp.aprox_density.z;

        frcFlowProperty.vorticity.x += fsfp.vorticity.x;
        frcFlowProperty.vorticity.y += fsfp.vorticity.y;
        frcFlowProperty.vorticity.z += fsfp.vorticity.z;

        frcFlowProperty.velocity_gradient.xx += fsfp.velocity_gradient.xx;
        frcFlowProperty.velocity_gradient.xy += fsfp.velocity_gradient.xy;
        frcFlowProperty.velocity_gradient.xz += fsfp.velocity_gradient.xz;
        frcFlowProperty.velocity_gradient.yx += fsfp.velocity_gradient.yx;
        frcFlowProperty.velocity_gradient.yy += fsfp.velocity_gradient.yy;
        frcFlowProperty.velocity_gradient.yz += fsfp.velocity_gradient.yz;
        frcFlowProperty.velocity_gradient.zx += fsfp.velocity_gradient.zx;
        frcFlowProperty.velocity_gradient.zy += fsfp.velocity_gradient.zy;
        frcFlowProperty.velocity_gradient.zz += fsfp.velocity_gradient.zz;
    }

private:
    ForceFlowProperty frcFlowProperty;

    __D__ inline void atomicAddAprox_Density(Aprox_Density *dst, const Aprox_Density& s) const
    {
        atomicAdd(&dst->x, s.x);
        atomicAdd(&dst->y, s.y);
        atomicAdd(&dst->z, s.z);
    }

    __D__ inline void atomicAddVorticity(Vorticity *dst, const Vorticity& s) const
    {
        atomicAdd(&dst->x, s.x);
        atomicAdd(&dst->y, s.y);
        atomicAdd(&dst->z, s.z);
    }

    __D__ inline void atomicAddVelocity_Gradient(Velocity_Gradient *dst, const Velocity_Gradient& s) const
    {
        atomicAdd(&dst->xx, s.xx);
        atomicAdd(&dst->xy, s.xy);
        atomicAdd(&dst->xz, s.xz);
        atomicAdd(&dst->yx, s.yx);
        atomicAdd(&dst->yy, s.yy);
        atomicAdd(&dst->yz, s.yz);
        atomicAdd(&dst->zx, s.zx);
        atomicAdd(&dst->zy, s.zy);
        atomicAdd(&dst->zz, s.zz);
    }
};
