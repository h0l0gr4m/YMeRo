#pragma once

#include <core/datatypes.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

struct ForceStressFlowProperty
{
    float3 force;
    Stress stress;
    Aprox_Density aprox_density;
    Vorticity vorticity;
    Velocity_Gradient velocity_gradient;
};

template <typename BasicView>
class ForceStressFlowPropertyAccumulator
{
public:

    __D__ inline ForceStressFlowPropertyAccumulator() :
        frcStressFlowProperty({{0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f, 0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f},
                             {0.f, 0.f, 0.f,0.f, 0.f, 0.f,0.f, 0.f, 0.f}})
    {}

    __D__ inline void atomicAddToDst(const ForceStressFlowProperty& fsfp, PVviewWithStressFlowProperties<BasicView>& view, int id) const
    {
        atomicAdd(      view.forces   + id, fsfp.force );
        atomicAddStress(view.stresses + id, fsfp.stress);
        atomicAddStress(view.stresses + id,  fsfp.stress);
        atomicAddAprox_Density(view.aprox_densities + id, fsfp.aprox_density);
        atomicAddVorticity(view.vorticities + id, fsfp.vorticity);
        atomicAddVelocity_Gradient(view.velocity_gradients + id , fsfp.velocity_gradient);
    }

    __D__ inline void atomicAddToSrc(const ForceStressFlowProperty& fsfp, PVviewWithStressFlowProperties<BasicView>& view, int id) const
    {
        atomicAdd( view.forces   + id, -fsfp.force );
        atomicAddStress(view.stresses + id,  fsfp.stress);
        atomicAddAprox_Density(view.aprox_densities + id, fsfp.aprox_density);
        atomicAddVorticity(view.vorticities + id, fsfp.vorticity);
        atomicAddVelocity_Gradient(view.velocity_gradients + id , fsfp.velocity_gradient);
    }

    __D__ inline ForceStressFlowProperty get() const {return frcStressFlowProperty;}

    __D__ inline void add(const ForceStressFlowProperty& fsfp)
    {
        frcStressFlowProperty.force += fsfp.force;

        frcStressFlowProperty.stress.xx += fsfp.stress.xx;
        frcStressFlowProperty.stress.xy += fsfp.stress.xy;
        frcStressFlowProperty.stress.xz += fsfp.stress.xz;
        frcStressFlowProperty.stress.yy += fsfp.stress.yy;
        frcStressFlowProperty.stress.yz += fsfp.stress.yz;
        frcStressFlowProperty.stress.zz += fsfp.stress.zz;

        frcStressFlowProperty.aprox_density.x += fsfp.aprox_density.x;
        frcStressFlowProperty.aprox_density.y += fsfp.aprox_density.y;
        frcStressFlowProperty.aprox_density.z += fsfp.aprox_density.z;

        frcStressFlowProperty.vorticity.x += fsfp.vorticity.x;
        frcStressFlowProperty.vorticity.y += fsfp.vorticity.y;
        frcStressFlowProperty.vorticity.z += fsfp.vorticity.z;

        frcStressFlowProperty.velocity_gradient.xx += fsfp.velocity_gradient.xx;
        frcStressFlowProperty.velocity_gradient.xy += fsfp.velocity_gradient.xy;
        frcStressFlowProperty.velocity_gradient.xz += fsfp.velocity_gradient.xz;
        frcStressFlowProperty.velocity_gradient.yx += fsfp.velocity_gradient.yx;
        frcStressFlowProperty.velocity_gradient.yy += fsfp.velocity_gradient.yy;
        frcStressFlowProperty.velocity_gradient.yz += fsfp.velocity_gradient.yz;
        frcStressFlowProperty.velocity_gradient.zx += fsfp.velocity_gradient.zx;
        frcStressFlowProperty.velocity_gradient.zy += fsfp.velocity_gradient.zy;
        frcStressFlowProperty.velocity_gradient.zz += fsfp.velocity_gradient.zz;
    }

private:
    ForceStressFlowProperty frcStressFlowProperty;

    __D__ inline void atomicAddStress(Stress *dst, const Stress& s) const
    {
        atomicAdd(&dst->xx, s.xx);
        atomicAdd(&dst->xy, s.xy);
        atomicAdd(&dst->xz, s.xz);
        atomicAdd(&dst->yy, s.yy);
        atomicAdd(&dst->yz, s.yz);
        atomicAdd(&dst->zz, s.zz);
    }

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
