#pragma once

#include <core/datatypes.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/cuda_common.h>
#include <core/utils/helper_math.h>

struct flowproperties
{
    Aprox_Density aprox_density;
    Vorticity vorticity;
    Velocity_Gradient velocity_gradient;
    Density_Gradient density_gradient;
};


class FlowPropertyAccumulator
{
public:

    __D__ inline FlowPropertyAccumulator() :
        flowpro({
          {0.f, 0.f, 0.f},
          {0.f, 0.f, 0.f},
          {0.f, 0.f, 0.f,0.f, 0.f, 0.f,0.f, 0.f, 0.f},
          {0.f, 0.f, 0.f}
                })
    {}

    __D__ inline void atomicAddToDst(const flowproperties& fsfp, PVviewWithFlowProperties& view, int id) const
    {
        atomicAddAprox_Density(view.aprox_densities + id, fsfp.aprox_density);
        atomicAddVorticity(view.vorticities + id, fsfp.vorticity);
        atomicAddVelocity_Gradient(view.velocity_gradients + id , fsfp.velocity_gradient);
        atomicAddDensity_Gradient(view.density_gradients + id, fsfp.density_gradient);

    }

    __D__ inline void atomicAddToSrc(const flowproperties& fsfp, PVviewWithFlowProperties& view, int id) const
    {
        atomicAddAprox_Density(view.aprox_densities + id, fsfp.aprox_density);
        atomicAddVorticity(view.vorticities + id, fsfp.vorticity);
        atomicAddVelocity_Gradient(view.velocity_gradients + id , fsfp.velocity_gradient);
        atomicAddDensity_Gradient(view.density_gradients + id, fsfp.density_gradient);

    }

    __D__ inline flowproperties get() const {return flowpro;}

    __D__ inline void add(const flowproperties& fsfp)
    {
        flowpro.aprox_density.x += fsfp.aprox_density.x;
        flowpro.aprox_density.y += fsfp.aprox_density.y;
        flowpro.aprox_density.z += fsfp.aprox_density.z;

        flowpro.vorticity.x += fsfp.vorticity.x;
        flowpro.vorticity.y += fsfp.vorticity.y;
        flowpro.vorticity.z += fsfp.vorticity.z;

        flowpro.velocity_gradient.xx += fsfp.velocity_gradient.xx;
        flowpro.velocity_gradient.xy += fsfp.velocity_gradient.xy;
        flowpro.velocity_gradient.xz += fsfp.velocity_gradient.xz;
        flowpro.velocity_gradient.yx += fsfp.velocity_gradient.yx;
        flowpro.velocity_gradient.yy += fsfp.velocity_gradient.yy;
        flowpro.velocity_gradient.yz += fsfp.velocity_gradient.yz;
        flowpro.velocity_gradient.zx += fsfp.velocity_gradient.zx;
        flowpro.velocity_gradient.zy += fsfp.velocity_gradient.zy;
        flowpro.velocity_gradient.zz += fsfp.velocity_gradient.zz;

        flowpro.density_gradient.x += fsfp.density_gradient.x;
        flowpro.density_gradient.y += fsfp.density_gradient.y;
        flowpro.density_gradient.z += fsfp.density_gradient.z;
    }

private:
    flowproperties flowpro;

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


    __D__ inline void atomicAddDensity_Gradient(Density_Gradient *dst, const Density_Gradient& s) const
    {
        atomicAdd(&dst->x, s.x);
        atomicAdd(&dst->y, s.y);
        atomicAdd(&dst->z, s.z);
    }
};
