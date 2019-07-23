#pragma once

#include <core/datatypes.h>
#include <core/utils/cuda_rng.h>

#include <core/utils/cpu_gpu_defines.h>
#include <core/utils/helper_math.h>

#include <random>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/interactions/pairwise_interactions/fetchers.h>

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




class NNInput_Computation : public ParticleFetcherWithVelocityandFlowProperties
{
public:
  using ViewType     = PVviewWithFlowProperties;
  using ParticleType = ParticleWithFlowProperties;

    NNInput_Computation(float rc,float viscosity,std::string nninputs_name = "NNInputs") :
        ParticleFetcherWithVelocityandFlowProperties(rc),nninputs_name(nninputs_name),viscosity(viscosity)
    {}

    void setup(LocalParticleVector *lpv1)
    {
      pv1NNInput = lpv1->dataPerParticle.getData<NNInput>(ChannelNames::NNInputs)->devPtr();
    }

    __D__ inline void operator()(const ParticleType dst, int dstId) const
    {

      //calculate 6 invariances of the gradient of u (according to paper xxx)
      //calculate symmetrix part of velocity_gradient
      const float sxy = (dst.velocity_gradient.xy+dst.velocity_gradient.yx)/2;
      const float sxz = (dst.velocity_gradient.xz+dst.velocity_gradient.zx)/2;
      const float syz = (dst.velocity_gradient.yz+dst.velocity_gradient.zy)/2;

      //calculate anti-symmetric particle_vector
      const float rxy = (dst.velocity_gradient.xy-dst.velocity_gradient.yx)/2;
      const float rxz = (dst.velocity_gradient.xz-dst.velocity_gradient.zx)/2;
      const float ryx = (dst.velocity_gradient.yx-dst.velocity_gradient.xy)/2;
      const float ryz = (dst.velocity_gradient.yz-dst.velocity_gradient.zy)/2;
      const float rzx = (dst.velocity_gradient.zx-dst.velocity_gradient.xz)/2;
      const float rzy = (dst.velocity_gradient.zy-dst.velocity_gradient.yz)/2;

      //calculate s²
      const float s2xx = dst.velocity_gradient.xx*dst.velocity_gradient.xx+sxy*sxy+sxz*sxz;
      const float s2xy = dst.velocity_gradient.xx*sxy + dst.velocity_gradient.yy*sxy + sxz*syz;
      const float s2xz = dst.velocity_gradient.xx*sxz + sxy*syz+sxz*dst.velocity_gradient.zz;
      const float s2yy = sxy*sxy+dst.velocity_gradient.yy*dst.velocity_gradient.yy+ syz*syz;
      const float s2yz = sxy * sxz + dst.velocity_gradient.yy*syz + syz*dst.velocity_gradient.zz;
      const float s2zz = sxz*sxz +syz*syz + dst.velocity_gradient.zz*dst.velocity_gradient.zz;

      //calculate r²
      const float r2xx = rxy*ryx+rxz*rzx;
      const float r2xy = rxz*rzy;
      const float r2xz = rxy*ryz;
      const float r2yx = ryz*rzx;
      const float r2yy = ryx*rxy + ryz*rzy;
      const float r2yz = ryx *rxz;
      const float r2zx = rzy*ryx;
      const float r2zy = rzx * rxy;
      const float r2zz = rzx* rxz + rzy*ryz;

      //calculate s³
      const float s3xx = s2xx *dst.velocity_gradient.xx + s2xy*sxy+s2xz*sxz;
      const float s3yy = s2xy * sxy + dst.velocity_gradient.yy * s2yy + s2yz*syz;
      const float s3zz = s2xz * sxz + s2yz * syz + dst.velocity_gradient.zz*s2zz;

      //calculate terms from R²S (only trace)
      const float r2sxx = r2xx*dst.velocity_gradient.xx +r2xy*sxy + r2xz*sxz;
      const float r2syy = r2yx*sxy +r2yy*dst.velocity_gradient.yy+r2yz*syz;
      const float r2szz = r2zx *sxz + r2zy*syz + r2zz*dst.velocity_gradient.zz;

      //calculat terms from R²S² (only trace)
      const float r2s2xx = r2xx*s2xx + r2xy*s2xy + r2xz*s2xz;
      const float r2s2yy = r2yx*s2xy + r2yy*s2yy + r2yz*s2yz;
      const float r2s2zz = r2zx*s2xz + r2zy*s2yz + r2zz*s2zz;

      //calculate trace of s, s² ,r² , s³ ,r²s, r²s³
      pv1NNInput[dstId].i1 = dst.velocity_gradient.xx+dst.velocity_gradient.yy+dst.velocity_gradient.zz;
      pv1NNInput[dstId].i2 = s2xx+s2yy+s2zz;
      pv1NNInput[dstId].i3 = r2xx+r2yy+r2zz;
      pv1NNInput[dstId].i4 = s3xx+s3yy+s3zz;
      pv1NNInput[dstId].i5 = r2sxx + r2syy + r2szz;
      pv1NNInput[dstId].i6 = r2s2xx + r2s2yy + r2s2zz;


      // calculate length of the vorticity
      float3 vor = make_float3(0.0);
      vor.x = dst.vorticity.x;
      vor.y = dst.vorticity.y;
      vor.z = dst.vorticity.z;
      pv1NNInput[dstId].v1 = length(vor);

      //calculate difference densities
      pv1NNInput[dstId].d1 = dst.aprox_density.x;
      pv1NNInput[dstId].d2 = dst.aprox_density.y;
      pv1NNInput[dstId].d3 = dst.aprox_density.z;
      pv1NNInput[dstId].b1 = 1;
      pv1NNInput[dstId].vis = viscosity;

   }
private:
      NNInput *pv1NNInput, *pv2NNInput;
      std::string nninputs_name;
      float viscosity;
};
