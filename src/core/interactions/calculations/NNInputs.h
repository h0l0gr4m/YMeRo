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

    NNInput_Computation(float rc,std::string nninputs_name = "NNInputs") :
        ParticleFetcherWithVelocityandFlowProperties(rc),nninputs_name(nninputs_name)
    {}

    void setup(LocalParticleVector *lpv1)
    {
      pv1NNInput = lpv1->extraPerParticle.getData<NNInput>(ChannelNames::NNInputs)->devPtr();
    }

    __D__ inline void operator()(const ParticleType dst, int dstId) const
    {

      //calculate 3 invariances of the gradient of u
      pv1NNInput[dstId].g1 = dst.velocity_gradient.xx+dst.velocity_gradient.yy+dst.velocity_gradient.zz;

      const float A11A22 = dst.velocity_gradient.xx*dst.velocity_gradient.yy;
      const float A11A33 = dst.velocity_gradient.xx*dst.velocity_gradient.zz;
      const float A33A22 = dst.velocity_gradient.zz*dst.velocity_gradient.yy;
      const float A12A21 = dst.velocity_gradient.xy*dst.velocity_gradient.yx;
      const float A23A32 = dst.velocity_gradient.yz*dst.velocity_gradient.zy;
      const float A13A31 = dst.velocity_gradient.xz*dst.velocity_gradient.zx;
      const float A13A22A31 = A13A31*dst.velocity_gradient.yy;
      const float A12A23A31 = dst.velocity_gradient.xy*dst.velocity_gradient.yz*dst.velocity_gradient.zx;
      const float A13A21A32 = dst.velocity_gradient.xz*dst.velocity_gradient.yx*dst.velocity_gradient.zy;
      const float A11A23A32 = dst.velocity_gradient.xx * A23A32;
      const float A12A21A33 = A12A21 * dst.velocity_gradient.zz;
      const float A11A22A33 = A11A22*dst.velocity_gradient.zz;

      pv1NNInput[dstId].g2 = A11A22+A11A33+A33A22-A12A21-A23A32-A13A31;
      pv1NNInput[dstId].g3 = -A13A22A31+A12A23A31+A13A21A32-A11A23A32-A12A21A33+A11A22A33;

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


   }
private:
      NNInput *pv1NNInput, *pv2NNInput;
      std::string nninputs_name;
};
