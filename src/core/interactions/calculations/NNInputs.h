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








class NNInputs
{
public:
    NNInputs(std::string nninputs_name = "NNInputs") :
        nninputs_name(nninputs_name)
    {}




    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {
      pv1NNInput= lpv1->extraPerParticle.getData<NNInput>("NNInputs")->devPtr();
      pv2NNInput = lpv2->extraPerParticle.getData<NNInput>("NNInputs")->devPtr();
      pv1Vorticity = lpv1->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();
      pv2Vorticity = lpv2->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();
      pv1Velocity_Gradient = lpv1->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
      pv2Velocity_Gradient = lpv2->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
      pv1Aprox_Density = lpv1->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->devPtr();
      pv2Aprox_Density = lpv2->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->devPtr();

    }

    __D__ inline void operator()(const Particle dst, int dstId) const
    {
      pv1NNInput[dstId].g1 = pv1Velocity_Gradient[dstId].xx+pv1Velocity_Gradient[dstId].yy+pv1Velocity_Gradient[dstId].zz;
      const float A11A22 = pv1Velocity_Gradient[dstId].xx*pv1Velocity_Gradient[dstId].yy;
      const float A11A33 = pv1Velocity_Gradient[dstId].xx*pv1Velocity_Gradient[dstId].zz;
      const float A33A22 = pv1Velocity_Gradient[dstId].zz*pv1Velocity_Gradient[dstId].yy;
      const float A12A21 = pv1Velocity_Gradient[dstId].xy*pv1Velocity_Gradient[dstId].yx;
      const float A23A32 = pv1Velocity_Gradient[dstId].yz*pv1Velocity_Gradient[dstId].zy;
      const float A13A31 = pv1Velocity_Gradient[dstId].xz*pv1Velocity_Gradient[dstId].zx;
      const float A13A22A31 = A13A31*pv1Velocity_Gradient[dstId].yy;
      const float A12A23A31 = pv1Velocity_Gradient[dstId].xy*pv1Velocity_Gradient[dstId].yz*pv1Velocity_Gradient[dstId].zx;
      const float A13A21A32 = pv1Velocity_Gradient[dstId].xz*pv1Velocity_Gradient[dstId].yx*pv1Velocity_Gradient[dstId].zy;
      const float A11A23A32 = pv1Velocity_Gradient[dstId].xx * A23A32;
      const float A12A21A33 = A12A21 * pv1Velocity_Gradient[dstId].zz;
      const float A11A22A33 = A11A22*pv1Velocity_Gradient[dstId].zz;
      pv1NNInput[dstId].g2 = A11A22+A11A33+A33A22-A12A21-A23A32-A13A31;
      pv1NNInput[dstId].g3 = -A13A22A31+A12A23A31+A13A21A32-A11A23A32-A12A21A33+A11A22A33;
      float3 vor = make_float3(0.0);
      vor.x = pv1Vorticity[dstId].x;
      vor.y = pv1Vorticity[dstId].y;
      vor.z = pv1Vorticity[dstId].z;
      pv1NNInput[dstId].v1 = length(vor);
      pv1NNInput[dstId].d1 = pv1Aprox_Density[dstId].x;
      pv1NNInput[dstId].d2 = pv1Aprox_Density[dstId].y;
      pv1NNInput[dstId].d3 = pv1Aprox_Density[dstId].z;
      pv1NNInput[dstId].b1 = 1;
   }
private:
      NNInput *pv1NNInput, *pv2NNInput;
      Vorticity *pv1Vorticity, *pv2Vorticity;
      Aprox_Density *pv1Aprox_Density, *pv2Aprox_Density;
      Velocity_Gradient *pv1Velocity_Gradient, *pv2Velocity_Gradient;
      std::string nninputs_name;
};
