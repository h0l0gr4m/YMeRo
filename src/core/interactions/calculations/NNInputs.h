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
      pv1Div = lpv1->extraPerParticle.getData<Divergence>("div_name")->devPtr();
      pv2Div = lpv2->extraPerParticle.getData<Divergence>("div_name")->devPtr();
      pv1Vorticity = lpv1->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();
      pv2Vorticity = lpv2->extraPerParticle.getData<Vorticity>("vorticity_name")->devPtr();
      pv1Velocity_Gradient = lpv1->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
      pv2Velocity_Gradient = lpv2->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->devPtr();
    }

    __D__ inline void operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
      pv1NNInput[dstId].d = pv1Div[dstId].div;
      pv1NNInput[dstId].g1 = pv1Div[dstId].div;
      pv1NNInput[dstId].g2 = pv1Div[dstId].div;
      pv1NNInput[dstId].g3 = pv1Div[dstId].div;
      pv1NNInput[dstId].g4 = pv1Div[dstId].div;
      pv1NNInput[dstId].g5 = pv1Div[dstId].div;
      pv1NNInput[dstId].g6 = pv1Div[dstId].div;
      pv1NNInput[dstId].v1 = pv1Div[dstId].div;
      pv1NNInput[dstId].v2 = pv1Div[dstId].div;
      pv1NNInput[dstId].v3 = pv1Div[dstId].div;

   }
private:
      NNInput *pv1NNInput, *pv2NNInput;
      Divergence *pv1Div, *pv2Div;
      Vorticity *pv1Vorticity, *pv2Vorticity;
      Velocity_Gradient *pv1Velocity_Gradient, *pv2Velocity_Gradient;
      std::string nninputs_name;
};
