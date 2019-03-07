#pragma once

#include <core/datatypes.h>
#include "fetchers.h"

#include <core/interactions/accumulators/force.h>


#include <random>


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




class Pairwise_SmartDPD : public ParticleFetcherWithVelocity
{
public:

  using ViewType     = PVview;
  using ParticleType = Particle;

    Pairwise_SmartDPD(std::string parameterName,float rc, float a, float gamma, float kbT, float dt, float power) :
        ParticleFetcherWithVelocity(rc), parameterName(parameterName), a(a), gamma(gamma), power(power),kbT(kbT),dt(dt)
    {
        rc2 = rc*rc;
        invrc = 1.0 / rc;
    }

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2, float t)
    {
        // seed = t;
        // better use random seed (time-based) instead of time
        // time-based is IMPORTANT for momentum conservation!!
        // t is float, use it's bit representation as int to seed RNG
        int v = *((int*)&t);
        std::mt19937 gen(v);
        std::uniform_real_distribution<float> udistr(0.001, 1);
        seed = udistr(gen);
        pv1DPDparameter = lpv1->extraPerParticle.getData<DPDparameter>(ChannelNames::DPDparameters)->devPtr();
        pv2DPDparameter = lpv2->extraPerParticle.getData<DPDparameter>(ChannelNames::DPDparameters)->devPtr();

    }

    __D__ inline float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const
    {
        const float alpha_p = (pv1DPDparameter[dstId].alpha_p + pv2DPDparameter[srcId].alpha_p)/2;
        const float gamma_p = (pv1DPDparameter[dstId].gamma_p + pv2DPDparameter[srcId].gamma_p)/2;
        float sigma_p = sqrt(2 * gamma_p * kbT / dt);
        const float3 dr = dst.r - src.r;
        const float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij * invrc;
        const float wr = fastPower(argwr, power);
        const float3 dr_r = dr * invrij;
        const float3 du = dst.u - src.u;
        const float rdotv = dot(dr_r, du);

        const float myrandnr = Logistic::mean0var1(seed, min(src.i1, dst.i1), max(src.i1, dst.i1));

        const float strength = alpha_p * argwr - (gamma_p * wr * rdotv + sigma_p * myrandnr) * wr;
        return dr_r * strength;
    }

        __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

public:
    std::string parameterName;
    DPDparameter *pv1DPDparameter, *pv2DPDparameter;
    float a, gamma, power, rc,kbT,dt;
    float invrc, rc2;
    float seed;
};
