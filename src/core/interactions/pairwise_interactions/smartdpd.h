#pragma once

#include <core/datatypes.h>
#include "fetchers.h"
#include <core/interactions/utils/step_random_gen.h>
#include <core/ymero_state.h>
#include <core/interactions/accumulators/force.h>
#include <core/utils/restart_helpers.h>


#include <random>
#include "interface.h"


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




class PairwiseSmartDPDHandler : public ParticleFetcherWithVelocityandFlowProperties
{
public:

  using ViewType     = PVviewWithFlowProperties;
  using ParticleType = ParticleWithFlowProperties;

    PairwiseSmartDPDHandler(std::string parameterName,float rc, float a, float gamma, float kbT, float dt, float power) :
        ParticleFetcherWithVelocityandFlowProperties(rc), parameterName(parameterName), a(a), gamma(gamma), power(power),kbT(kbT),dt(dt)
    {
        rc2 = rc*rc;
        invrc = 1.0 / rc;
    }



    __D__ inline float3 operator()(const ParticleType dst, int dstId, const ParticleType src, int srcId) const
    {
        // const float alpha_p = 0;
        // const float gamma_p = 20.25;

        const float alpha_p = (pv1DPDparameter[dstId].alpha_p + pv2DPDparameter[srcId].alpha_p)/2;
        const float gamma_p = (pv1DPDparameter[dstId].gamma_p + pv2DPDparameter[srcId].gamma_p)/2;
        const float sigma_p = sqrt(2 * gamma_p * kbT / dt);
        const float3 dr = dst.p.r - src.p.r;
        const float rij2 = dot(dr, dr);
        if (rij2 > rc2) return make_float3(0.0f);

        const float invrij = rsqrtf(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij * invrc;
        const float wr = fastPower(argwr, power);
        const float3 dr_r = dr * invrij;
        const float3 du = dst.p.u - src.p.u;
        const float rdotv = dot(dr_r, du);

        const float myrandnr = Logistic::mean0var1(seed, min(src.p.i1, dst.p.i1), max(src.p.i1, dst.p.i1));

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



class PairwiseSmartDPD : public PairwiseSmartDPDHandler, public PairwiseKernel
{
public:

    using HandlerType = PairwiseSmartDPDHandler;

    PairwiseSmartDPD(std::string parameterName,float rc, float a, float gamma, float kbT, float dt, float power, long seed=42424242) :
        PairwiseSmartDPDHandler(parameterName,rc, a, gamma, kbT, dt, power),
        stepGen(seed)
    {}

    const HandlerType& handler() const
    {
        return (const HandlerType&)(*this);
    }

    void setup(LocalParticleVector* lpv1, LocalParticleVector* lpv2, CellList* cl1, CellList* cl2,  const YmrState *state)
    {
        // seed = t;
        // better use random seed (time-based) instead of time
        // time-based is IMPORTANT for momentum conservation!!
        // t is float, use it's bit representation as int to seed RNG
        float t = state->currentTime;
        int v = *((int*)&t);
        std::mt19937 gen(v);
        std::uniform_real_distribution<float> udistr(0.001, 1);
        seed = udistr(gen);
        pv1DPDparameter = lpv1->dataPerParticle.getData<DPDparameter>(ChannelNames::DPDparameters)->devPtr();
        pv2DPDparameter = lpv2->dataPerParticle.getData<DPDparameter>(ChannelNames::DPDparameters)->devPtr();
    }
    void writeState(std::ofstream& fout) override
    {
        TextIO::writeToStream(fout, stepGen);
    }

    bool readState(std::ifstream& fin) override
    {
        return TextIO::readFromStream(fin, stepGen);
    }

protected:
    StepRandomGen stepGen;
};
