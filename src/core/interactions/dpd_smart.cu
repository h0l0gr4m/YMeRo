#include "dpd_smart.h"
#include <memory>
#include "pairwise_smart.h"
#include "pairwise_interactions/smartdpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionSmartDPD::InteractionSmartDPD(const YmrState *state, std::string name, std::string parameterName, float rc, float a, float gamma, float kbt, float power, bool allocateImpl) :
    Interaction(state, name, rc),
    parameterName(parameterName),a(a), gamma(gamma), kbt(kbt), power(power)
{
    if (allocateImpl)
    {
        Pairwise_SmartDPD dpd(parameterName,rc, a, gamma, kbt, state->dt, power);
        impl = std::make_unique<InteractionPairSmart<Pairwise_SmartDPD>> (state,name,parameterName,a,gamma ,rc, dpd);
    }

}

InteractionSmartDPD::InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName, float rc, float a, float gamma, float kbt,  float power) :
    InteractionSmartDPD(state,name,parameterName, rc, a, gamma, kbt, power, true)
{}

InteractionSmartDPD::~InteractionSmartDPD() = default;

void InteractionSmartDPD::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    impl->setPrerequisites(pv1, pv2);
}

void InteractionSmartDPD::regular(ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             cudaStream_t stream)
{
    impl->regular(pv1, pv2, cl1, cl2, stream);
}

void InteractionSmartDPD::halo   (ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             cudaStream_t stream)
{
    impl->halo   (pv1, pv2, cl1, cl2,stream);
}

void InteractionSmartDPD::initStep(ParticleVector *pv1, ParticleVector *pv2, cudaStream_t stream)
{
    impl->initStep(pv1, pv2, stream);
}

void InteractionSmartDPD::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
        float a, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;

    Pairwise_SmartDPD dpd(parameterName,this->rc, a, gamma, kbt, state->dt, power);
    auto ptr = static_cast< InteractionPairSmart<Pairwise_SmartDPD>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
