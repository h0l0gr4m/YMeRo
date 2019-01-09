#include "dpd_smart.h"
#include <memory>
#include "pairwise_smart.h"
#include "pairwise_interactions/smartdpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionSmartDPD::InteractionSmartDPD(std::string name,std::string parameterName, float rc, float a, float gamma, float kbt, float dt, float power, bool allocateImpl) :
    Interaction(name, rc),
    parameterName(parameterName),a(a), gamma(gamma), kbt(kbt), dt(dt), power(power)
{
    if (allocateImpl) {
        Pairwise_SmartDPD dpd(parameterName,rc, a, gamma, kbt, dt, power);
        impl = std::make_unique<InteractionPairSmart<Pairwise_SmartDPD>> (name,parameterName,a,gamma ,rc, dpd);
    }
}

InteractionSmartDPD::InteractionSmartDPD(std::string name,std::string parameterName, float rc, float a, float gamma, float kbt, float dt, float power) :
    InteractionSmartDPD(name,parameterName, rc, a, gamma, kbt, dt, power, true)
{}

InteractionSmartDPD::~InteractionSmartDPD() = default;

void InteractionSmartDPD::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    impl->setPrerequisites(pv1, pv2);
}

void InteractionSmartDPD::regular(ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             const float t, cudaStream_t stream)
{
    impl->regular(pv1, pv2, cl1, cl2, t, stream);
}

void InteractionSmartDPD::halo   (ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             const float t, cudaStream_t stream)
{
    impl->halo   (pv1, pv2, cl1, cl2, t, stream);
}

void InteractionSmartDPD::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
        float a, float gamma, float kbt, float dt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (dt    == Default) dt    = this->dt;
    if (power == Default) power = this->power;

    Pairwise_SmartDPD dpd(parameterName,this->rc, a, gamma, kbt, dt, power);
    auto ptr = static_cast< InteractionPairSmart<Pairwise_SmartDPD>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
