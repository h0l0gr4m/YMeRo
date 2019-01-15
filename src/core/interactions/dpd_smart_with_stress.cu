#include <memory>

#include "dpd_smart_with_stress.h"
#include "pairwise_smart_with_stress.h"
#include "pairwise_interactions/smartdpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionSmartDPDWithStress::InteractionSmartDPDWithStress(const YmrState *state,std::string name,std::string parameterName, std::string stressName,
                                                   float rc, float a, float gamma, float kbt, float power, float stressPeriod) :
    InteractionSmartDPD(state,name,parameterName, rc, a, gamma, kbt, power, false),
    stressPeriod(stressPeriod)
{
    Pairwise_SmartDPD dpd(parameterName,rc, a, gamma, kbt, state->dt, power);
    impl = std::make_unique<SmartInteractionPair_withStress<Pairwise_SmartDPD>> (state,name,parameterName,a,gamma, stressName, rc, stressPeriod, dpd);
}

InteractionSmartDPDWithStress::~InteractionSmartDPDWithStress() = default;

void InteractionSmartDPDWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
                                               float a, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;

    Pairwise_SmartDPD dpd(parameterName,this->rc, a, gamma, kbt,state->dt, power);
    auto ptr = static_cast< SmartInteractionPair_withStress<Pairwise_SmartDPD>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
