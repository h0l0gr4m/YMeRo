#include <memory>

#include "dpd_smart_with_stress.h"
#include "pairwise_smart_with_stress.h"
#include "pairwise_interactions/smartdpd.h"
#include "calculations/FlowProperties.h"
#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionSmartDPDWithStress::InteractionSmartDPDWithStress(const YmrState *state,std::string name,std::string parameterName, std::string stressName,std::vector<float> weights,
                                                   float rc, float a, float gamma, float kbt, float power, float stressPeriod) :
    InteractionSmartDPD(state,name,parameterName,weights, rc, a, gamma, kbt, power, false),
    stressPeriod(stressPeriod)
{
    Pairwise_SmartDPD dpd(parameterName,rc, a, gamma, kbt, state->dt, power);
    FlowProperties<Pairwise_SmartDPD> fp ("fp_name",dpd);
    impl = std::make_unique<SmartInteractionPair_withStress<FlowProperties<Pairwise_SmartDPD>>>(state,name,parameterName,Weights,a,gamma, stressName, rc, stressPeriod, fp);
}

InteractionSmartDPDWithStress::~InteractionSmartDPDWithStress() = default;

void InteractionSmartDPDWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,float a, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;

    Pairwise_SmartDPD dpd(parameterName,this->rc, a, gamma, kbt,state->dt, power);
    FlowProperties<Pairwise_SmartDPD> fp ("fp_name",dpd);
    auto ptr = static_cast< SmartInteractionPair_withStress<FlowProperties<Pairwise_SmartDPD>>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, fp);
}
