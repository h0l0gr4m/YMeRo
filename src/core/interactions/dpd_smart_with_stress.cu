#include <memory>

#include "dpd_smart_with_stress.h"
#include "pairwise_with_stress.impl.h"
#include "pairwise_interactions/smartdpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionSmartDPDWithStress::InteractionSmartDPDWithStress(const YmrState *state,std::string name,std::string parameterName,std::string stressName,std::string NeuralNetType,std::vector<float> weights,
                                                   float viscosity,float rc, float kbt, float power, float stressPeriod) :
    InteractionSmartDPD(state,name,parameterName,NeuralNetType,weights,viscosity, rc, kbt, power, false),
    stressPeriod(stressPeriod)
{
    PairwiseSmartDPD dpd(parameterName,rc, kbt, state->dt, power);
    impl = std::make_unique<InteractionPair_withStress<PairwiseSmartDPD>>(state,name, rc, stressPeriod, dpd);
}

InteractionSmartDPDWithStress::~InteractionSmartDPDWithStress() = default;

void InteractionSmartDPDWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,float kbt, float power)
{
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;

    PairwiseSmartDPD dpd(parameterName,this->rc,  kbt,state->dt, power);
    auto ptr = static_cast< InteractionPair_withStress<PairwiseSmartDPD>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
