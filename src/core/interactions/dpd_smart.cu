#include "dpd_smart.h"
#include <memory>
#include "pairwise_smart.h"
#include "pairwise_interactions/dpd.h"
#include "calculations/FlowProperties.h"


#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionSmartDPD::InteractionSmartDPD(const YmrState *state, std::string name, std::string parameterName,std::vector<float> weights, float rc, float a, float gamma, float kbt, float power, bool allocateImpl) :
    Interaction(state, name, rc),
    parameterName(parameterName),a(a), gamma(gamma), kbt(kbt), power(power)
{
    if (allocateImpl)
    {
        auto devP = Weights.hostPtr();
        memcpy(devP, &weights[0], weights.size() * sizeof(float));
        Weights.uploadToDevice(0);
        Pairwise_SmartDPD dpd(parameterName,rc, a, gamma, kbt, state->dt, power);
        FlowProperties<Pairwise_SmartDPD> fp("fp_name", dpd);
        impl = std::make_unique<InteractionPairSmart<FlowProperties<Pairwise_SmartDPD>>> (state,name,parameterName,Weights,a,gamma ,rc,fp);

    }

}

InteractionSmartDPD::InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName,std::vector<float> weights, float rc, float a, float gamma, float kbt,  float power) :
    InteractionSmartDPD(state,name,parameterName,weights, rc, a, gamma, kbt, power, true)
{}

InteractionSmartDPD::~InteractionSmartDPD() = default;

void InteractionSmartDPD::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2,CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2,cl1,cl2);
}

std::vector<Interaction::InteractionChannel> InteractionSmartDPD::getFinalOutputChannels() const
{
    return impl->getFinalOutputChannels();
}

void InteractionSmartDPD::local(ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionSmartDPD::halo   (ParticleVector* pv1, ParticleVector* pv2,
                             CellList* cl1, CellList* cl2,
                             cudaStream_t stream)
{
    impl->halo   (pv1, pv2, cl1, cl2,stream);
}


void InteractionSmartDPD::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
        float a, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;


    Pairwise_SmartDPD dpd(parameterName,this->rc, a, gamma, kbt, state->dt, power);
    FlowProperties<Pairwise_SmartDPD> fp("fp_name", dpd);
    auto ptr = static_cast< InteractionPairSmart<FlowProperties<Pairwise_SmartDPD>>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, fp);
}
