#include "dpd_smart.h"
#include <memory>
#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/smartdpd.h"
#include "pairwise.impl.h"
#include "particle_kernel.h"

#include "pairwise_interactions/FlowProperties.h"


#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>




InteractionFlowProperty::InteractionFlowProperty(const YmrState *state, std::string name, float rc) :
    Interaction(state, name, rc)
{
    PairwiseFlowProperties fp(rc);
    impl = std::make_unique<InteractionPair<PairwiseFlowProperties>> (state, name, rc, fp);
}

InteractionFlowProperty::~InteractionFlowProperty() = default;

void InteractionFlowProperty::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2, cl1, cl2);

    info("Interaction '%s' requires channel '%s' from PVs '%s' and '%s'",
         name.c_str(), ChannelNames::DPDparameters.c_str(), pv1->name.c_str(), pv2->name.c_str());


    pv1->requireDataPerParticle <Vorticity> (ChannelNames::vorticities,ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <Vorticity> (ChannelNames::vorticities,ExtraDataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle <Vorticity> (ChannelNames::vorticities);
    cl2->requireExtraDataPerParticle <Vorticity> (ChannelNames::vorticities);

    pv1->requireDataPerParticle <Aprox_Density> (ChannelNames::aprox_densities,ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <Aprox_Density> (ChannelNames::aprox_densities,ExtraDataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle <Aprox_Density> (ChannelNames::aprox_densities);
    cl2->requireExtraDataPerParticle <Aprox_Density> (ChannelNames::aprox_densities);

    pv1->requireDataPerParticle <Density_Gradient> (ChannelNames::density_gradients,ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <Density_Gradient> (ChannelNames::density_gradients,ExtraDataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle <Density_Gradient> (ChannelNames::density_gradients);
    cl2->requireExtraDataPerParticle <Density_Gradient> (ChannelNames::density_gradients);

    pv1->requireDataPerParticle <Velocity_Gradient> (ChannelNames::velocity_gradients,ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <Velocity_Gradient> (ChannelNames::velocity_gradients,ExtraDataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle <Velocity_Gradient> (ChannelNames::velocity_gradients);
    cl2->requireExtraDataPerParticle <Velocity_Gradient> (ChannelNames::velocity_gradients);

}

std::vector<Interaction::InteractionChannel> InteractionFlowProperty::getIntermediateOutputChannels() const
{
    return {{ChannelNames::vorticities, Interaction::alwaysActive},{ChannelNames::aprox_densities, Interaction::alwaysActive},{ChannelNames::velocity_gradients, Interaction::alwaysActive},{ChannelNames::density_gradients, Interaction::alwaysActive}};
}
std::vector<Interaction::InteractionChannel> InteractionFlowProperty::getFinalOutputChannels() const
{
    return {};
}

void InteractionFlowProperty::local(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    impl->local(pv1, pv2, cl1, cl2, stream);
}

void InteractionFlowProperty::halo (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    impl->halo(pv1, pv2, cl1, cl2, stream);
}





InteractionSmartDPD::InteractionSmartDPD(const YmrState *state, std::string name, std::string parameterName,std::vector<float> weights, float rc, float a, float gamma, float kbt, float power, bool allocateImpl) :
    Interaction(state, name, rc),
    parameterName(parameterName),a(a), gamma(gamma), kbt(kbt), power(power)
{
    if (allocateImpl)
    {
        auto devP = Weights.hostPtr();
        memcpy(devP, &weights[0], weights.size() * sizeof(float));
        Weights.uploadToDevice(0);
        PairwiseSmartDPD dpd(parameterName,rc, a, gamma, kbt, state->dt, power);
        impl = std::make_unique<InteractionPair<PairwiseSmartDPD>> (state,name,rc,dpd);

    }

}

InteractionSmartDPD::InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName,std::vector<float> weights, float rc, float a, float gamma, float kbt,  float power) :
    InteractionSmartDPD(state,name,parameterName,weights, rc, a, gamma, kbt, power, true)
{}

InteractionSmartDPD::~InteractionSmartDPD() = default;

void InteractionSmartDPD::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2,CellList *cl1, CellList *cl2)
{
    impl->setPrerequisites(pv1, pv2,cl1,cl2);
    pv1->requireDataPerParticle <DPDparameter> (ChannelNames::DPDparameters,ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <DPDparameter> (ChannelNames::DPDparameters,ExtraDataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle <DPDparameter> (ChannelNames::DPDparameters);
    cl2->requireExtraDataPerParticle <DPDparameter> (ChannelNames::DPDparameters);

    auto pv1DPDparameter = pv1->local()->extraPerParticle.getData<DPDparameter>(ChannelNames::DPDparameters)->devPtr();
    auto pv2DPDparameter = pv2->local()->extraPerParticle.getData<DPDparameter>(ChannelNames::DPDparameters)->devPtr();
    int nth = 128;
    int np = pv1->local()->size();
    SAFE_KERNEL_LAUNCH(
        copy_kernel,
        getNblocks(np, nth), nth, 0, 0,
        pv1DPDparameter,pv2DPDparameter,np,a,gamma);


}

std::vector<Interaction::InteractionChannel> InteractionSmartDPD::getFinalOutputChannels() const
{
    return impl->getFinalOutputChannels();
}

std::vector<Interaction::InteractionChannel> InteractionSmartDPD::getIntermediateInputChannels() const
{
    return {{ChannelNames::vorticities, Interaction::alwaysActive},{ChannelNames::aprox_densities, Interaction::alwaysActive},{ChannelNames::velocity_gradients, Interaction::alwaysActive},{ChannelNames::density_gradients, Interaction::alwaysActive}};
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


    PairwiseSmartDPD dpd(parameterName,this->rc, a, gamma, kbt, state->dt, power);
    auto ptr = static_cast<InteractionPair<PairwiseSmartDPD>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
