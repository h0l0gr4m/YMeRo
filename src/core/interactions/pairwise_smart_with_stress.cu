#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"
#include "pairwise_interactions/mdpd.h"
#include "pairwise_smart_with_stress.h"
#include "pairwise_interactions/smartdpd.h"

#include <core/celllist.h>
#include <core/utils/common.h>

template<class PairwiseInteraction>
InteractionPairSmart_withStress<PairwiseInteraction>::InteractionPairSmart_withStress( const YmrState *state,std::string name,std::string parameterName, std::string stressName,PinnedBuffer<float> Weights,float a,float gamma, float rc, float stressPeriod, PairwiseInteraction pair) :

    Interaction(state, name, rc),
    stressPeriod(stressPeriod),
    interaction(state, name,parameterName,Weights,a,gamma,rc, pair),
    interactionWithStress(state, name,parameterName,Weights,a,gamma, rc,PairwiseStressWrapper<PairwiseInteraction>(pair))
{}

template<class PairwiseInteraction>
InteractionPairSmart_withStress<PairwiseInteraction>::~InteractionPairSmart_withStress() = default;

template<class PairwiseInteraction>
void InteractionPairSmart_withStress<PairwiseInteraction>::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    interaction.setPrerequisites(pv1,pv2,cl1,cl2);
    info("Interaction '%s' requires channel '%s' from PVs '%s' and '%s'",
         name.c_str(), ChannelNames::stresses.c_str(), pv1->name.c_str(), pv2->name.c_str());

    pv1->requireDataPerParticle <Stress> (ChannelNames::stresses, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <Stress> (ChannelNames::stresses, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
    cl2->requireExtraDataPerParticle <Stress> (ChannelNames::stresses);
}

template<class PairwiseInteraction>
std::vector<Interaction::InteractionChannel> InteractionPairSmart_withStress<PairwiseInteraction>::getFinalOutputChannels() const
{
    auto activePredicateStress = [this]() {
       float t = state->currentTime;
       return (lastStressTime+stressPeriod <= t) || (lastStressTime == t);
    };

    return {{ChannelNames::forces, Interaction::alwaysActive},
            {ChannelNames::stresses, activePredicateStress}};
}

template<class PairwiseInteraction>
void InteractionPairSmart_withStress<PairwiseInteraction>::local(
        ParticleVector* pv1, ParticleVector* pv2,
        CellList* cl1, CellList* cl2, cudaStream_t stream)
{
    float t = state->currentTime;

    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        interactionWithStress.local(pv1, pv2, cl1, cl2, stream);
        lastStressTime = t;
    }
    else
        interaction.local(pv1, pv2, cl1, cl2, stream);
}

template<class PairwiseInteraction>
void InteractionPairSmart_withStress<PairwiseInteraction>::halo   (
        ParticleVector *pv1, ParticleVector *pv2,
        CellList *cl1, CellList *cl2,
        cudaStream_t stream)
{
    float t = state->currentTime;

    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        interactionWithStress.halo(pv1, pv2, cl1, cl2, stream);
        lastStressTime = t;
    }
    else
        interaction.halo(pv1, pv2, cl1, cl2, stream);
}

template<class PairwiseInteraction>
void InteractionPairSmart_withStress<PairwiseInteraction>::setSpecificPair(
        std::string pv1name, std::string pv2name, PairwiseInteraction pair)
{
    interaction.          setSpecificPair(pv1name, pv2name, pair);
    interactionWithStress.setSpecificPair(pv1name, pv2name, PairwiseStressWrapper<PairwiseInteraction>(pair));
}

template class InteractionPairSmart_withStress<Pairwise_SmartDPD>;
