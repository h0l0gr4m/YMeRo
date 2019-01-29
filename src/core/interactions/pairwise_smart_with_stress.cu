#include "pairwise_smart_with_stress.h"

#include "pairwise_interactions/smartdpd.h"

#include <core/datatypes.h>


/**
 * Implementation of short-range symmetric pairwise interactions
 */

template<class PairwiseInteraction>
void SmartInteractionPair_withStress<PairwiseInteraction>::regular(
        ParticleVector* pv1, ParticleVector* pv2,
        CellList* cl1, CellList* cl2,
       cudaStream_t stream)
{
    float t = state->currentTime;
    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        if (pv2lastStressTime[pv1] != t)
        {
            pv1->local()->extraPerParticle.getData<Stress>(stressName)->clear(0);
            pv2lastStressTime[pv1] = t;
        }

        if (pv2lastStressTime[pv2] != t)
        {
            pv2->local()->extraPerParticle.getData<Stress>(stressName)->clear(0);
            pv2lastStressTime[pv2] = t;
        }

        interactionWithStress.regular(pv1, pv2, cl1, cl2,  stream);
        lastStressTime = t;
    }
    else
        interaction.regular(pv1, pv2, cl1, cl2, stream);
}

template<class PairwiseInteraction>
void SmartInteractionPair_withStress<PairwiseInteraction>::halo   (
        ParticleVector* pv1, ParticleVector* pv2,
        CellList* cl1, CellList* cl2,
        cudaStream_t stream)
{
    float t = state->currentTime;
    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        if (pv2lastStressTime[pv1] != t)
        {
            pv1->local()->extraPerParticle.getData<Stress>(stressName)->clear(stream);
            pv2lastStressTime[pv1] = t;
        }

        if (pv2lastStressTime[pv2] != t)
        {
            pv2->local()->extraPerParticle.getData<Stress>(stressName)->clear(stream);
            pv2lastStressTime[pv2] = t;
        }

        interactionWithStress.halo(pv1, pv2, cl1, cl2, stream);
        lastStressTime = t;
    }
    else
        interaction.halo(pv1, pv2, cl1, cl2,stream);
}

template<class PairwiseInteraction>
void SmartInteractionPair_withStress<PairwiseInteraction>::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    info("Interaction '%s' requires channel 'parameterName' from PVs '%s' and '%s'",
         name.c_str(), pv1->name.c_str(), pv2->name.c_str());


    pv1->requireDataPerParticle<DPDparameter>(parameterName, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<DPDparameter>(parameterName, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    info("Interaction '%s' requires channel 'stress' from PVs '%s' and '%s'",
         name.c_str(), pv1->name.c_str(), pv2->name.c_str());

    pv1->requireDataPerParticle<Stress>(stressName, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Stress>(stressName, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<Divergence>("div_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Divergence>("div_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<Vorticity>("vorticity_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Vorticity>("vorticity_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<Velocity_Gradient>("v_grad_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Velocity_Gradient>("v_grad_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<NNInput>("NNInputs",ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<NNInput>("NNInputs",ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv2lastStressTime[pv1] = -1;
    pv2lastStressTime[pv2] = -1;
}

template<class PairwiseInteraction>
SmartInteractionPair_withStress<PairwiseInteraction>::SmartInteractionPair_withStress(
    const YmrState *state,std::string name,std::string parameterName,float a,float gamma,std::string stressName, float rc, float stressPeriod, PairwiseInteraction pair) :
    a(a),
    gamma(gamma),
    parameterName(parameterName),
    Interaction(state,name, rc),
    stressName(stressName),
    stressPeriod(stressPeriod),
    interaction(state,name,parameterName,a,gamma,rc, pair),
    interactionWithStress(state,name,parameterName,a,gamma,rc, PairwiseStressWrapper<PairwiseInteraction>(stressName, pair))
{ }

template<class PairwiseInteraction>
void SmartInteractionPair_withStress<PairwiseInteraction>::initStep(ParticleVector *pv1, ParticleVector *pv2, cudaStream_t stream)
{
    float t = state->currentTime;
    pv1->local()->extraPerParticle.getData<Divergence>("div_name")->clear(stream);
    pv2->local()->extraPerParticle.getData<Divergence>("div_name")->clear(stream);

    pv1->local()->extraPerParticle.getData<Vorticity>("vorticity_name")->clear(stream);
    pv2->local()->extraPerParticle.getData<Vorticity>("vorticity_name")->clear(stream);

    pv1->local()->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->clear(stream);
    pv2->local()->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->clear(stream);

    pv1->local()->extraPerParticle.getData<NNInput>("NNInputs")->clear(stream);
    pv2->local()->extraPerParticle.getData<NNInput>("NNInputs")->clear(stream);

    if (lastStressTime+stressPeriod <= t || lastStressTime == t) {

        if (pv2lastStressTime[pv1] != t)
            pv1->local()->extraPerParticle.getData<Stress>(stressName)->clear(stream);

        if (pv2lastStressTime[pv2] != t)
            pv2->local()->extraPerParticle.getData<Stress>(stressName)->clear(stream);
    }
}


template<class PairwiseInteraction>
void SmartInteractionPair_withStress<PairwiseInteraction>::setSpecificPair(
        std::string pv1name, std::string pv2name, PairwiseInteraction pair)
{
    interaction.          setSpecificPair(pv1name, pv2name, pair);
    interactionWithStress.setSpecificPair(pv1name, pv2name, PairwiseStressWrapper<PairwiseInteraction>(stressName, pair));
}


template class SmartInteractionPair_withStress<Pairwise_SmartDPD>;
