#pragma once
#include "interface.h"
#include "pairwise_smart.h"
#include "pairwise_interactions/stress_wrapper.h"
#include "pairwise_interactions/smartdpd.h"
#include "calculations/FlowProperties.h"



#include <core/datatypes.h>
#include <map>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class SmartInteractionPair_withStress : public Interaction
{
public:
    enum class InteractionType { Regular, Halo };

    void local(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2,  cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2,  cudaStream_t stream) override;
    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2,CellList* cl1, CellList* cl2) override ;

    SmartInteractionPair_withStress(const YmrState *state,std::string name,std::string parameterName,PinnedBuffer<float> Weights,float a,float gamma, std::string stressName, float rc, float stressPeriod, PairwiseInteraction pair);

    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);
    std::vector<InteractionChannel> getFinalOutputChannels() const override;


    ~SmartInteractionPair_withStress() = default;

private:
    float a;
    float gamma;
    float stressPeriod;
    float lastStressTime{-1e6};


    std::map<ParticleVector*, float> pv2lastStressTime;
    std::string stressName;
    std::string parameterName;
    PinnedBuffer<float> Weights;

    InteractionPairSmart<PairwiseInteraction> interaction;
    InteractionPairSmart<PairwiseStressWrapper<PairwiseInteraction>> interactionWithStress;
};
