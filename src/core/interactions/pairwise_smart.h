#pragma once
#include "interface.h"
#include "pairwise_interactions/smartdpd.h"
#include "calculations/FlowProperties.h"
#include "calculations/NNInputs.h"



#include <map>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class InteractionPairSmart : public Interaction
{
public:
    enum class InteractionType { Regular, Halo };

    InteractionPairSmart(const YmrState *state,std::string name, std::string parameterName, PinnedBuffer<float> Weights,float a, float gamma,  float rc, PairwiseInteraction pair) :
      ~InteractionPairSmart() = default;

    void local(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2,  cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2,  cudaStream_t stream) override;

    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);

private:

    void computeLocal(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);
    void computeHalo (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);
    PairwiseInteraction& getPairwiseInteraction(std::string pv1name, std::string pv2name);

    float a;
    float gamma;
    PinnedBuffer<float> Weights;
    std::map< std::pair<std::string, std::string>, PairwiseInteraction > intMap;
    std::string parameterName;
    PairwiseInteraction defaultPair;
    DPDparameter *pv1DPDparameter, *pv2DPDparameter;
};
