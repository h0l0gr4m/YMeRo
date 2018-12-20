#pragma once
#include "interface.h"
#include "pairwise_interactions/smartdpd.h"


#include <map>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class InteractionPairSmart : public Interaction
{
public:
    enum class InteractionType { Regular, Halo };

    void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);

    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2);

    InteractionPairSmart(std::string name, std::string parameterName,  float rc, PairwiseInteraction pair) :
        Interaction(name, rc),parameterName(parameterName), defaultPair(pair)
    { }

    void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);



    ~InteractionPairSmart() = default;

private:
    std::map< std::pair<std::string, std::string>, PairwiseInteraction > intMap;
    std::string parameterName;
    PairwiseInteraction defaultPair;
    DPDparameter *pv1DPDparameter, *pv2DPDparameter;
};
