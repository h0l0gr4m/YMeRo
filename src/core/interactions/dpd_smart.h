#pragma once

#include "interface.h"
#include <memory>
#include <limits>
#include <core/utils/pytypes.h>
#include <core/containers.h>

class InteractionSmartDPD : public Interaction
{
public:
    constexpr static float Default = std::numeric_limits<float>::infinity();

    InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName,float rc, float a, float gamma, float kbt, float power);

    ~InteractionSmartDPD();

    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) ;
    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream) override;
    void initStep(ParticleVector *pv1, ParticleVector *pv2, cudaStream_t stream) override;


    virtual void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
                                 float a=Default, float gamma=Default, float kbt=Default,
                                 float power=Default);

protected:

    InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName, float rc, float a, float gamma, float kbt,  float power, bool allocateImpl);

    std::unique_ptr<Interaction> impl;

    // Default values
    float a, gamma, kbt,  power;
    std::string parameterName;
};
