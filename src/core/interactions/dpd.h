#pragma once

#include "interface.h"
#include <memory>
#include <core/utils/pytypes.h>

/**
 * Implementation of Velocity-Verlet integration in one step
 */
struct InteractionDPD : Interaction
{
    std::unique_ptr<Interaction> impl;

    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;
    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    
    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2, 
        float a, float gamma, float kbt, float dt, float power);
    
    InteractionDPD(std::string name, float rc, float a, float gamma, float kbt, float dt, float power);

    ~InteractionDPD();
};
