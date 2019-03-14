#pragma once

#include "../particle_vector.h"

#include <core/utils/common.h>
#include <core/utils/cuda_common.h>

/**
 * GPU-compatible struct of all the relevant data
 */
struct PVview
{
    int size = 0;
    float4 *particles = nullptr;
    float4 *forces = nullptr;

    float mass = 0, invMass = 0;

    PVview(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr)
    {
        if (lpv == nullptr) return;

        size = lpv->size();
        particles = reinterpret_cast<float4*>(lpv->coosvels.devPtr());
        forces    = reinterpret_cast<float4*>(lpv->forces.devPtr());

        mass = pv->mass;
        invMass = 1.0 / mass;
    }
};


struct PVviewWithOldParticles : public PVview
{
    float4 *old_particles = nullptr;

    PVviewWithOldParticles(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            old_particles = reinterpret_cast<float4*>( lpv->extraPerParticle.getData<Particle>(ChannelNames::oldParts)->devPtr() );
    }
};


struct PVviewWithDensities : public PVview
{
    float *densities = nullptr;

    PVviewWithDensities(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
            densities = lpv->extraPerParticle.getData<float>(ChannelNames::densities)->devPtr();
    }
};



struct PVviewWithFlowProperties : public PVview
{
    Vorticity *vorticities = nullptr;
    Aprox_Density *aprox_densities = nullptr;
    Velocity_Gradient *velocity_gradients = nullptr;
    Density_Gradient *density_gradients = nullptr;

    PVviewWithFlowProperties(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        PVview(pv, lpv)
    {
        if (lpv != nullptr)
        {
            vorticities = lpv->extraPerParticle.getData<Vorticity>(ChannelNames::vorticities)->devPtr();
            aprox_densities = lpv->extraPerParticle.getData<Aprox_Density>(ChannelNames::aprox_densities)->devPtr();
            velocity_gradients = lpv->extraPerParticle.getData<Velocity_Gradient>(ChannelNames::velocity_gradients)->devPtr();
            density_gradients = lpv->extraPerParticle.getData<Density_Gradient>(ChannelNames::density_gradients)->devPtr();
        }
    }
};

template <typename BasicView>
struct PVviewWithStresses : public BasicView
{
    Stress *stresses = nullptr;

    PVviewWithStresses(ParticleVector *pv = nullptr, LocalParticleVector *lpv = nullptr) :
        BasicView(pv, lpv)
    {
        if (lpv != nullptr)
            stresses = lpv->extraPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();
    }
};
