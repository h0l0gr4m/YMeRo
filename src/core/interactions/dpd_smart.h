#pragma once

#include <vector>
#include "interface.h"
#include <memory>
#include <limits>
#include <core/utils/pytypes.h>
#include <core/containers.h>

class InteractionFlowProperty : public Interaction
{
public:
    InteractionFlowProperty(const YmrState *state, std::string name, float rc);

    ~InteractionFlowProperty();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    std::vector<InteractionChannel> getIntermediateOutputChannels() const override;
    std::vector<InteractionChannel> getFinalOutputChannels() const override;

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

protected:

    std::unique_ptr<Interaction> impl;
};



class InteractionSmartDPD : public Interaction
{
public:
    constexpr static float Default = std::numeric_limits<float>::infinity();

    InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName,std::string NeuralNetType,std::vector<float> weights,float viscosity,float rc,  float kbt, float power);

    ~InteractionSmartDPD();

    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    std::vector<InteractionChannel> getFinalOutputChannels() const override;
    std::vector<InteractionChannel> getIntermediateInputChannels() const override;

    void local(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream) override;

    void localNeuralNetwork(ParticleVector* pv, CellList *cl, cudaStream_t stream) override;
    void haloNeuralNetwork(ParticleVector* pv, CellList *cl, cudaStream_t stream) override;



    virtual void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
                                 float kbt=Default,
                                 float power=Default);

protected:

    InteractionSmartDPD(const YmrState *state,std::string name,std::string parameterName,std::string NeuralNetType,std::vector<float> weights,float viscosity, float rc, float kbt,  float power, bool allocateImpl);

    std::unique_ptr<Interaction> impl;

    // Default values
    float  kbt,  power,viscosity ;
    std::string parameterName;
    PinnedBuffer<float> Weights;
    std::vector<float> weights;
    std::string NeuralNetType;
};
