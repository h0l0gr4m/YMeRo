#pragma once

#include "dpd_smart.h"


class InteractionSmartDPDWithStress : public InteractionSmartDPD
{
public:
    InteractionSmartDPDWithStress(const YmrState *state,std::string name,std::string parameterName, std::string stressName,std::string NeuralNetType, std::vector<float> weights, float rc, float kbt,float power, float stressPeriod);

    ~InteractionSmartDPDWithStress();

    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
                         float kbt=Default,
                         float power=Default) override;

protected:
    float stressPeriod;
};
