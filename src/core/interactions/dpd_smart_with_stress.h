#pragma once

#include "dpd_smart.h"


class InteractionSmartDPDWithStress : public InteractionSmartDPD
{
public:
    InteractionSmartDPDWithStress(const YmrState *state,std::string name,std::string parameterName, std::string stressName,std::vector<float> weights, float rc, float a, float gamma, float kbt,float power, float stressPeriod);

    ~InteractionSmartDPDWithStress();

    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
                         float a=Default, float gamma=Default, float kbt=Default,
                         float power=Default) override;

protected:
    float stressPeriod;
};
