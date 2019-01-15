#pragma once

#include "dpd_smart.h"


class InteractionSmartDPDWithStress : public InteractionSmartDPD
{
public:
    InteractionSmartDPDWithStress(std::string name,std::string parameterName, std::string stressName, float rc, float a, float gamma, float kbt, float dt, float power, float stressPeriod);

    ~InteractionSmartDPDWithStress();

    void setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
                         float a=Default, float gamma=Default, float kbt=Default,
                         float dt=Default, float power=Default) override;

protected:
    float stressPeriod;
};
