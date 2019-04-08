#include <memory>

#include "dpd_with_stress.h"
#include "pairwise_with_stress.impl.h"
#include "pairwise_interactions/dpd.h"

#include <core/utils/make_unique.h>
#include <core/pvs/particle_vector.h>


InteractionDPDWithStress::InteractionDPDWithStress(const YmrState *state, std::string name,
                                                   float rc, float a, float gamma, float kbt, float power, float stressPeriod) :
    InteractionDPD(state, name, rc, a, gamma, kbt, power, false)
{
    PairwiseDPD dpd(rc, a, gamma, kbt, state->dt, power);
    impl = std::make_unique<InteractionPair_withStress<PairwiseDPD>> (state, name, rc, stressPeriod, dpd);
}

InteractionDPDWithStress::~InteractionDPDWithStress() = default;

<<<<<<< HEAD
void InteractionDPDWithStress::setSpecificPair(ParticleVector* pv1, ParticleVector* pv2,
=======
void InteractionDPDWithStress::setSpecificPair(ParticleVector *pv1, ParticleVector *pv2, 
>>>>>>> c86ae8f55aa2a8da7e962d4b3b68feef3033bca4
                                               float a, float gamma, float kbt, float power)
{
    if (a     == Default) a     = this->a;
    if (gamma == Default) gamma = this->gamma;
    if (kbt   == Default) kbt   = this->kbt;
    if (power == Default) power = this->power;


    PairwiseDPD dpd(this->rc, a, gamma, kbt, state->dt, power);
    auto ptr = static_cast< InteractionPair_withStress<PairwiseDPD>* >(impl.get());

    ptr->setSpecificPair(pv1->name, pv2->name, dpd);
}
