#include "pairwise_smart.h"

#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/logger.h>

#include "pairwise_kernels.h"


#include "pairwise_interactions/stress_wrapper.h"
#include "pairwise_interactions/smartdpd.h"
#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"
#include "pairwise_interactions/norandom_dpd.h"

#include "calculations/NeuralNet_kernel.h"
#include "calculations/FlowProperties.h"
#include "calculations/NNInputs.h"
#include "calculations/nninput_kernel.h"

__global__ void copy_kernel(DPDparameter* devPointer1,DPDparameter* devPointer2,int np, float a, float gamma)
{
   const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
   if (dstId >= np) return;
   devPointer1[dstId].alpha_p = a;
   devPointer1[dstId].gamma_p = gamma;
   devPointer2[dstId].alpha_p = a;
   devPointer2[dstId].gamma_p = gamma;

}

/**
 * Convenience macro wrapper
 *
 * Select one of the available kernels for external interaction depending
 * on the number of particles involved, report it and call
 */
 #define DISPATCH_EXTERNAL(P1, P2, P3, TPP, INTERACTION_FUNCTION)                \
 do{ debug2("Dispatched to "#TPP" thread(s) per particle variant");              \
     SAFE_KERNEL_LAUNCH(                                                         \
             computeExternalInteractions_##TPP##tpp<P1 COMMA P2 COMMA P3>,       \
             getNblocks(TPP*view.size, nth), nth, 0, stream,                     \
             view, cl2->cellInfo(), rc*rc, INTERACTION_FUNCTION); } while (0)

 #define CHOOSE_EXTERNAL(P1, P2, P3, INTERACTION_FUNCTION)                                              \
 do{  if (view.size < 1000  ) { DISPATCH_EXTERNAL(P1, P2, P3, 27, INTERACTION_FUNCTION); }              \
 else if (view.size < 10000 ) { DISPATCH_EXTERNAL(P1, P2, P3, 9,  INTERACTION_FUNCTION); }              \
 else if (view.size < 400000) { DISPATCH_EXTERNAL(P1, P2, P3, 3,  INTERACTION_FUNCTION); }              \
 else                         { DISPATCH_EXTERNAL(P1, P2, P3, 1,  INTERACTION_FUNCTION); } } while(0)




/**
 * Interface to _compute() with local interactions.
 */
template<class PariwiseInteraction>
void InteractionPairSmart<PariwiseInteraction>::regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, cudaStream_t stream)
{
    //if (pv1->local()->size() < pv2->local()->size())
        _compute(InteractionType::Regular, pv1, pv2, cl1, cl2,state->currentTime, stream);
    //else
    //    _compute(InteractionType::Regular, pv2, pv1, cl2, cl1, t, stream);
}

/**
 * Interface to _compute() with halo interactions.
 *
 * The following cases exist:
 * - If one of \p pv1 or \p pv2 is ObjectVector, then only call to the _compute()
 *   needed: for halo ObjectVector another ParticleVector (or ObjectVector).
 *   This is because ObjectVector will collect the forces from remote processors,
 *   so we don't need to compute them twice.
 *
 * - Both are ParticleVector. Then if they are different, two _compute() calls
 *   are made such that halo1 \<-\> local2 and halo2 \<-\> local1. If \p pv1 and
 *   \p pv2 are the same, only one call is needed
 */
template<class PairwiseInteraction>
void InteractionPairSmart<PairwiseInteraction>::halo(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2,  cudaStream_t stream)
{
    auto isov1 = dynamic_cast<ObjectVector*>(pv1) != nullptr;
    auto isov2 = dynamic_cast<ObjectVector*>(pv2) != nullptr;

    // Two object vectors. Compute just one interaction, doesn't matter which
    if (isov1 && isov2)
    {
        _compute(InteractionType::Halo, pv1, pv2, cl1, cl2,state->currentTime, stream);
        return;
    }

    // One object vector. Compute just one interaction, with OV as the first argument
    if (isov1)
    {
        _compute(InteractionType::Halo, pv1, pv2, cl1, cl2,state->currentTime, stream);
        return;
    }

    if (isov2)
    {
        _compute(InteractionType::Halo, pv2, pv1, cl2, cl1,state->currentTime, stream);
        return;
    }

    // Both are particle vectors. Compute one interaction if pv1 == pv2 and two otherwise
    _compute(InteractionType::Halo, pv1, pv2, cl1, cl2,state->currentTime, stream);
    if(pv1 != pv2)
        _compute(InteractionType::Halo, pv2, pv1, cl2, cl1,state->currentTime, stream);
}

/**
 * Compute forces between all the pairs of particles that are closer
 * than #rc to each other.
 *
 * Depending on \p type and whether \p pv1 == \p pv2 call
 * computeSelfInteractions() or computeExternalInteractions_1tpp()
 * (or other variants of external interaction kernels).
 *
 * @tparam PariwiseInteraction is a functor that computes the force
 * given a pair of particles. It has to
 * provide two functions:
 * - This function will be called once before interactions computation
 *   and allows the functor to obtain required variables or data
 *   channels from the two ParticleVector and CellList:
 *   \code setup(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, float t) \endcode
 *
 * - This should be a \c \_\_device\_\_ operator that computes
 *   the force. It will be called for each close enough particle pair:
 *   \code float3 operator()(const Particle dst, int dstId, const Particle src, int srcId) const \endcode
 *   Return value of that call is force acting on the first particle,
 *   force acting on the second one is just opposite.
 */
template<class PairwiseInteraction>
void InteractionPairSmart<PairwiseInteraction>::_compute(InteractionType type,
        ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
    auto it = intMap.find({pv1->name, pv2->name});
    if (it != intMap.end())
        debug("Using SPECIFIC parameters for PV pair '%s' -- '%s'", pv1->name.c_str(), pv2->name.c_str());
    else
        debug("Using default parameters for PV pair '%s' -- '%s'", pv1->name.c_str(), pv2->name.c_str());


    auto& pair = (it == intMap.end()) ? defaultPair : it->second;

    if (type == InteractionType::Regular)
    {
        pair.setup(pv1->local(), pv2->local(), cl1, cl2, t);
        /*  Self interaction */
        if (pv1 == pv2)
        {
            const int np = pv1->local()->size();
            debug("Computing internal forces for %s (%d particles)", pv1->name.c_str(), np);

            const int nth = 128;
            auto cinfo = cl1->cellInfo();

            SAFE_KERNEL_LAUNCH(
                    computeSelfInteractions,
                    getNblocks(np, nth), nth, 0, stream,
                    np, cinfo, rc*rc, pair);

        }
        else /*  External interaction */
        {   const int np1 = pv1->local()->size();
            const int np2 = pv2->local()->size();
            debug("Computing external forces for %s - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), np1, np2);

            PVview view(pv1, pv1->local());
            cl1->setViewPtrs(view);
            const int nth = 128;
            if (np1 > 0 && np2 > 0)
                CHOOSE_EXTERNAL(InteractionOut::NeedAcc, InteractionOut::NeedAcc, InteractionMode::RowWise, pair);
        }
    }

    /*  Halo interaction */
    if (type == InteractionType::Halo)
    {
        pair.setup(pv1->halo(), pv2->local(), cl1, cl2, t);
        const int np1 = pv1->halo()->size();  // note halo here
        const int np2 = pv2->local()->size();
        debug("Computing halo forces for %s(halo) - %s (%d - %d particles)", pv1->name.c_str(), pv2->name.c_str(), np1, np2);

        PVview view(pv1, pv1->halo());
        const int nth = 128;
        if (np1 > 0 && np2 > 0)
            if (dynamic_cast<ObjectVector*>(pv1) == nullptr) // don't need forces for pure particle halo
            {
                CHOOSE_EXTERNAL(InteractionOut::NoAcc,   InteractionOut::NeedAcc, InteractionMode::Dilute, pair);
            }
            else
            {
                CHOOSE_EXTERNAL(InteractionOut::NeedAcc, InteractionOut::NeedAcc, InteractionMode::Dilute, pair);
            }
    }
    const int np = pv1->local()->size();
    const int nth = 128;
    NNInputs inputsnn(Weights);
    inputsnn.setup(pv1->local(), pv2->local(), cl1, cl2, t);
    SAFE_KERNEL_LAUNCH(
            computeNNInputs,
            getNblocks(np, nth), nth, 0, stream,
            np,inputsnn);
    auto dev_weightPt = Weights.hostPtr();

    NNInput *pv1NNInputs = pv1->local()->extraPerParticle.getData<NNInput>("NNInputs")->devPtr();
    SAFE_KERNEL_LAUNCH(
            NeuralNet,
            getNblocks(16*np, nth), nth, 0, stream,
            np, 4,pv1DPDparameter, pv1NNInputs,dev_weightPt);

}


template<class PairwiseInteraction>
void InteractionPairSmart<PairwiseInteraction>::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    info("Interaction '%s' requires channel 'parameterName' from PVs '%s' and '%s'",
         name.c_str(), pv1->name.c_str(), pv2->name.c_str());


    const int nth = 128;
    const int np = pv1->local()->size();
    pv1->requireDataPerParticle<DPDparameter>(parameterName,ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<DPDparameter>(parameterName,ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<Vorticity>("vorticity_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Vorticity>("vorticity_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<Velocity_Gradient>("v_grad_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Velocity_Gradient>("v_grad_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<Aprox_Density>("aprox_density_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Aprox_Density>("aprox_density_name", ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv1->requireDataPerParticle<NNInput>("NNInputs",ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<NNInput>("NNInputs",ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);


    pv1DPDparameter = pv1->local()->extraPerParticle.getData<DPDparameter>(parameterName)->devPtr();
    pv2DPDparameter = pv2->local()->extraPerParticle.getData<DPDparameter>(parameterName)->devPtr();
    SAFE_KERNEL_LAUNCH(
            copy_kernel,
            getNblocks(np, nth), nth, 0, 0,
            pv1DPDparameter,pv2DPDparameter,np,a,gamma);

}

template<class PairwiseInteraction>
void InteractionPairSmart<PairwiseInteraction>::initStep(ParticleVector *pv1, ParticleVector *pv2, cudaStream_t stream)
{

    pv1->local()->extraPerParticle.getData<Vorticity>("vorticity_name")->clear(stream);
    pv2->local()->extraPerParticle.getData<Vorticity>("vorticity_name")->clear(stream);

    pv1->local()->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->clear(stream);
    pv2->local()->extraPerParticle.getData<Velocity_Gradient>("v_grad_name")->clear(stream);

    pv1->local()->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->clear(stream);
    pv2->local()->extraPerParticle.getData<Aprox_Density>("aprox_density_name")->clear(stream);

    pv1->local()->extraPerParticle.getData<NNInput>("NNInputs")->clear(stream);
    pv2->local()->extraPerParticle.getData<NNInput>("NNInputs")->clear(stream);
}

template<class PairwiseInteraction>
void InteractionPairSmart<PairwiseInteraction>::setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair)
{
    intMap.insert({{pv1name, pv2name}, pair});
    intMap.insert({{pv2name, pv1name}, pair});
}



//for testing purpose
template class InteractionPairSmart<FlowProperties<Pairwise_SmartDPD>>;
template class InteractionPairSmart<FlowProperties<Pairwise_DPD>>;
template class InteractionPairSmart<PairwiseStressWrapper<FlowProperties<Pairwise_SmartDPD>>>;
