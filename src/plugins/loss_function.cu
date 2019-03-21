#include "loss_function.h"
#include "utils/simple_serializer.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/folders.h>
#include <core/utils/common.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace LossFunctionKernels
{
__global__ void totalLoss(PVview view,const float viscosity, const Stress *stress,const Velocity_Gradient *velocity_gradient,  LossFunction::ReductionType *loss)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    LossFunction::ReductionType L = 0;
    Particle p;

    if (tid < view.size) {
        Stress s = stress[tid];
        Velocity_Gradient v_g = velocity_gradient[tid];
        //calculate velocity_gradient + velocity_gradient transposed
        const Velocity_Gradient v_g_transposed = {v_g.xx , v_g.yx , v_g.zx , v_g.xy , v_g.yy ,v_g.zy,v_g.xz,v_g.yz,v_g.zz};
        Velocity_Gradient v_g_new = v_g + v_g_transposed;
        Velocity_Gradient v_g_new2 = viscosity * v_g_new;

        //calculate stress tensor minus its 1/3 own  trace
        const float tr_stress = -(s.xx + s.yy + s.zz) / 3.0;
        const Stress s_new = s + tr_stress;
        //calculate Stress - v_g_new

        //caluclate Frobeniusnorm
        const float a11 = s_new.xx-v_g_new2.xx;
        const float a12 = s_new.xy-v_g_new2.xy;
        const float a13 = s_new.xz-v_g_new2.xz;
        const float a21 = s_new.xy-v_g_new2.yx;
        const float a22 = s_new.yy-v_g_new2.yy;
        const float a23 = s_new.yz-v_g_new2.yz;
        const float a31 = s_new.xz-v_g_new2.zx;
        const float a32 = s_new.yz-v_g_new2.zy;
        const float a33 = s_new.zz-v_g_new2.zz;
        L = sqrt(a11*a11 +a12*a12 +a13*a13 +a21*a21 +a22*a22 +a23*a23 +a31*a31 +a32*a32+ a33*a33);


    }

    L = warpReduce(L, [](LossFunction::ReductionType a, LossFunction::ReductionType b) { return a+b; });

    if (__laneid() == 0)
        atomicAdd(loss, L);
}
}

LossFunctionPlugin::LossFunctionPlugin(const YmrState *state, std::string name, std::string pvName,const float viscosity,
                                            int dumpEvery) :
    SimulationPlugin(state, name),
    pvName(pvName),
    dumpEvery(dumpEvery),
    viscosity(viscosity)
{}

LossFunctionPlugin::~LossFunctionPlugin() = default;

void LossFunctionPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);


    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void LossFunctionPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, pvName);
    send(sendBuffer);
}

void LossFunctionPlugin::beforeIntegration(cudaStream_t stream)
{
    if (state->currentStep % dumpEvery != 0 || state->currentStep == 0) return;

    PVview view(pv, pv->local());
    const Stress *stress = pv->local()->extraPerParticle.getData<Stress>(ChannelNames::stresses)->devPtr();
    const Velocity_Gradient *velocity_gradient = pv->local()->extraPerParticle.getData<Velocity_Gradient>(ChannelNames::velocity_gradients)->devPtr();


    localLossFunction.clear(stream);

    SAFE_KERNEL_LAUNCH(
        LossFunctionKernels::totalLoss,
        getNblocks(view.size, 128), 128, 0, stream,
        view, viscosity,stress,velocity_gradient, localLossFunction.devPtr() );

    localLossFunction.downloadFromDevice(stream, ContainersSynch::Synch);

    savedTime = state->currentTime;
    needToSend = true;
}

void LossFunctionPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!needToSend) return;

    debug2("Plugin %s is sending now data", name.c_str());
    PVview view(pv, pv->local());

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, savedTime, localLossFunction[0]/view.size);
    send(sendBuffer);

    needToSend = false;
}

//=================================================================================

LossFunctionDumper::LossFunctionDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(path)
{
    if (std::is_same<LossFunction::ReductionType, float>::value)
        mpiReductionType = MPI_FLOAT;
    else if (std::is_same<LossFunction::ReductionType, double>::value)
        mpiReductionType = MPI_DOUBLE;
    else
        die("Incompatible type");
}

LossFunctionDumper::~LossFunctionDumper()
{
    if (activated)
        fclose(fdump);
}

void LossFunctionDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void LossFunctionDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::string pvName;
    SimpleSerializer::deserialize(data, pvName);

    if (activated)
    {
        auto fname = path + "/" + pvName + ".txt";
        fdump = fopen(fname.c_str(), "w");
        if (!fdump) die("Could not open file '%s'", fname.c_str());
        fprintf(fdump, "# time Loss\n");
    }
}

void LossFunctionDumper::deserialize(MPI_Status& stat)
{
    TimeType curTime;
    LossFunction::ReductionType localLossFunction, totalLossFunction;

    SimpleSerializer::deserialize(data, curTime, localLossFunction);

    if (!activated) return;

    MPI_Check( MPI_Reduce(&localLossFunction, &totalLossFunction, 1, mpiReductionType, MPI_SUM, 0, comm) );


    fprintf(fdump, "%g %.6e\n", curTime, totalLossFunction);
}
