#include "lossdensity_function.h"
#include "utils/simple_serializer.h"
#include <core/containers.h>
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include "interface.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/folders.h>
#include <core/utils/common.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace LossDensityFunctionKernels
{
__global__ void totalLoss(PVview view,const float viscosity, const Density_Gradient *density_gradients,LossDensityFunction::ReductionType *lossDensity)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double L = 0;
    Particle p;

    if (tid < view.size) {
        //calculate density gradient part of Loss LossDensityFunction
        Density_Gradient dg = density_gradients[tid];
        const float density_part = dg.x*dg.x + dg.y*dg.y + dg.z*dg.z ;

        L = density_part;
      }

    L= warpReduce(L, [](LossDensityFunction::ReductionType a, LossDensityFunction::ReductionType b) { return a+b; });
    if (__laneid() == 0)
        atomicAdd(lossDensity, L);


}
}

LossDensityFunctionPlugin::LossDensityFunctionPlugin(const YmrState *state, std::string name, std::string pvName,const float viscosity,
                                            int dumpEvery) :
    SimulationPlugin(state, name),
    pvName(pvName),
    dumpEvery(dumpEvery),
    viscosity(viscosity)
{}

LossDensityFunctionPlugin::~LossDensityFunctionPlugin() = default;

void LossDensityFunctionPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);


    info("Plugin %s initialized for the following particle vector: %s", name.c_str(), pvName.c_str());
}

void LossDensityFunctionPlugin::handshake()
{
    SimpleSerializer::serialize(sendBuffer, pvName);
    send(sendBuffer);
}

void LossDensityFunctionPlugin::beforeIntegration(cudaStream_t stream)
{
    if (state->currentStep % dumpEvery != 0 || state->currentStep == 0) return;

    PVview view(pv, pv->local());
    const Density_Gradient *density_gradients = pv->local()->dataPerParticle.getData<Density_Gradient>(ChannelNames::density_gradients)->devPtr();


    localLossDensityFunction.clear(stream);


    SAFE_KERNEL_LAUNCH(
        LossDensityFunctionKernels::totalLoss,
        getNblocks(view.size, 128), 128, 0, stream,
        view, viscosity,density_gradients, localLossDensityFunction.devPtr() );

    localLossDensityFunction.downloadFromDevice(stream, ContainersSynch::Synch);


    savedTime = state->currentTime;
    needToSend = true;
}

void LossDensityFunctionPlugin::serializeAndSend(cudaStream_t stream)
{
    if (!needToSend) return;

    debug2("Plugin %s is sending now data", name.c_str());
    PVview view(pv, pv->local());

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, savedTime, localLossDensityFunction[0]/view.size);
    send(sendBuffer);

    needToSend = false;
}

//=================================================================================

LossDensityFunctionDumper::LossDensityFunctionDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(path)
{
    if (std::is_same<LossDensityFunction::ReductionType, float>::value)
        mpiReductionType = MPI_FLOAT;
    else if (std::is_same<LossDensityFunction::ReductionType, double>::value)
        mpiReductionType = MPI_DOUBLE;
    else
        die("Incompatible type");
}

LossDensityFunctionDumper::~LossDensityFunctionDumper()
{
    if (activated)
        fclose(fdump);
}

void LossDensityFunctionDumper::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    activated = createFoldersCollective(comm, path);
}

void LossDensityFunctionDumper::handshake()
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

void LossDensityFunctionDumper::deserialize(MPI_Status& stat)
{
    YmrState::TimeType curTime;
    LossDensityFunction::ReductionType localLossDensityFunction, totalLossDensityFunction;

    SimpleSerializer::deserialize(data, curTime, localLossDensityFunction);

    if (!activated) return;

    MPI_Check( MPI_Reduce(&localLossDensityFunction, &totalLossDensityFunction, 1, mpiReductionType, MPI_SUM, 0, comm) );


    fprintf(fdump, "%g %.6e\n", curTime, totalLossDensityFunction);
}
