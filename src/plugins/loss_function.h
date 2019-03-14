#pragma once

#include <core/containers.h>

#include "interface.h"

class ParticleVector;

namespace LossFunction
{
using ReductionType = double;
}

class LossFunctionPlugin : public SimulationPlugin
{
public:
    LossFunctionPlugin(const YmrState *state, std::string name, std::string pvName,const float viscosity,
                       int dumpEvery);

    ~LossFunctionPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void beforeIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName;
    int dumpEvery;
    bool needToSend = false;


    PinnedBuffer<LossFunction::ReductionType> localLossFunction {1};
    TimeType savedTime = 0;

    std::vector<char> sendBuffer;

    ParticleVector *pv;
    const float viscosity;
};


class LossFunctionDumper : public PostprocessPlugin
{
public:
    LossFunctionDumper(std::string name, std::string path);

    ~LossFunctionDumper();

    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;

private:
    std::string path;

    bool activated = true;
    MPI_Datatype mpiReductionType;
    FILE *fdump = nullptr;
};
