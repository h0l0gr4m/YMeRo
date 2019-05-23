#pragma once

#include <core/containers.h>
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/folders.h>
#include <core/utils/common.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include "interface.h"

class ParticleVector;

namespace LossDensityFunction
{
using ReductionType = double;
}

class LossDensityFunctionPlugin : public SimulationPlugin
{
public:
  LossDensityFunctionPlugin(const YmrState *state, std::string name, std::string pvName,const float viscosity,
                     int dumpEvery);

  ~LossDensityFunctionPlugin();

  void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

  void beforeIntegration(cudaStream_t stream) override;
  void serializeAndSend(cudaStream_t stream) override;
  void handshake() override;

  bool needPostproc() override { return true; }

private:
  std::string pvName;
  int dumpEvery;
  bool needToSend = false;


  PinnedBuffer<LossDensityFunction::ReductionType> localLossDensityFunction {1};
  YmrState::TimeType savedTime = 0;

  std::vector<char> sendBuffer;

  ParticleVector *pv;
  const float viscosity;
};


class LossDensityFunctionDumper : public PostprocessPlugin
{
public:
  LossDensityFunctionDumper(std::string name, std::string path);

  ~LossDensityFunctionDumper();

  void deserialize(MPI_Status& stat) override;
  void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
  void handshake() override;

private:
  std::string path;

  bool activated = true;
  MPI_Datatype mpiReductionType;
  FILE *fdump = nullptr;
};
