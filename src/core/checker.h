#pragma once 
#include <vector>
#include <core/datatypes.h>
#include <core/ymero_object.h>
#include <core/containers.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/pvs/views/pv.h>


class Checker : public YmrSimulationObject
{

public:
	Checker(const YmrState *state, std::string name);
	~Checker();	
       
        void check(ParticleVector *pv,cudaStream_t stream);
        PinnedBuffer<int> flag; 
};
