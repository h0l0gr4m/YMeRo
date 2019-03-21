#include "NNInputs.h"

__global__ void computeNNInputs(typename NNInput_Computation::ViewType view, NNInput_Computation nninputs)
{
    const int particleId= blockIdx.x*blockDim.x + threadIdx.x;
    if (particleId >= view.size) return;
    const auto dstP = nninputs.read(view,particleId);
    nninputs(dstP,particleId);
};
