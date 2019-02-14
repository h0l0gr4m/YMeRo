#pragma once

#include <cassert>
#include <type_traits>

#include <core/datatypes.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/interactions/pairwise_interactions/smartdpd.h>
#include <core/interactions/calculations/FlowProperties.h>
#include <core/interactions/calculations/NNInputs.h>
#include <core/pvs/views/pv.h>




__device__ int getGlobalIdx_3D_3D(){
int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}


//****************************************************************************
//if iteration is equal to 4 , this is a fourth warp reduction
//****************************************************************************
__device__ inline  float warpReduce(float val, int iteration)
{

#pragma unroll
    for (int offset = 1 ; offset <= iteration; offset*=2 )
    {
        val += warpShflDown(val, offset);
    }
    return val;
}



__global__ void NeuralNet(int size, int iteration,DPDparameter *pv1DPDparameter, NNInput *pv1NNInputs, float *Weights)
{

	int thread = getGlobalIdx_3D_3D();
  // printf("%d \n ",thread);
  if (thread > 16*(size-1))
    return;

  uint32_t laneid = thread % 32;
  uint32_t warpid = thread / 32;
  uint32_t particle = (warpid * 2) + 1 + laneid/ 16 ;
  uint32_t input_index = (laneid-16) % 8 ;
  uint32_t weight_index = laneid % 16;
  float value = pv1NNInputs[particle][input_index]*Weights[weight_index];
  printf("value: %f , input_index: %d, particle: %d , NNInput[particle][input_index]: %f , Weights[weight_index]: %f \n" , value,input_index,particle ,pv1NNInputs[particle][input_index],Weights[weight_index]);
  value = warpReduce(value,iteration);
  if(laneid % 8 == 0)
  {  if(weight_index<8)
    {
      pv1DPDparameter[particle].alpha_p=value;
      // printf("thread: %d ,warpid: %d, laneid: %d, particle: %d, alpha_value: %f \n" ,thread,warpid,laneid,particle,pv1DPDparameter[particle].alpha_p);
    }
    else
      pv1DPDparameter[particle].gamma_p=value;
      // printf("thread: %d ,warpid: %d, laneid: %d, particle: %d, gammma_value: %f \n" ,thread,warpid,laneid,particle,pv1DPDparameter[particle].gamma_p);


  }
}
