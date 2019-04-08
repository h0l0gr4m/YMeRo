#pragma once

#include <cassert>
#include <type_traits>

#include <core/datatypes.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/interactions/pairwise_interactions/smartdpd.h>
#include <core/interactions/calculations/NNInputs.h>
#include <core/pvs/views/pv.h>
#include <cstdlib>
#include <math.h>




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
__device__ inline  float warpReduce(float val)
{

#pragma unroll
    for (int offset = 8 ; offset>0 ; offset/=2 )
    {
        val += warpShflDown(val, offset);
    }
    return val;
}



__global__ void NeuralNet(int size,DPDparameter *pv1DPDparameter, NNInput *pv1NNInputs, float *Weights)
{
  // printf("size: %d \n" ,size);
	int thread = getGlobalIdx_3D_3D();
  if (thread > 32*size-1)
    return;

  uint32_t weight_index = 0;
  uint32_t input_index = 0 ;
  float value = 0;
  uint32_t warpid = thread / 32;
  uint32_t laneid = thread % 32;
  uint32_t particle = warpid;

  if (laneid >15 && laneid < 27)
  {
      input_index = laneid % 16;
      weight_index = laneid % 16 +11;
  }
  else if (laneid < 11)
  {
      input_index = laneid % 16;
      weight_index = laneid;

  }
  else
  {
   return;
  }

  value = pv1NNInputs[particle][input_index]*Weights[weight_index];
  value = warpReduce(value);
  // printf(" thread: %d, warpid : %d , laneid: %d , particle: %d , input_index: %d , weight_index: %d , pv1NNInputs[particle[input_index] : %f ,Weights[weight_index] : %f , value: %f  \n "   ,thread, warpid,laneid,particle,input_index,weight_index,pv1NNInputs[particle][input_index] , Weights[weight_index] ,value);

  if(laneid % 16 == 0)
  {
    if(weight_index<11)
    {
       value = (value + sqrt(value*value +1))/2 ;
       pv1DPDparameter[particle].alpha_p = value;
       // printf("pv1DPDparameter[particle].alpha_p: %f , value: %f, particle: %d, laneid: %d , weight_index: %d  \n", pv1DPDparameter[particle].alpha_p ,value, particle, laneid,weight_index);
    }


    else
    {
       value = (value + sqrt(value*value +1))/2 ;
       pv1DPDparameter[particle].gamma_p=value;
       // if (pv1DPDparameter[particle].gamma_p != 20.25)
       // printf("pv1DPDparameter[particle].gamma_p: %f , value: %f, particle: %d, laneid: %d , weight_index: %d , inpud_index: %d \n", pv1DPDparameter[particle].gamma_p ,value, particle, laneid,weight_index,input_index);
    }
  }



}
