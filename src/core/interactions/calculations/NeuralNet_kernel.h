#pragma once

#include <cassert>
#include <type_traits>
#include <vector>
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



__global__ void LinearNeuralNet(int size,DPDparameter *pv1DPDparameter, NNInput *pv1NNInputs, float *Weights)
{
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

  if(laneid % 16 == 0)
  {
    if(weight_index<11)
    {
       value = (value + sqrt(value*value +1))/2 ;
       pv1DPDparameter[particle].alpha_p = value;
    }


    else
    {
       value = (value + sqrt(value*value +1))/2 ;
       pv1DPDparameter[particle].gamma_p=value;
    }
  }



}
__global__ void NonLinearNeuralNet_1(int size,Intermediate_Input *pvIntermediate_Inputs, NNInput *pvNNInputs, float *Weights)
{

  int thread = getGlobalIdx_3D_3D();
  if (thread > 4*32*size-1)
    return;

  uint32_t weight_index = 0;
  uint32_t input_index = 0 ;
  float value = 0;
  uint32_t warpid = thread / 32;
  uint32_t laneid = thread % 32;
  uint32_t particle = warpid/4;

  if (laneid >15 && laneid <27)
  {
      input_index = laneid % 16;
      weight_index = laneid % 16 +(warpid%4)*22+11;
  }
  else if (laneid < 11)
  {
      input_index = laneid % 16;
      weight_index = laneid + (warpid%4)*22;

  }
  else
  {
   return;
  }
if(particle < 3000)
value = pvNNInputs[particle][input_index]*Weights[weight_index];
value = warpReduce(value);

  if(laneid % 16 == 0)
  {
    if(weight_index<11)
    {
      if(value>0)
      pvIntermediate_Inputs[particle][(warpid%4)*2+1]=value;
      else
      pvIntermediate_Inputs[particle][(warpid%4)*2+1]=0;

    }

   else
    {
      if(value>0)
      pvIntermediate_Inputs[particle][(warpid%4)*2+1]=value;
      else
      pvIntermediate_Inputs[particle][(warpid%4)*2+1]=0;

    }
  }

}
__global__ void NonLinearNeuralNet_2(int size,Intermediate_Input *pvIntermediate_Inputs, DPDparameter *pvDPDparameters, float *Weights)
{
  int thread = getGlobalIdx_3D_3D();
  if (thread > 32*size-1)
    return;

  uint32_t weight_index = 0;
  uint32_t input_index = 0 ;
  float value = 0;
  uint32_t warpid = thread / 32;
  uint32_t laneid = thread % 32;
  uint32_t particle = warpid;

  if (laneid >15 && laneid < 24)
  {
      input_index = laneid % 16;
      weight_index = laneid % 16 +96;
  }
  else if (laneid < 9)
  {
      input_index = laneid % 16;
      weight_index = laneid+88;

  }
  else
  {
   return;
  }

  value = pvIntermediate_Inputs[particle][input_index]*Weights[weight_index];
  value = warpReduce(value);

  if(laneid % 16 == 0)
  {
    if(weight_index<96)
    {
       value = (value + sqrt(value*value +1))/2 ;
       pvDPDparameters[particle].alpha_p = value;
    }


    else
    {
       value = (value + sqrt(value*value +1))/2 ;
       pvDPDparameters[particle].gamma_p=value;
    }
 
  }
}

