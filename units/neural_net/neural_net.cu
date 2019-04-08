#define protected public
#define private public

#include <algorithm>
#include <stdio.h>
#include <vector>
#include <valarray>
#include <iterator>
#include <stdio.h>
#include <random>
#include <cstdio>
#include <typeinfo>
#include <core/interactions/calculations/NeuralNet_kernel.h>
#include <core/interactions/pairwise.impl.h>

#include <core/datatypes.h>
#include <core/containers.h>
#include <core/logger.h>

#include <gtest/gtest.h>
#include <unistd.h>

Logger logger;

TEST(NeuralNet,parallel_vs_serial)
{
  PinnedBuffer<float> Weights(22);
  float LO = -10;
  float HI = 10;
  for(auto &weight : Weights)
  {
    float r3 = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
    weight = r3;
  }


  PinnedBuffer<DPDparameter> results_cpu(20);
  PinnedBuffer<NNInput> NNInputs(20);

  //Initialize random NN Inputs
  for( auto &nninput : NNInputs)
  {
    nninput.i1 = rand() % 10;
    nninput.i2 = rand() % 10;
    nninput.i3 = rand() % 10;
    nninput.i4 = rand() % 10;
    nninput.i5 = rand() % 10;
    nninput.i6 = rand() % 10;
    nninput.v1 = rand() % 10;
    nninput.d1 = rand() % 10;
    nninput.d2 = rand() % 10;
    nninput.d3 = rand() % 10;
  }
  //
  //Calculate result in a normal way
  for(int i = 0 ; i < 20 ; i ++)
    {
      float result_alpha = 0;
      float result_gamma = 0;
      for( int input = 0; input < 11 ; input ++)
        {
          result_alpha += NNInputs[i][input]*Weights[input];
          result_gamma += NNInputs[i][input]*Weights[input+11];

        }
      results_cpu[i].alpha_p = result_alpha;
      results_cpu[i].gamma_p = result_gamma;

    }

  //Calculate Result on GPU

  Weights.uploadToDevice(0);
  NNInputs.uploadToDevice(0);
  PinnedBuffer<DPDparameter> results_gpu(20);

  auto devPtrWeights = Weights.devPtr();
  auto devPtrNNInputs = NNInputs.devPtr();
  auto devPtrresult_gpu = results_gpu.devPtr();
  int nth = 128;
  int size = 20;

  SAFE_KERNEL_LAUNCH(
    NeuralNet,getNblocks(32*size,nth),nth,0,0,
    size,devPtrresult_gpu,devPtrNNInputs,devPtrWeights);

  results_gpu.downloadFromDevice(0);

  bool correct = true;
  for(int i = 0 ; i < 20; i++)
  {
    if (!(results_cpu[i].alpha_p == results_gpu[i].alpha_p && results_cpu[i].gamma_p == results_gpu[i].gamma_p))
        correct = false;
    if (correct == false)
      break;
  }
  ASSERT_TRUE(correct);


}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  logger.init(MPI_COMM_WORLD, "dpd.log", 9);

  testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  MPI_Finalize();
  return ret;
}
