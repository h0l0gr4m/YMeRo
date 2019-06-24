//Copy kernel, for copying initial a and gamma value to every particles
__global__ void copy_kernel(DPDparameter* devPointer1,DPDparameter* devPointer2,int np, float a, float gamma)
{

   const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
   if (dstId >= np) return;
   devPointer1[dstId].alpha_p = a;
   devPointer1[dstId].gamma_p = gamma;
   devPointer2[dstId].alpha_p = a;
   devPointer2[dstId].gamma_p = gamma;

}


__global__ void copy_viscosity(NNInput* devPV,int np, float viscosity)
{
   const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
   if (dstId >= np) return;
   devPV[dstId].vis = viscosity;

}
