template<typename Calculation>
__global__ void computeNNInputs(const int np, Calculation calculation)
{
    const int particleId= blockIdx.x*blockDim.x + threadIdx.x;
    if (particleId >= np) return;
    calculation(particleId);
};
