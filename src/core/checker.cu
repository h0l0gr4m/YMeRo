#include <core/checker.h> 
#include <core/pvs/views/pv.h>
#include <stdlib.h>

__global__ void check_pvs(PVview pv_view,int* flag,float dt)
{
    const int particleId= blockIdx.x*blockDim.x + threadIdx.x;
    if (particleId >= pv_view.size) return;
    Particle p = Particle(pv_view.particles,particleId);
    if(!isfinite(dot(p.u,p.u)))
             *flag = 1;
    if(!isfinite(dot(p.r,p.r)))
    	     *flag = 1; 
}


Checker::Checker(const YmrState *state, std::string name) : 
	YmrSimulationObject(state,name)  

{
flag.resize_anew(1);
flag[0]=0; 
}

Checker::~Checker()=default;


void Checker::check(ParticleVector *pv,cudaStream_t stream)
{
  using ViewType = PVview;
  ViewType  view (pv,pv->local());
  int nth = 128;
  SAFE_KERNEL_LAUNCH(
   check_pvs,getNblocks(view.size,nth),nth,0,stream,
    view,flag.devPtr(),state->dt);
  
 flag.downloadFromDevice(stream);
 if(flag[0]==1)
  {
	printf("Flag has changed! Something is wrong, flag: %d \n ", flag[0]);
	exit(0);        
  }

 

} 
