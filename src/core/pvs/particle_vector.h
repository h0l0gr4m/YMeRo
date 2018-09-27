#pragma once

#include <string>
#include <core/datatypes.h>
#include <core/containers.h>
#include <core/domain.h>
#include <core/utils/pytypes.h>

#include "extra_data/extra_data_manager.h"

class ParticleVector;

enum class ParticleVectorType {
    Local,
    Halo
};

class LocalParticleVector
{
public:
    ParticleVector* pv;

    PinnedBuffer<Particle> coosvels;
    DeviceBuffer<Force> forces;
    ExtraDataManager extraPerParticle;

    // Local coordinate system; (0,0,0) is center of the local domain
    LocalParticleVector(ParticleVector* pv, int n=0);

    int size() { return np; }
    virtual void resize(const int n, cudaStream_t stream);
    virtual void resize_anew(const int n);
    
    virtual ~LocalParticleVector();

protected:
    int np;
};


class ParticleVector
{
protected:

    LocalParticleVector *_local, *_halo;
    
public:
    
    DomainInfo domain;    

    float mass;
    std::string name;
    // Local coordinate system; (0,0,0) is center of the local domain

    bool haloValid = false;
    bool redistValid = false;

    int cellListStamp{0};

    ParticleVector(std::string name, float mass, int n=0);

    LocalParticleVector* local() { return _local; }
    LocalParticleVector* halo()  { return _halo;  }

    virtual void checkpoint(MPI_Comm comm, std::string path);
    virtual void restart(MPI_Comm comm, std::string path);

    
    // Python getters / setters
    // Use default blocking stream
    std::vector<int> getIndices_vector();
    PyTypes::VectorOfFloat3 getCoordinates_vector();
    PyTypes::VectorOfFloat3 getVelocities_vector();
    PyTypes::VectorOfFloat3 getForces_vector();
    
    void setCoosVels_globally(PyTypes::VectorOfFloat6& coosvels, cudaStream_t stream=0);
    void createIndicesHost();

    void setCoordinates_vector(PyTypes::VectorOfFloat3& coordinates);
    void setVelocities_vector(PyTypes::VectorOfFloat3& velocities);
    void setForces_vector(PyTypes::VectorOfFloat3& forces);
    
    
    virtual ~ParticleVector();
    
    template<typename T>
    void requireDataPerParticle(std::string name, bool needExchange)
    {
        requireDataPerParticle<T>(name, needExchange, 0);
    }
    
    template<typename T>
    void requireDataPerParticle(std::string name, bool needExchange, int shiftDataType)
    {
        requireDataPerParticle<T>(local(), name, needExchange, shiftDataType);
        requireDataPerParticle<T>(halo(),  name, needExchange, shiftDataType);
    }

protected:
    ParticleVector(std::string name, float mass,
                   LocalParticleVector *local, LocalParticleVector *halo );

    virtual void _checkpointParticleData(MPI_Comm comm, std::string path);
    virtual void _restartParticleData(MPI_Comm comm, std::string path);

    void advanceRestartIdx();
    int restartIdx = 0;

private:

    template<typename T>
    void requireDataPerParticle(LocalParticleVector* lpv, std::string name, bool needExchange, int shiftDataType)
    {
        lpv->extraPerParticle.createData<T> (name, lpv->size());
        if (needExchange) lpv->extraPerParticle.requireExchange(name);
        if (shiftDataType != 0) lpv->extraPerParticle.requireShift(name, shiftDataType);
    }
};




