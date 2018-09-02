#ifndef _UTILITY_H_
#define _UTILITY_H_

#include "proto.h"

extern "C"{
float sqr(float x);

void read_phantom(float *img, const char *file_name);

void write_image(float *img, const char *file_name);

void write_sinogram(float *sinogram, const char *file_name);

void normalize(float *img);
}

static void handle_error(cudaError_t err, const char* msg, const char* file, int line)
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Error in file %s, line %d. [%s] \n\tmsg:{%s}\n", file, line, cudaGetErrorString(err), msg);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err, msg) (handle_error(err, msg, __FILE__, __LINE__))

typedef struct{
    Node *pre;
    int pos;
    float delta;
} Node;

typedef struct{
    int *mutex;
    Lock()
    {
        char *msg = new char[100];
        msg = "Malloc lock";
        HANDLE_ERROR(cudaMalloc((void**)&mutex, sizeof(int)), msg);
        msg = "Initiate lock";
        HANDLE_ERROR(cudaMemset(mutex, 0, sizeof(int)), msg);
        free(msg);
    }
    ~Lock()
    {
        cudaFree(mutex);
    }
    __device__ void lock()
    {
        while(atomicCAS(mutex, 0, 1) != 0);
        __threadfence();
    }

    __device__ void unlock()
    {
        __threadfence();
        atomicExch(mutex, 0);
    }
} Lock;

typedef struct{
    int *counter;
    Global_counter(int initial = 0)
    {
        char *msg = new char[100];
        msg = "Malloc counter";
        HANDLE_ERROR(cudaMalloc((void**)&counter, sizeof(int)), msg);
        msg = "Initailize counter";
        HANDLE_ERROR(cudaMemcpy(counter, &initial, sizeof(int), cudaMemcpyHostToDevice), msg);
    }
    ~Global_counter()
    {
        cudaFree(counter);
    }
} Global_counter;

#endif
