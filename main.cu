#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "proj.h"
#include "proto.h"
#include "utility.h"

#include "kernel.cu"
#include "main.h"


float g[NPROJ*NRAY]; //sinogram
float f[(IMGSIZE+2)*(IMGSIZE+2)]; //image
float v[(IMGSIZE+2)*(IMGSIZE+2)]; //edge information

int blocks=NPROJ;
int threads=((NRAY-1)/32+1)*32;

void MSbeam();

void ray2thread(int* angle, int *position);

int main(int argc, char *argv[]) {
	
    read_phantom(f, "data/phant_g1.dat");
    
    normalize(f);
    write_image(f, "data/std.dat");
    
    A(g, f);


    MSbeam();
    
    normalize(f);
    
    write_image(f, "data/img.dat");
    
    write_image(v, "data/edge.dat");
    
    return 0;
}
void MSbeam() {

    int i;
    for (i = 0; i<(IMGSIZE+2)*(IMGSIZE+2); ++i) {
        f[i] = 0.;
        v[i] = 1.;
    }
    
	float *d_g=NULL;
	float *d_f=NULL;
	float *d_v=NULL;
	int *d_angle=NULL;
	int *d_position=NULL;

    cudaError_t err = cudaSuccess;
	err=cudaMalloc((void**)&d_f,sizeof(float)*(IMGSIZE+2)*(IMGSIZE+2));
    if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector f (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }

	err=cudaMalloc((void**)&d_v,sizeof(float)*(IMGSIZE+2)*(IMGSIZE+2));
    if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector v (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }

	err=cudaMalloc((void**)&d_g,sizeof(float)*NPROJ*NRAY);
    if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector g (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }
	
	err=cudaMalloc((void**)&d_angle,sizeof(int)*blocks*threads);
    if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector angle (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }
	
	err=cudaMalloc((void**)&d_position,sizeof(int)*blocks*threads);
    if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device vector position (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }
	
	
	err=cudaMemcpy(d_f, f, sizeof(float)*(IMGSIZE+2)*(IMGSIZE+2), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector f from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaMemcpy(d_v, v, sizeof(float)*(IMGSIZE+2)*(IMGSIZE+2), cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector v from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaMemcpy(d_g, g, sizeof(float)*NPROJ*NRAY, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


    float lambda = 0.001;


	int angle[blocks*threads];
	int position[blocks*threads];

	ray2thread(angle, position);

	err=cudaMemcpy(d_angle, angle, sizeof(int)*blocks*threads, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector angle from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaMemcpy(d_position, position, sizeof(int)*blocks*threads, cudaMemcpyHostToDevice);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector position from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

    Node *records = NULL;
	Node *d_records = NULL;
	records = (Node*)alloc(sizeof(Node) * MAX_RECORDS);
	err = cudaMalloc((void**)&d_records, sizeof(Node) * MAX_RECORDS);
    if (err != cudaSuccess){
		fprintf(stderr, "Failed to allocate device records (error code %s)!\n", cudaGetErrorString(err));
    	exit(EXIT_FAILURE);
    }

	Lock lock;
	Global_counter counter;

	for(int j = 0; i < 10; ++j)
		XCT_Reconstruction<<<blocks, threads>>>(d_f, d_v, d_g, d_angle, d_position, lambda, d_records, lock, counter);

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch MumfordShah kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err=cudaMemcpy(f, d_f, sizeof(float)*(IMGSIZE+2)*(IMGSIZE+2), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector f from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	err=cudaMemcpy(v, d_v, sizeof(float)*(IMGSIZE+2)*(IMGSIZE+2), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy vector v from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err=cudaMemcpy(records, d_records, sizeof(Node) * (MAX_RECORDS), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess){
		fprintf(stderr, "Failed to copy records from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}


	err=cudaFree(d_f);
    if (err != cudaSuccess){
       	fprintf(stderr, "Failed to free vector d_f (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	err=cudaFree(d_v);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free vector d_v (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	err=cudaFree(d_g);
    if (err != cudaSuccess){
       	fprintf(stderr, "Failed to free vector d_f (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

	err=cudaFree(d_records);
    if (err != cudaSuccess){
       	fprintf(stderr, "Failed to free d_records (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
}

void ray2thread(int *angle, int *position){
	int i;
	for (i=0; i<blocks*threads; ++i){
		angle[i]=rand()%NPROJ;
		position[i]=rand()%NRAY;
	}
}
