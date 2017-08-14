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

#endif
