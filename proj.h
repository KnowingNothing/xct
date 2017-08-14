#ifndef _PROJ_H_
#define _PROJ_H_

#include "proto.h"

extern "C"{
void pick(int* np, int* nr);
void A(float *g,float *f);
void wray(int np,int nr,int *line, float *weight, int *numb, float *snorm);
}

#endif
