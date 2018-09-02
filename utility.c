#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "proto.h"

float sqr(float x) {return x*x;}

void read_phantom(float *img, const char *file_name) {

    FILE *fin = fopen(file_name,"r");
	memset(img, sizeof(img), 0);

    int i,j;
    for (i = 1; i<(IMGSIZE+1); ++i)
		for (j=1; j<(IMGSIZE+1); ++j)
        	fscanf(fin,"%f",&img[i*(IMGSIZE+2)+j]);

    fclose(fin);
}

void normalize(float *img) {
    float minval = 1e30,maxval = -1e30;
    int i;
    for (i = 0; i<(IMGSIZE+2)*(IMGSIZE+2); ++i) {
        float tmp = img[i];
        if (minval>tmp) minval = tmp;
        if (maxval<tmp) maxval = tmp;
    }
    maxval -= minval;
    for (i = 0; i<(IMGSIZE+2)*(IMGSIZE+2); ++i)
        img[i] = (img[i]-minval)/maxval*255.;
}

void write_image(float *img, const char *file_name) {
    FILE *fout = fopen(file_name,"w");
    int i,j;
    for (i = 1; i<(IMGSIZE+1); ++i) {
        for (j = 1; j<(IMGSIZE+1); ++j)
            fprintf(fout,"%f ",img[i*(IMGSIZE+2)+j]);
        fprintf(fout,"\n");
    }
    fclose(fout);
}

void write_sinogram(float *sinogram, const char *file_name) {
    FILE *fout = fopen(file_name,"w");
    int i, j;
    for (i = 0; i<NPROJ; ++i) {
    	for (j = 0; j<NRAY; ++j) {
       		fprintf(fout,"%f ",sinogram[i*NRAY+j]);
		}
        fprintf(fout,"\n");
    }
    fclose(fout);
}
