#ifndef __FLUIDSIMULATOR_H_
#define __FLUIDSIMULATOR_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Field.cuh"
#define DIFFUSION_RATE 5.0f
#define DT 0.1f

__global__ void diffuse(float* initD, float* currD, float* prevD, unsigned char* bitmap) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    int left = x == 0 ? offset - 1 : offset;
    int right = x == Field::DIM - 1 ? offset + 1 : offset;

    int top = y == 0 ? offset - Field::DIM : offset;
    int bottom = y == Field::DIM - 1 ? offset + Field::DIM : offset;

    float a = DT * (Field::DIM * Field::DIM) * DIFFUSION_RATE;

    currD[offset] = initD[offset] + a * (prevD[top] + prevD[bottom] + 
            prevD[left] + prevD[right]) / (1.0 + 4.0 * a);

    unsigned char grey = initD[offset] * 255;

    bitmap[(offset << 2) + 0] = grey; 
    bitmap[(offset << 2) + 1] = grey;
    bitmap[(offset << 2) + 2] = grey;
    bitmap[(offset << 2) + 3] = (unsigned char) 255;
}

#endif