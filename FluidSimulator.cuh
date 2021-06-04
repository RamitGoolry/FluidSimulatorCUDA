#ifndef __FLUIDSIMULATOR_H_
#define __FLUIDSIMULATOR_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Field.cuh"
#define DIFFUSION_RATE 1.0f
#define DT 0.1f

#define IX(i, j) ((i) + (Field::DIM) * (j))

__global__ void advect(float* initD, float* currD, vec2* currV, unsigned char * bitmap) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    float dt0 = DT * Field::DIM;

    float x_, y_;

    x_ = x - dt0 * currV[offset].x;
    y_ = y - dt0 * currV[offset].y;

    if(x_ < 0.5) x_ = 0.5;
    if(x_ > Field::DIM + 0.5) x_ = 0.5;
    
    if(y_ < 0.5) y_ = 0.5;
    if(y_ > Field::DIM + 0.5) y_ = 0.5;
    
    int i0 = x_, j0 = y_;
    int i1 = i0 + 1, j1 = j0 + 1;

    float s1 = x_ - i0;
    float s0 = 1 - s1;
    float t1 = y_ - j0;
    float t0 = 1 - t1;

    currD[offset] = s0 * (currD[IX(i0, j0)] * t0 + currD[IX(i0, j1)] * t1) + s1 * (currD[IX(i1, j0)] * t0 + currD[IX(i1, j1)] * t1);
}

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
}

#endif