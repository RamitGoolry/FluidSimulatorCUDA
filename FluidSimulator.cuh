#ifndef __FLUIDSIMULATOR_H_
#define __FLUIDSIMULATOR_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Field.cuh"

#define DIFFUSION_RATE 0.000001f
#define DT 0.01f

#define VISC 2.0f

#define ITERS 5

#define IX(i, j) ((i) + (Field::DIM) * (j))

__device__ void set_bnd_density(float * d) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    int i = offset / (Field::DIM * Field::DIM);
    int j = offset % (Field::DIM * Field::DIM);

    d[IX(0, 0)] = 0.5f * (d[IX(1, 0)] + d[IX(0, 1)]);
    d[IX(0, Field::DIM - 1)] = 0.5f * (d[IX(1, Field::DIM - 1)] + d[IX(0, Field::DIM - 2)]);
    d[IX(Field::DIM - 1, 0)] = 0.5f * (d[IX(Field::DIM - 2, 0)] + d[IX(Field::DIM - 1, 1)]);
    d[IX(Field::DIM - 1, Field::DIM - 1)] = 0.5f * (d[IX(Field::DIM - 2, Field::DIM - 1)] + d[IX(Field::DIM - 1, Field::DIM - 2)]);

    if ( i == 0 || i == Field::DIM - 1) return;
    if ( j == 0 || j == Field::DIM - 1) return;

    d[IX(i, 0)] = d[IX(i, 1)];
    d[IX(i, Field::DIM - 1)] = d[IX(i, Field::DIM - 2)];

    d[IX(0, j)] = d[IX(1, j)];
    d[IX(Field::DIM - 1, 0)] = d[IX(Field::DIM - 2, 1)];
}

__global__ void advect_density(float* currD, float* prevD,vec2* currV, unsigned char * bitmap) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    float dt0 = DT * Field::DIM;

    float x_, y_;

    x_ = x - dt0 * currV[offset].x;
    y_ = y - dt0 * currV[offset].y;

    if(x_ < 0.5) x_ = 0.5;
    if(x_ > Field::DIM + 0.5) x_ = Field::DIM + 0.5;
    
    if(y_ < 0.5) y_ = 0.5;
    if(y_ > Field::DIM + 0.5) y_ = Field::DIM + 0.5;
    
    int i0 = x_, j0 = y_;
    int i1 = i0 + 1, j1 = j0 + 1;

    float s1 = x_ - i0;
    float s0 = 1 - s1;
    float t1 = y_ - j0;
    float t0 = 1 - t1;

    currD[offset] = 
        s0 * (prevD[IX(i0, j0)] * t0 + prevD[IX(i0, j1)] * t1) + 
        s1 * (prevD[IX(i1, j0)] * t0 + prevD[IX(i1, j1)] * t1);

    set_bnd_density(currD);
}

__device__ void lin_solve_density(float* currD, float* prevD, float a, float c, int iters) {
	int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    int x = offset / (Field::DIM * Field::DIM);
    int y = offset % (Field::DIM * Field::DIM);

    if(offset >= Field::DIM * Field::DIM) return;

    int left = x == 0 ? offset - 1 : offset;
    int right = x == Field::DIM - 1 ? offset + 1 : offset;

    int top = y == 0 ? offset - Field::DIM : offset;
    int bottom = y == Field::DIM - 1 ? offset + Field::DIM : offset;

    float cRecip = 1.0 / c;

    for(int k = 0; k < iters; k++) {
        currD[offset] = (prevD[offset] + a * (prevD[top] + prevD[bottom] + prevD[left] + prevD[right])) * cRecip;
        set_bnd_density(currD);
    }
}

__global__ void diffuse_density(float* currD, float* prevD, unsigned char* bitmap) {
    float a = DT * (Field::DIM * Field::DIM) * DIFFUSION_RATE;
    lin_solve_density(currD, prevD, a, 1 + 4.0 * a, ITERS);
}

__device__ void set_bnd_velocity(vec2 * v) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    int i = offset / (Field::DIM * Field::DIM);
    int j = offset % (Field::DIM * Field::DIM);

    v[IX(0, 0)].x = (v[IX(1, 0)].x + v[IX(0, 1)].x) * 0.5f;
    v[IX(0, 0)].y = (v[IX(1, 0)].y + v[IX(0, 1)].y) * 0.5f;

    v[IX(0, Field::DIM - 1)].x = (v[IX(1, Field::DIM - 1)].x + v[IX(0, Field::DIM - 2)].x) * 0.5f;
    v[IX(0, Field::DIM - 1)].y = (v[IX(1, Field::DIM - 1)].y + v[IX(0, Field::DIM - 2)].y) * 0.5f;

    v[IX(Field::DIM - 1, 0)].x = (v[IX(Field::DIM - 2, 0)].x + v[IX(Field::DIM - 1, 1)].x) * 0.5f;
    v[IX(Field::DIM - 1, 0)].y = (v[IX(Field::DIM - 2, 0)].y + v[IX(Field::DIM - 1, 1)].y) * 0.5f;

    v[IX(Field::DIM - 1, Field::DIM - 1)].x = (v[IX(Field::DIM - 2, Field::DIM - 1)].x + v[IX(Field::DIM - 1, Field::DIM - 2)].x) * 0.5f;
    v[IX(Field::DIM - 1, Field::DIM - 1)].y = (v[IX(Field::DIM - 2, Field::DIM - 1)].y + v[IX(Field::DIM - 1, Field::DIM - 2)].y) * 0.5f;

    if ( i == 0 || i == Field::DIM - 1) return;
    if ( j == 0 || j == Field::DIM - 1) return;

    v[IX(i, 0)].x = -v[IX(i, 1)].x;
    v[IX(i, 0)].y = -v[IX(i, 1)].y;
    v[IX(i, Field::DIM - 1)].x = -v[IX(i, Field::DIM - 2)].x;
    v[IX(i, Field::DIM - 1)].y = -v[IX(i, Field::DIM - 2)].y;

    v[IX(0, j)].x = -v[IX(1, j)].x;
    v[IX(0, j)].y = -v[IX(1, j)].y;
    v[IX(Field::DIM - 1, 0)].x = -v[IX(Field::DIM - 2, 1)].x;
    v[IX(Field::DIM - 1, 0)].y = -v[IX(Field::DIM - 2, 1)].y;
}

__device__ void lin_solve_velocity(vec2* currV, vec2* prevV, float a, float c, int iters) {
	int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    int x = offset / (Field::DIM * Field::DIM);
    int y = offset % (Field::DIM * Field::DIM);

    if(offset >= Field::DIM * Field::DIM) return;

    int left = x == 0 ? offset - 1 : offset;
    int right = x == Field::DIM - 1 ? offset + 1 : offset;

    int top = y == 0 ? offset - Field::DIM : offset;
    int bottom = y == Field::DIM - 1 ? offset + Field::DIM : offset;

    float cRecip = 1.0 / c;

    for(int k = 0; k < iters; k++) {
        currV[offset].x = (prevV[offset].x + a * (prevV[top].x + prevV[bottom].x + prevV[left].x + prevV[right].x)) * cRecip;
        currV[offset].y = (prevV[offset].y + a * (prevV[top].y + prevV[bottom].y + prevV[left].y + prevV[right].y)) * cRecip;
        set_bnd_velocity(currV);
    }
}

__global__ void diffuse_velocity(vec2* currV, vec2* prevV) {
    float a = DT * (Field::DIM * Field::DIM) * VISC;
    lin_solve_velocity(currV, prevV, a, 1 + 4.0 * a, ITERS);
}

#endif