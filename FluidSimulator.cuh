#ifndef __FLUIDSIMULATOR_H_
#define __FLUIDSIMULATOR_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Field.cuh"

#define DIFFUSION_RATE 500.0f
#define DT 0.003f

#define VISC 1.0f

#define ITERS 20

#define IX(i, j) ((i) + (Field::DIM) * (j))

__device__ void set_bnd(int b, float * d) {
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

    d[IX(i, 0)] = b == 2 ? -d[IX(i, 1)] : d[IX(i, 1)];
    d[IX(i, Field::DIM - 1)] = b == 2 ? -d[IX(i, Field::DIM - 2)] : d[IX(i, Field::DIM - 2)];

    d[IX(0, j)] = b == 1 ? -d[IX(1, j)] : d[IX(1, j)];
    d[IX(Field::DIM - 1, 0)] = b == 1 ? -d[IX(Field::DIM - 2, 1)] : d[IX(Field::DIM - 2, 1)];
}

__global__ void advect_density(float* currD, float* prevD, float * u, float * v) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    float dt0 = DT * Field::DIM;

    float x_, y_;

    x_ = x - dt0 * u[offset];
    y_ = y - dt0 * v[offset];

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

    set_bnd(0, currD);
}

__global__ void diffuse_density(float* currD, float* prevD) {
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    if(offset >= Field::DIM * Field::DIM) return;
    
    int i = offset / (Field::DIM * Field::DIM);
    int j = offset % (Field::DIM * Field::DIM);

    int left = offset - 1;
    int right = offset + 1;

    int top = offset - Field::DIM;
    int bottom = offset + Field::DIM;

    float a = DT * (Field::DIM * Field::DIM) * DIFFUSION_RATE;
    float c = 1.0f * 4.0f * a;

    for(int k = 0; k < ITERS; k++) {
        currD[offset] = (prevD[offset] + a * (prevD[top] + prevD[bottom] + prevD[left] + prevD[right])) / c;
        set_bnd(0, currD);
    }
}

__global__ void diffuse_velocity(float* u, float* v, float* u0, float* v0) {
	int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    if(offset >= Field::DIM * Field::DIM) return;

    int i = offset / (Field::DIM * Field::DIM);
    int j = offset % (Field::DIM * Field::DIM);

    int left = offset - 1;
    int right = offset + 1;

    int top = offset - Field::DIM;
    int bottom = offset + Field::DIM;

    float a = DT * (Field::DIM * Field::DIM) * VISC;
    float c = 1.0f + 4.0f * a;

    for(int k = 0; k < ITERS; k++) {
        u[offset] = (u0[offset] + a * (u0[top] + u0[bottom] + u0[left] + u0[right])) / c;
        v[offset] = (v0[offset] + a * (v0[top] + v0[bottom] + v0[left] + v0[right])) / c;
        set_bnd(1, u);
        set_bnd(2, v);
    }
}

__global__ void advect_velocity(float* u, float* v, float* u0, float* v0) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    float dt0 = DT * Field::DIM;

    float x_, y_;

    x_ = x - dt0 * u[offset];
    y_ = y - dt0 * v[offset];

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

    u[offset] = 
        s0 * (u0[IX(i0, j0)] * t0 + u0[IX(i0, j1)] * t1) + 
        s1 * (u0[IX(i1, j0)] * t0 + u0[IX(i1, j1)] * t1);

    v[offset] = 
        s0 * (v0[IX(i0, j0)] * t0 + v0[IX(i0, j1)] * t1) + 
        s1 * (v0[IX(i1, j0)] * t0 + v0[IX(i1, j1)] * t1);

    set_bnd(1, u);
    set_bnd(2, v);
}

__global__ void project(float* u, float*v, float* p, float* div) {
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    float h = 1.0f / Field::DIM;

    int i = offset / (Field::DIM * Field::DIM);
    int j = offset % (Field::DIM * Field::DIM);

    if ( i == 0 || i == Field::DIM - 1) return;
    if ( j == 0 || j == Field::DIM - 1) return;

    div[IX(i, j)] = -0.5f * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
    p[IX(i, j)] = 0;

    set_bnd(0, div);
    set_bnd(0, p);

    for(int k = 0; k < 20; k++) {
        p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i, j - 1)] + p[IX(i, j + 1)]) * 0.25f;

        __syncthreads();
    }

    set_bnd(0, p);

    u[IX(i, j)] -= (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / (2*h);
    v[IX(i, j)] -= (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / (2*h);

    set_bnd(1, u);
    set_bnd(2, v);
}

#endif