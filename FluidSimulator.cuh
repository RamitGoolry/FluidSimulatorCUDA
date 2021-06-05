#ifndef __FLUIDSIMULATOR_H_
#define __FLUIDSIMULATOR_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Field.cuh"

#define SCALAR 0
#define HORIZONTAL 1
#define VERTICAL 2

#define DIFFUSION_RATE 0.1f
#define DT 0.005f

#define VISC 10.0f

#define ITERS 5

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

    d[IX(i, 0)] = b == VERTICAL ? -d[IX(i, 1)] : d[IX(i, 1)];
    d[IX(i, Field::DIM - 1)] = b == VERTICAL ? -d[IX(i, Field::DIM - 2)] : d[IX(i, Field::DIM - 2)];

    d[IX(0, j)] = b == HORIZONTAL ? -d[IX(1, j)] : d[IX(1, j)];
    d[IX(Field::DIM - 1, j)] = b == HORIZONTAL ? -d[IX(Field::DIM - 2, j)] : d[IX(Field::DIM - 2, j)];
}

__global__ void advect(int b, float* currD, float* prevD, float * u, float * v) {
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    if(offset >= Field::DIM*Field::DIM) return;

    float dt0 = DT * Field::DIM;

    float x, y;

    x = x_ - dt0 * u[offset];
    y = y_ - dt0 * v[offset]; 

    x = max(0.5f, min(Field::DIM - 1.5, x));
    y = max(0.5f, min(Field::DIM - 1.5, y));    
    
    int i0 = x, j0 = y;
    int i1 = i0 + 1, j1 = j0 + 1;

    float s1 = x - i0;
    float s0 = 1 - s1;
    float t1 = y - j0;
    float t0 = 1 - t1;

    currD[offset] = 
        s0 * (prevD[IX(i0, j0)] * t0 + prevD[IX(i0, j1)] * t1) + 
        s1 * (prevD[IX(i1, j0)] * t0 + prevD[IX(i1, j1)] * t1);

    set_bnd(b, currD);
}

__global__ void diffuse(int b, float* currD, float* prevD) {
    int x_ = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x_ + y_ * blockDim.x * gridDim.x;

    if(offset >= Field::DIM * Field::DIM) return;

    int left = offset - 1;
    int right = offset + 1;

    int top = offset - Field::DIM;
    int bottom = offset + Field::DIM;

    float a = DT * (Field::DIM * Field::DIM) * DIFFUSION_RATE;
    float c = 1.0f * 4.0f * a;

    for(int k = 0; k < ITERS; k++) {
        currD[offset] = (prevD[offset] + a * (prevD[top] + prevD[bottom] + prevD[left] + prevD[right])) / c;
        set_bnd(b, currD);
    }
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

    set_bnd(SCALAR, div);
    set_bnd(SCALAR, p);

    for(int k = 0; k < 20; k++) {
        p[IX(i, j)] = (div[IX(i, j)] + p[IX(i - 1, j)] + p[IX(i, j - 1)] + p[IX(i, j + 1)]) * 0.25f;

        __syncthreads();
    }

    set_bnd(0, p);

    u[IX(i, j)] -= (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / (2*h);
    v[IX(i, j)] -= (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / (2*h);

    set_bnd(HORIZONTAL, u);
    set_bnd(VERTICAL, v);
}

#endif