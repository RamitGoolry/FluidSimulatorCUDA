#ifndef __FIELD_H__
#define __FIELD_H__

#include "vec2.h"
#include <stdlib.h>

#include <random>

struct Field {
	vec2* velocity;
	float* density;
	
	vec2 cellSize;

	const static int DIM = 640;

	Field() {
		float width = DIM;
		float height = DIM;

		velocity = (vec2 *) malloc(DIM * DIM * sizeof(vec2));
		density = (float *) malloc(DIM * DIM * sizeof(vec2));

		cellSize.x = width / DIM;
		cellSize.y = height / DIM;

		std::default_random_engine generator;
		std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

		for(int i = 0; i < DIM * DIM; i++) {
			velocity[i] = vec2(dist(generator), dist(generator));

			float x = (float) i / (float) (DIM * DIM); // TODO Define with Bezier
			density[i] = int(x >= 0.49 && x <= 0.51);
		}
	}

	~Field() {
		free(velocity);
		free(density);
	}

	int toIndex(float x, float y) {
		int i = (int) (x / cellSize.x);
		int j = (int) (y / cellSize.y);
		
		if (i < 0 || j < 0) return -1;
		if (i >= DIM || j >= DIM) return -1;

		return (i * DIM) * j;
	}

	vec2 toCoordinate(int i, int j) {
		float x = i / DIM;
		float y = j / DIM;
	
		return vec2((float) ((x + 0.5f) * cellSize.x), (float) ((y + 0.5f) * cellSize.y));
	}

	vec2 getVelocity(int i) {
		if(i == -1) return vec2(0, 0);
		return velocity[i];
	}

	vec2 getVelocity(float x, float y) {
		int i = toIndex(x, y);
		return getVelocity(i);
	}

	float getDensity(int i) {
		if (i == -1) return 0;
		return density[i];
	}

	float getDensity(float x, float y) {
		int i = toIndex(x, y);
		if (i == -1) return 0;
		return density[i];
	}
};

struct CuField {
	vec2* velocity;
	float* density;

	vec2 cellSize;

	const static int DIM = 640;

	CuField() {}

	CuField(Field &f) {
		float width = DIM;
		float height = DIM;

		cudaMalloc((void**) &velocity, DIM * DIM * sizeof(vec2));
		cudaMalloc((void**) &density, DIM * DIM * sizeof(float));

		cudaMemcpy(f.velocity, velocity, DIM * DIM * sizeof(vec2), cudaMemcpyDeviceToHost);
		cudaMemcpy(f.density, density, DIM * DIM * sizeof(float), cudaMemcpyDeviceToHost);
	}

	~CuField() {
		cudaFree(velocity);
		cudaFree(density);
	}

	__device__ int toIndex(float x, float y) {
		int i = (int) (x / cellSize.x);
		int j = (int) (y / cellSize.y);
		
		if (i < 0 || j < 0) return -1;
		if (i >= DIM || j >= DIM) return -1;

		return (i * DIM) * j;
	}

	__device__ vec2 toCoordinate(int i, int j) {
		float x = i / DIM;
		float y = j / DIM;
	
		return vec2((float) ((x + 0.5f) * cellSize.x), (float) ((y + 0.5f) * cellSize.y));
	}

	__device__ vec2 getVelocity(int i) {
		if(i == -1) return vec2(0, 0);
		return velocity[i];
	}

	__device__ vec2 getVelocity(float x, float y) {
		int i = toIndex(x, y);
		return getVelocity(i);
	}

	__device__ float getDensity(int i) {
		if (i == -1) return 0;
		return density[i];
	}

	__device__ float getDensity(float x, float y) {
		int i = toIndex(x, y);
		if (i == -1) return 0;
		return density[i];
	}
};

#endif
