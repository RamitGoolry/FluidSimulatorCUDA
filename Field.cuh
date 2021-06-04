#ifndef __FIELD_H__
#define __FIELD_H__

#include "vec2.h"
#include <stdlib.h>

#include <random>

struct Field {
	vec2* velocity;
	float* density;

	const static int DIM = 640;

	Field() {
		velocity = (vec2 *) malloc(DIM * DIM * sizeof(vec2));
		density = (float *) malloc(DIM * DIM * sizeof(vec2));

		std::default_random_engine generator;
		std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

		for(int i = 0; i < DIM * DIM; i++) {
			velocity[i] = vec2(dist(generator), dist(generator));

			float x = (float) i / (float) (DIM * DIM); // TODO Define with Bezier
			density[i] = int(x >= 0.495 && x <= 0.505);
		}
	}

	~Field() {
		free(velocity);
		free(density);
	}

	int toIndex(float x, float y) {
		int i = (int) x;
		int j = (int) y;
		
		if (i < 0 || j < 0) return -1;
		if (i >= DIM || j >= DIM) return -1;

		return (i * DIM) * j;
	}

	vec2 toCoordinate(int i, int j) {
		float x = i / DIM;
		float y = j / DIM;
	
		return vec2((float) (x + 0.5f) , (float) (y + 0.5f));
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

	const static int DIM = 640;

	CuField() {}

	void build(Field& f) {
		cudaMalloc((void**) &velocity, DIM * DIM * sizeof(vec2));
		cudaMalloc((void**) &density, DIM * DIM * sizeof(float));

		cudaMemcpy(f.velocity, velocity, DIM * DIM * sizeof(vec2), cudaMemcpyHostToDevice);
		cudaMemcpy(f.density, density, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
	}

	~CuField() {
		cudaFree(velocity);
		cudaFree(density);
	}

	__device__ int toIndex(float x, float y) {
		int i = (int) x;
		int j = (int) y;
		
		if (i < 0 || j < 0) return -1;
		if (i >= DIM || j >= DIM) return -1;

		return (i * DIM) * j;
	}

	__device__ vec2 toCoordinate(int i, int j) {
		float x = i / DIM;
		float y = j / DIM;
	
		return vec2((float) (x + 0.5f), (float) (y + 0.5f));
	}

	__device__ vec2 getVelocity(int i) {
		if(i == -1) return vec2(0, 0);
		return velocity[i];
	}

	__device__ vec2 getVelocity(float x, float y) {
		int i = toIndex(x, y);
		if (i == -1) return vec2(0, 0);
		return velocity[i];
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
