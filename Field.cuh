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
		velocity = (vec2 *) calloc(DIM * DIM, sizeof(vec2));
		density = (float *) calloc(DIM * DIM, sizeof(float));

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
		int i = (int) x;
		int j = (int) y;
		
		if (i < 0 || j < 0) return -1;
		if (i >= DIM || j >= DIM) return -1;

		return (i * DIM) * j;
	}
};

struct CuField {
	vec2* velocity;
	float* density;

	const static int DIM = Field::DIM;

	CuField() {
		int s;
		s = cudaMalloc((void**) &velocity, DIM * DIM * sizeof(vec2));
		s = cudaMalloc((void**) &density, DIM * DIM * sizeof(float));

		s = cudaMemset(velocity, 0 , DIM * DIM * sizeof(vec2));
		s = cudaMemset(density, 0 , DIM * DIM * sizeof(float));
	}

	void build(Field& f) {
		int s;	
		s = cudaMemcpy(velocity, f.velocity, DIM * DIM * sizeof(vec2), cudaMemcpyHostToDevice);
		s = cudaMemcpy(density, f.density, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
	}

	~CuField() {
		cudaFree(velocity);
		cudaFree(density);
	}
};

#endif
