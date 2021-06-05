#ifndef __FIELD_H__
#define __FIELD_H__

#include <stdlib.h>

#include <random>

struct Field {
	float *u, *v;
	float *density;

	const static int DIM = 480;

	Field() {
		u = (float *) calloc(DIM * DIM, sizeof(float));
		v = (float *) calloc(DIM * DIM, sizeof(float));
		density = (float *) calloc(DIM * DIM, sizeof(float));

		std::default_random_engine generator;
		std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

		for(int i = 0; i < DIM * DIM; i++) {
			u[i] = dist(generator);
			v[i] = dist(generator);

			float x = (float) (i / (DIM)) / DIM; // TODO Define with Bezier
			density[i] = int(x >= 0.48 && x <= 0.52);
		}
	}

	~Field() {
		free(u);
		free(v);
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
	float *u, *v;
	float *density;

	const static int DIM = Field::DIM;

	CuField() {
		cudaMalloc((void**) &u, DIM * DIM * sizeof(float));
		cudaMalloc((void**) &v, DIM * DIM * sizeof(float));
		cudaMalloc((void**) &density, DIM * DIM * sizeof(float));

		cudaMemset(u, 0 , DIM * DIM * sizeof(float));
		cudaMemset(v, 0 , DIM * DIM * sizeof(float));
		cudaMemset(density, 0 , DIM * DIM * sizeof(float));
	}

	void build(Field& f) {
		cudaMemcpy(u, f.u, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(v, f.v, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(density, f.density, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
	}

	~CuField() {
		cudaFree(u);
		cudaFree(v);
		cudaFree(density);
	}
};

#endif