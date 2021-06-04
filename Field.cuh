#ifndef __FIELD_H__
#define __FIELD_H__

#include "vec2.h"
#include <stdlib.h>

#include <random>

struct Field {
	vec2* velocity;
	float* density;
	
	vec2 start, cellSize;

	const static int DIM = 640;

	Field(vec2 start, vec2 end) {
		if(end.x < start.x) {
			vec2 temp = start;
			start = end;
			end = temp;
		}

		float width = end.x - start.x;
		float height = end.y - start.y;

		this->start = start;

		velocity = (vec2 *) calloc(DIM * DIM, sizeof(vec2));

		cellSize.x = width / DIM;
		cellSize.y = height / DIM;

		std::default_random_engine generator;
		std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

		for(int i = 0; i < DIM * DIM; i++) {
			velocity[i] = vec2(dist(generator), dist(generator));

			int x = i / DIM; // TODO Define with Bezier
			density[i] = int(x >= 0.48 && x <= 0.52);
		}
	}

	__device__ int toIndex(float x, float y) {
		int i = (int) ((x - start.x) / cellSize.x);
		int j = (int) ((y - start.y) / cellSize.y);
		
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
		return getDensity(i);
	}
};

#endif
