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

	Field() {
		start = vec2(0, 0);
		vec2 end = vec2(DIM, DIM);

		float width = end.x - start.x;
		float height = end.y - start.y;

		this->start = start;

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
		int i = (int) ((x - start.x) / cellSize.x);
		int j = (int) ((y - start.y) / cellSize.y);
		
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

#endif
