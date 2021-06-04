#ifndef __VEC2_H__
#define __VEC2_H__

struct vec2 {
	float x, y;

	vec2() {
		x = y = 0;
	}

	vec2(float x_, float y_) {
		x = x_;
		y = y_;
	}

	__device__ vec2 operator + (vec2& other) {
		return vec2(x + other.x, y + other.y);
	}

	__device__ vec2 operator - (vec2& other) {
		return vec2(x - other.x, y - other.y);
	}

	__device__ vec2 operator * (float scalar) {
		return vec2(x * scalar, y * scalar);
	}

	__device__ float dot(vec2& other) {
		return x * other.x + y * other.y;
	}
};

#endif
