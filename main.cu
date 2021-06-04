#include "cpu_anim.h"
#include "Field.cuh"

struct DataBlock {
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;

	Field field;
	// CuField cuField, prev_cuField;
};

__global__ void kernel(unsigned char *ptr, int ticks) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;
}

void navier_step(DataBlock * d, int ticks) { 
	dim3 blocks(Field::DIM/16, Field::DIM/16);
	dim3 threads(16, 16);

	kernel<<<blocks, threads>>> (d->dev_bitmap, ticks);

	cudaMemcpy(
			d->bitmap->get_ptr(), 
			d->dev_bitmap, 
			d->bitmap->image_size(),
			cudaMemcpyDeviceToHost
	);
}

void cleanup(DataBlock *d) {
	cudaFree(d->dev_bitmap);
}

int main( void ) {
	DataBlock data;
	CPUAnimBitmap bitmap(Field::DIM, Field::DIM, &data);

	for(int i = 0; i < Field::DIM * Field::DIM; i++) {
		float grey = data.field.getDensity(i);
		bitmap.pixels[(i * 4) + 0] = (int) (grey * 255.0f);
		bitmap.pixels[(i * 4) + 1] = (int) (grey * 255.0f);
		bitmap.pixels[(i * 4) + 2] = (int) (grey * 255.0f);
		bitmap.pixels[(i * 4) + 3] = 255;
	}

	data.bitmap = &bitmap;

	cudaMalloc((void**) &data.dev_bitmap, bitmap.image_size());

	cudaMemcpy(
		data.dev_bitmap,
		data.bitmap->get_ptr(),
		data.bitmap->image_size(),
		cudaMemcpyHostToDevice
	);

	bitmap.anim_and_exit(
			(void (*) (void *, int)) navier_step,
			(void (*) (void*)) cleanup);
}
