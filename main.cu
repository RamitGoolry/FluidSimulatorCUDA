#include "cpu_anim.h"
#include "Field.cuh"
#include "FluidSimulator.cuh"

struct DataBlock {
	unsigned char *dev_bitmap;
	CPUAnimBitmap *bitmap;

	Field field;
	CuField *cuField, *prev_cuField;

	DataBlock() {
		cuField = new CuField();
		prev_cuField = new CuField();

		prev_cuField->build(field);
	}
};

__global__ void copy_density_to_bitmap(unsigned char * bitmap, const float * density) {	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	unsigned char grey = density[offset] * 255;
 
	bitmap[(offset * 4) + 0] = grey;
	bitmap[(offset * 4) + 1] = grey;
	bitmap[(offset * 4) + 2] = grey;
	bitmap[(offset * 4) + 3] = 255;
}

__global__ void copy_velocity_to_bitmap(unsigned char * bitmap, const float * us, const float * vs) {	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	float u = us[offset];
	float v = vs[offset];
 
	bitmap[(offset * 4) + 0] = (u + 1) / 2.0f * 255;
	bitmap[(offset * 4) + 1] = 0;
	bitmap[(offset * 4) + 2] = (v + 1) / 2.0f * 255;
	bitmap[(offset * 4) + 3] = 255;
}

__global__ void copy_DEBUG_bitmap(unsigned char * bitmap, const float* density, const float * us, const float * vs) {	
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset = x + y * blockDim.x * gridDim.x;

	float u = us[offset];
	float v = vs[offset];
 
	bitmap[(offset * 4) + 0] = (u + 1) / 2.0f * 255;
	bitmap[(offset * 4) + 1] = density[offset] * 255;
	bitmap[(offset * 4) + 2] = (v + 1) / 2.0f * 255;
	bitmap[(offset * 4) + 3] = 255;
}

void navier_step(DataBlock * d, int ticks) {
	// Define number of Blocks and Threads
	dim3 blocks(Field::DIM/16, Field::DIM/16);
	dim3 threads(16, 16);

	// Diffusion of the Velocity field
	// diffuse_velocity <<<blocks, threads>>> (
	// 	d->cuField->u, d->cuField->v, 
	// 	d->prev_cuField->u, d->prev_cuField->v
	// );

	// project <<<blocks, threads>>> (
	// 	d->prev_cuField->u, d->prev_cuField->v,
	// 	d->cuField->u, d->cuField->v 
	// );

	// advect_velocity <<<blocks, threads>>> (
	// 	d->cuField->u, d->cuField->v,
	// 	d->prev_cuField->u, d->prev_cuField->v
	// );


	// project <<<blocks, threads>>> (
	// 	d->cuField->u, d->cuField->v,
	// 	d->prev_cuField->u, d->prev_cuField->v
	// );

	// Diffusion
	diffuse_density<<<blocks, threads>>> (
		d->cuField->density, d->prev_cuField->density
	);

	// Advection
	advect_density<<<blocks, threads>>> (
		d->cuField->density, d->prev_cuField->density, 
		d->cuField->u, d->cuField->v
	);

	// Displaying Bitmap
	// copy_DEBUG_bitmap <<<blocks, threads>>> (
	// 	d->dev_bitmap, d->cuField->density,
	// 	d->cuField->u, d->cuField->v
	// );

	copy_density_to_bitmap <<<blocks, threads>>> (
		d->dev_bitmap, d->cuField->density
	);

	CuField * temp = d->cuField;
	d->cuField = d->prev_cuField;
	d->prev_cuField = temp;

	// Returning what we computed
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

	// Manually setting Output Bitmap to the field
	for(int i = 0; i < Field::DIM * Field::DIM; i++) {
		float grey = data.field.density[i];
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