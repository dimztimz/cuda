#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace std;

__device__ int juliaGPU(float zReal, float zImag) {
	cuFloatComplex z, c;
	z.x = zReal; z.y = zImag;
	c.x = -0.8f; c.y = 0.156f;

	for (int i = 0; i<256; i++) {
		z = cuCmulf(z, z);
		z = cuCaddf(z, c);
		if (cuCabsf(z) > 2.0f) {
			return i;
		}
	}
	return 0;
}

__global__ void juliaKernel(unsigned char * rgbaArray, int sirina, int visina, float xMin, float xMax, float yMin, float yMax)
{
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (y >= visina || x >= sirina) {
		return;
	}
	float zReal = xMin + x*(xMax-xMin)/sirina;
	float zImag = yMin + y*(yMax-yMin)/visina;
	int i = 4*(y*sirina + x);
	rgbaArray[i] = 0;
	rgbaArray[i+1] = juliaGPU(zReal, zImag);
	rgbaArray[i+2] = 0;
	rgbaArray[i+3] = 255;

}

void fillJuliaGPU(unsigned char * rgbaArray, int sirina, int visina, float xMin, float xMax, float yMin, float yMax)
{
	unsigned char * d_rgbaArray;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMalloc(&d_rgbaArray, sizeof(unsigned char) * 4 * sirina * visina);
	dim3 blockDim(8, 16, 1);
	dim3 gridDim(sirina / blockDim.x + (sirina % blockDim.x != 0), 
		visina / blockDim.y + (visina % blockDim.y != 0),
		1);
	cudaEventRecord(start);
	juliaKernel<<<gridDim, blockDim>>>(d_rgbaArray, sirina, visina, xMin, xMax, yMin, yMax);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("julia gpu: sir=%d vis=%d thrX=%d thrY=%d vreme=%f\n", sirina, visina, blockDim.x, blockDim.y, vreme);
	cudaMemcpy(rgbaArray, d_rgbaArray, sizeof(unsigned char) * 4 * sirina * visina, cudaMemcpyDeviceToHost);
	cudaFree(d_rgbaArray);
}