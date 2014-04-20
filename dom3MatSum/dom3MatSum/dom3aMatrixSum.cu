#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>


//blockot i mrezata se vo edna dimenzikja
__global__ void matrixSum1D(float * d_mat1, float * d_mat2, float * d_matSuma, int red, int kol)
{
	int globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
	int redIdx = globalThreadId / kol;
	int kolIdx = globalThreadId % kol;
	if (redIdx < red && kolIdx < kol)
	{
		//int i = redIdx * kol + kolIdx;
		int i = globalThreadId; //isto so pogore
		d_matSuma[i] = d_mat1[i] + d_mat2[i];
	}
}

//blocking. i blockovite i mrezata se vo dve dimenzii
__global__ void matrixSum2DBlocking(float * d_mat1, float * d_mat2, float * d_matSuma, int red, int kol)
{
	int redIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int kolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (redIdx < red && kolIdx < kol)
	{
		int i = redIdx * kol + kolIdx;
		d_matSuma[i] = d_mat1[i] + d_mat2[i];
	}
}

//blokot e dve dimenzii, mrezata edna
__global__ void matrixSum2DBlock1DGrid(float * d_mat1, float * d_mat2, float * d_matSuma, int red, int kol)
{
	int globalThreadId = blockDim.x * blockDim.y * blockIdx.x + blockDim.x*threadIdx.y + threadIdx.x;
	int redIdx = globalThreadId / kol;
	int kolIdx = globalThreadId % kol;
	if (redIdx < red && kolIdx < kol)
	{
		//int i = redIdx * kol + kolIdx;
		int i = globalThreadId; //isto so pogore
		d_matSuma[i] = d_mat1[i] + d_mat2[i];
	}
}

//blokot e se e vo 3 dimenzii
__global__ void matrixSum3D(float * d_mat1, float * d_mat2, float * d_matSuma, int red, int kol)
{
	int blockSize = blockDim.x * blockDim.y * blockDim.z;
	int localThreadId = blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	int blockId = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
	int globalThreadId = blockId * blockSize + localThreadId;
	
	int redIdx = globalThreadId / kol;
	int kolIdx = globalThreadId % kol;
	if (redIdx < red && kolIdx < kol)
	{
		//int i = redIdx * kol + kolIdx;
		int i = globalThreadId; //isto so pogore
		d_matSuma[i] = d_mat1[i] + d_mat2[i];
	}
}

int main(int argc, char * argv[])
{
	int red, kol, ARRAY_SIZE;
	float * h_mat1, * h_mat2, * h_matSum;
	float * d_mat1, * d_mat2, * d_matSum;
	cudaEvent_t start, end;
	float vreme;
	dim3 dimGrid, dimBlock;
	int tip = 1;

	if (argc < 2) {
		return 0;
	}
	
	if (strcmp(argv[1], "1") == 0) {
		tip = 1;
	} else if (strcmp(argv[1], "3") == 0) {
		tip = 3;
	} else {
		return 0;
	}

	if (tip == 1) {
		if (argc < 5) {
			red = 32;
			kol = 32;
			dimBlock.x = 32;
		} else {
			sscanf(argv[2] ,"%d", &red);
			sscanf(argv[3] ,"%d", &kol);
			sscanf(argv[4] ,"%d", &dimBlock.x);
		}
	} else if (tip == 3) {
		if (argc < 7) {
			red = 32;
			kol = 32;
			dimBlock.x = 4;
			dimBlock.y = 4;
			dimBlock.z = 2;
		} else {
			sscanf(argv[2] ,"%d", &red);
			sscanf(argv[3] ,"%d", &kol);
			sscanf(argv[4] ,"%d", &dimBlock.x);
			sscanf(argv[5] ,"%d", &dimBlock.y);
			sscanf(argv[6] ,"%d", &dimBlock.z);
		}
	}

	ARRAY_SIZE = red * kol;

	h_mat1 = (float*)calloc(ARRAY_SIZE, sizeof(float));
	h_mat2 = (float*)calloc(ARRAY_SIZE, sizeof(float));
	h_matSum = (float*)calloc(ARRAY_SIZE, sizeof(float));

	srand(5);

	for (int i = 0; i<ARRAY_SIZE; i++)
	{
		h_mat1[i] = (float)(i/kol * 1000 + i%kol);
		h_mat2[i] = (float)(i/kol * 1000 + i%kol);
	}

	cudaMalloc(&d_mat1, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_mat2, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_matSum, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_mat1, h_mat1, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2, h_mat2, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	if (tip == 1) {
		dimBlock.y = dimBlock.z = 1;
		dimGrid.y = dimGrid.z = 1;
		dimGrid.x = ARRAY_SIZE / dimBlock.x + (ARRAY_SIZE % dimBlock.x != 0);
		matrixSum1D<<<dimGrid, dimBlock>>>(d_mat1, d_mat2, d_matSum, red, kol);
		cudaEventRecord(end);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&vreme, start, end);

		cudaMemcpy(h_matSum, d_matSum, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

		/*for (int i = 0; i<ARRAY_SIZE; i++)
		{
			printf("%.0f + %.0f = %.0f\n", h_mat1[i], h_mat2[i], h_matSum[i]);
		}*/

		/*threads = dimBlock.x * dimBlock.y * dimBlock.z;
		blocks = dimGrid.x * dimGrid.y * dimGrid.z;*/
		printf("redici=%d koloni=%d, blockDimX=%d, blockDimY=%d, gridDimX=%d, gridDimY=%d, vreme=%f\n", red, kol, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y , vreme);
	} else if (tip == 3) {
		int blockSize = dimBlock.x * dimBlock.y * dimBlock.z;
		dimGrid.y = dimGrid.z = 1;
		dimGrid.x = ARRAY_SIZE / blockSize + (ARRAY_SIZE % blockSize != 0);
		
		matrixSum3D<<<dimGrid, dimBlock>>>(d_mat1, d_mat2, d_matSum, red, kol);
		cudaEventRecord(end);
		cudaDeviceSynchronize();
		cudaEventElapsedTime(&vreme, start, end);

		cudaMemcpy(h_matSum, d_matSum, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		
		/*for (int i = 0; i<ARRAY_SIZE; i++)
		{
			printf("%.0f + %.0f = %.0f\n", h_mat1[i], h_mat2[i], h_matSum[i]);
		}*/
		
		printf("redici=%d koloni=%d, blockDimX=%d, blockDimY=%d, blockDimZ=%d, gridDimX=%d, vreme=%f\n", red, kol, dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, vreme);
	}

	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_matSum);

	free(h_mat1);
	free(h_mat2);
	free(h_matSum);

}
