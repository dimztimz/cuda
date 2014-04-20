//Domasna 2A, Dimitrij Mijoski 111132

#include <stdlib.h>
#include <stdio.h>
#include <math.h>


using namespace std;

__global__ void matrixSum(float * d_mat1, float * d_mat2, float * d_matSuma, int red, int kol)
{
	int redIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int kolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (redIdx < red && kolIdx < kol)
	{
		int i = redIdx * kol + kolIdx;
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
	//int blocks, threads;

	if (argc < 5) {
		red = 32;
		kol = 32;
		dimBlock.x = 32;
		dimBlock.y = 16;
	} else {
		sscanf(argv[1] ,"%d", &red);
		sscanf(argv[2] ,"%d", &kol);
		sscanf(argv[3] ,"%d", &dimBlock.x);
		sscanf(argv[4] ,"%d", &dimBlock.y);
	}

	ARRAY_SIZE = red * kol;

	h_mat1 = (float*)calloc(ARRAY_SIZE, sizeof(float));
	h_mat2 = (float*)calloc(ARRAY_SIZE, sizeof(float));
	h_matSum = (float*)calloc(ARRAY_SIZE, sizeof(float));

	srand(5);

	for (int i = 0; i<ARRAY_SIZE; i++)
	{
		h_mat1[i] = i/kol * 1000 + i%kol;
		h_mat2[i] = i/kol * 1000 + i%kol;
	}

	cudaMalloc(&d_mat1, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_mat2, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_matSum, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_mat1, h_mat1, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat2, h_mat2, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	/*dim3 dimGrid(kol/4 + (kol%4 != 0) , red/4 + (red%4 != 0));
	dim3 dimBLock(4,4);*/
	dimGrid.x = kol/dimBlock.x + (kol%dimBlock.x != 0);
	dimGrid.y = red/dimBlock.y + (red%dimBlock.y != 0);
	matrixSum<<<dimGrid, dimBlock>>>(d_mat1, d_mat2, d_matSum, red, kol);
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

	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_matSum);

	free(h_mat1);
	free(h_mat2);
	free(h_matSum);

}
