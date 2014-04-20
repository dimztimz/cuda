#include <stdlib.h>
#include <stdio.h>
#include <math.h>


__global__ void matrixProductAndSum(float * d_mat1, float * d_mat2, float * d_matProd, float * d_matSum, int red1, int kol1red2, int kol2)
{
	int redIdx = blockDim.y * blockIdx.y + threadIdx.y;
	int kolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (redIdx < red1 && kolIdx < kol2)
	{
		int k;
		int i = redIdx * kol2 + kolIdx;
		d_matSum[i] = d_mat1[i] + d_mat2[i];
		d_matProd[i] = 0;
		for (k = 0; k < kol1red2; k++)
		{
			d_matProd[i] += d_mat1[redIdx * kol1red2 + k] * d_mat2[k * kol2 + kolIdx];
		}
	}
}

int main(int argc, char * argv[])
{
	int red, kol, ARRAY_SIZE, i, j;
	float * h_mat1, * h_mat2, * h_matProd, * h_matSum;
	float * d_mat1, * d_mat2, * d_matProd, * d_matSum;
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
	h_matProd = (float*)calloc(ARRAY_SIZE, sizeof(float));
	h_matSum = (float*)calloc(ARRAY_SIZE, sizeof(float));

	srand(5);

	for (int i = 0; i<ARRAY_SIZE; i++)
	{
		h_mat1[i] = i/kol * 1000 + i%kol;
		h_mat2[i] = i/kol * 1000 + i%kol;
	}

	cudaMalloc(&d_mat1, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_mat2, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_matProd, ARRAY_SIZE * sizeof(float));
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
	matrixProductAndSum<<<dimGrid, dimBlock>>>(d_mat1, d_mat2, d_matProd, d_matSum, red, kol, red);
	cudaEventRecord(end);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&vreme, start, end);

	cudaMemcpy(h_matProd, d_matProd, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	/*for (i = 0; i<red; i++)
	{
		for (j = 0; j<kol; j++)
		{
			printf("%4.0f ", h_mat1[i*kol + j]);
		}
		printf("\n");
	}
	printf("\n");
	for (i = 0; i<red; i++)
	{
		for (j = 0; j<kol; j++)
		{
			printf("%4.0f ", h_matProd[i*kol + j]);
		}
		printf("\n");
	}*/
	/*threads = dimBlock.x * dimBlock.y * dimBlock.z;
	blocks = dimGrid.x * dimGrid.y * dimGrid.z;*/
	printf("redici=%d koloni=%d, blockDimX=%d, blockDimY=%d, gridDimX=%d, gridDimY=%d, vreme=%f\n", red, kol, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y , vreme);

	cudaFree(d_mat1);
	cudaFree(d_mat2);
	cudaFree(d_matProd);
	cudaFree(d_matSum);

	free(h_mat1);
	free(h_mat2);
	free(h_matProd);
	free(h_matSum);

}
