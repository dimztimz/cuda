#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#include <cuda_runtime.h>

__global__ void transposeSingleThread(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	for (int i = 0; i<redici; i++) {
		for (int j = 0; j<koloni; j++) {
			outMatrix[j*redici + i] = inMatrix[i*koloni + j];
		}
	}
}

__global__ void transposeSingleThreadColToRow(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	for (int j = 0; j<koloni; j++) {
		for (int i = 0; i<redici; i++) {
			outMatrix[j*redici + i] = inMatrix[i*koloni + j];
		}
	}
}

__global__ void transposeThreadRowToCol(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	if (r >= redici) {
		return;
	}
	for (int k = 0; k<koloni; k++) {
		outMatrix[k*redici + r] = inMatrix[r*koloni + k];
	}
}

__global__ void transposeThreadColToRow(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	if (k >= koloni) {
		return;
	}
	for (int r = 0; r<redici; r++) {
		outMatrix[k*redici + r] = inMatrix[r*koloni + k];
	}
}

__global__ void transposeFullParralel1D(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int r = globalIdx / koloni;
	int k = globalIdx % koloni;
	if (r >= redici || k >= koloni) {
		return;
	}
	outMatrix[k*redici + r] = inMatrix[r*koloni + k];
}

__global__ void transposeFullParralel2DBlocking(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	if (r >= redici || k >= koloni) {
		return;
	}
	outMatrix[k*redici + r] = inMatrix[r*koloni + k];
}

void transposeCpu(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	for (int i = 0; i<redici; i++) {
		for (int j = 0; j<koloni; j++) {
			outMatrix[j*redici + i] = inMatrix[i*koloni + j];
		}
	}
}

void transposeCpuColToRow(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	for (int j = 0; j<koloni; j++) {
		for (int i = 0; i<redici; i++) {
			outMatrix[j*redici + i] = inMatrix[i*koloni + j];
		}
	}
}

bool proveriGreska(const float * referenceMat, const float * presmetanaMat, int len)
{
	bool greska = false;
	for (int i = 0; i<len; i++) {
		if (referenceMat[i] != presmetanaMat[i]) {
			greska = true;
			printf("Greska vo index %d, %f != %f\n", i, referenceMat[i], presmetanaMat[i]);
		}
	}
	return greska;
}

void initArray(float * mat, int len)
{
	for (int i = 0; i<len; i++) {
		mat[i] = (float)(i+1);
	}
}

int main(int argc, char * argv[])
{
	
	float * cpu_inMat, * cpu_outMat;
	float * gpu_inMat, * gpu_outMat;
	
	printf("CPU red vo kol\n");
	printf("N,vreme(ms)\n");
	for (int N = 32; N < 1500; N+=32) {
		cpu_inMat = (float*)calloc(N*N, sizeof(float));
		cpu_outMat = (float*)calloc(N*N, sizeof(float));
		double cpuTime = omp_get_wtime();
		transposeCpu(cpu_inMat, cpu_outMat, N, N);
		cpuTime = omp_get_wtime() - cpuTime;
		free(cpu_inMat);
		free(cpu_outMat);
		printf("%d,%lf\n", N, cpuTime*1000.0);
	}

	printf("CPU kol vo red\n");
	printf("N,vreme(ms)\n");
	for (int N = 32; N < 1500; N+=32) {
		cpu_inMat = (float*)calloc(N*N, sizeof(float));
		cpu_outMat = (float*)calloc(N*N, sizeof(float));
		double cpuTime = omp_get_wtime();
		transposeCpuColToRow(cpu_inMat, cpu_outMat, N, N);
		cpuTime = omp_get_wtime() - cpuTime;
		free(cpu_inMat);
		free(cpu_outMat);
		printf("%d,%lf\n", N, cpuTime*1000.0);
	}
	
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	printf("GPU red vo kol 1 niska\n");
	printf("N,vreme(ms)\n");
	for (int N = 32; N < 1500; N+=32) {
		cudaMalloc(&gpu_inMat, N*N*sizeof(float));
		cudaMalloc(&gpu_outMat, N*N*sizeof(float));
		cudaEventRecord(start);
		transposeSingleThread<<<1, 1>>>(gpu_inMat, gpu_outMat, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(gpu_inMat);
		cudaFree(gpu_outMat);
		printf("%d,%f\n", N, vreme);
	}
	
	printf("GPU kol vo red 1 niska\n");
	printf("N,vreme(ms)\n");
	for (int N = 32; N < 1500; N+=32) {
		cudaMalloc(&gpu_inMat, N*N*sizeof(float));
		cudaMalloc(&gpu_outMat, N*N*sizeof(float));
		cudaEventRecord(start);
		transposeSingleThreadColToRow<<<1, 1>>>(gpu_inMat, gpu_outMat, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(gpu_inMat);
		cudaFree(gpu_outMat);
		printf("%d,%f\n", N, vreme);
	}
	
	printf("GPU red vo kol, niska po redica\n");
	printf("N,blokovi,niski,vreme\n");
	for (int numThreads = 1; numThreads <= 512; numThreads <<= 1) {
		for (int N = max(32, numThreads); N < 1500; N+=32) {
			int numBlocks = N/numThreads + (N%numThreads != 0);

			cudaMalloc(&gpu_inMat, N*N*sizeof(float));
			cudaMalloc(&gpu_outMat, N*N*sizeof(float));
			cudaEventRecord(start);
			transposeThreadRowToCol<<<numBlocks, numThreads>>>(gpu_inMat, gpu_outMat, N, N);
			cudaEventRecord(end);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&vreme, start, end);
			cudaFree(gpu_inMat);
			cudaFree(gpu_outMat);
			printf("%d,%d,%d,%f\n", N, numBlocks, numThreads, vreme);
		}
	}

	printf("GPU kol vo red, niska po redica\n");
	printf("N,blokovi,niski,vreme\n");
	for (int numThreads = 1; numThreads <= 512; numThreads <<= 1) {
		for (int N = max(32, numThreads); N < 1500; N+=32) {
			int numBlocks = N/numThreads + (N%numThreads != 0);
			cudaMalloc(&gpu_inMat, N*N*sizeof(float));
			cudaMalloc(&gpu_outMat, N*N*sizeof(float));
			cudaEventRecord(start);
			transposeThreadColToRow<<<numBlocks, numThreads>>>(gpu_inMat, gpu_outMat, N, N);
			cudaEventRecord(end);
			cudaEventSynchronize(end);
			cudaEventElapsedTime(&vreme, start, end);
			cudaFree(gpu_inMat);
			cudaFree(gpu_outMat);
			printf("%d,%d,%d,%f\n", N, numBlocks, numThreads, vreme);
		}
	}

	printf("GPU celo paralelno, 1d\n");
	printf("N,blokovi,niski,vreme\n");
	for (int numThreads = 1; numThreads <= 512; numThreads <<= 1) {
		for (int N = max(32, numThreads); N < 1500; N+=32) {
			int len = N*N;
			int numBlocks = len/numThreads + (len % numThreads != 0);
			if (numBlocks < 65536) {
				cudaMalloc(&gpu_inMat, len*sizeof(float));
				cudaMalloc(&gpu_outMat, len*sizeof(float));
				cudaEventRecord(start);
				transposeFullParralel1D<<<numBlocks, numThreads>>>(gpu_inMat, gpu_outMat, N, N);
				cudaEventRecord(end);
				cudaEventSynchronize(end);
				cudaEventElapsedTime(&vreme, start, end);
				cudaFree(gpu_inMat);
				cudaFree(gpu_outMat);
				printf("%d,%d,%d,%f\n", N, numBlocks, numThreads, vreme);
			}
		}
	}

	printf("GPU celo paralelno, 2d blocking\n");
	printf("N,blokovi,blokoviX,blokoviY,niski,niskiX,niskiY,vreme\n");
	for (int numThreads = 1; numThreads <= 512; numThreads <<= 1) {
		dim3 blockDim(numThreads, 1, 1);
		for (; blockDim.x >= 1; blockDim.x >>= 1, blockDim.y <<= 1) {
			for (int N = 32; N < 1500; N+=32) {
				int len = N*N;
				dim3 gridDim(N/blockDim.x + (N%blockDim.x != 0),
					N/blockDim.y + (N%blockDim.y != 0),
					1);
				int numBlocks = gridDim.x * gridDim.y;
				if (numBlocks < 65536) {
					cudaMalloc(&gpu_inMat, len*sizeof(float));
					cudaMalloc(&gpu_outMat, len*sizeof(float));
					cudaEventRecord(start);
					transposeFullParralel2DBlocking<<<gridDim, blockDim>>>(gpu_inMat, gpu_outMat, N, N);
					cudaEventRecord(end);
					cudaEventSynchronize(end);
					cudaEventElapsedTime(&vreme, start, end);
					cudaFree(gpu_inMat);
					cudaFree(gpu_outMat);
					printf("%d,%d,%d,%d,%d,%d,%d,%f\n", N, gridDim.x*gridDim.y, gridDim.x, gridDim.y, numThreads, blockDim.x, blockDim.y, vreme);
				}
			}
		}
	}

	return 0;
}