#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

int main(int argc, char * argv[])
{
	int redici, koloni, blockX, blockY = 1;
	float * cpu_inMat, * cpu_outMatReference, * cpu_outMat;
	float * gpu_inMat, * gpu_outMat;
	
	if (argc < 4) {
		redici = koloni = 32;
		blockX = 32; blockY = 16;
	} else {
		sscanf(argv[1], "%d", &redici);
		sscanf(argv[2], "%d", &koloni);
		sscanf(argv[3], "%d", &blockX);
		if (argc > 4) {
			sscanf(argv[4], "%d", &blockY);
		}
	}
	const int arraySize = redici * koloni;
	cpu_inMat = (float*)calloc(arraySize, sizeof(float));
	cpu_outMat = (float*)calloc(arraySize, sizeof(float));
	cpu_outMatReference = (float*)calloc(arraySize, sizeof(float));

	cudaMalloc(&gpu_inMat, arraySize * sizeof(float));
	cudaMalloc(&gpu_outMat, arraySize * sizeof(float));
	
	for (int i = 0; i<arraySize; i++) {
		cpu_inMat[i] = (float)(i+1);
	}
	
	double cpuTime = omp_get_wtime();
	transposeCpu(cpu_inMat, cpu_outMatReference, redici, koloni);
	cpuTime = omp_get_wtime() - cpuTime;
	printf("Cpu Row to col\tredici=%d koloni=%d vreme=%lf\n", redici, koloni, cpuTime*1000.0);

	cpuTime = omp_get_wtime();
	transposeCpuColToRow(cpu_inMat, cpu_outMatReference, redici, koloni);
	cpuTime = omp_get_wtime() - cpuTime;
	printf("Cpu col to row\tredici=%d koloni=%d vreme=%lf\n", redici, koloni, cpuTime*1000.0);
	
	cudaMemcpy(gpu_inMat, cpu_inMat, arraySize*sizeof(float), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	cudaEventRecord(start);
	transposeSingleThread<<<1, 1>>>(gpu_inMat, gpu_outMat, redici, koloni);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
	printf("1 niska, red vo kol\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, 1, 1, vreme);
	proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	
	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	cudaEventRecord(start);
	transposeSingleThreadColToRow<<<1, 1>>>(gpu_inMat, gpu_outMat, redici, koloni);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
	printf("1 niska, kol vo red\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, 1, 1, vreme);
	proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);

	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	int threads1 = blockX * blockY;
	int blocks1 = redici / threads1 + (redici % threads1 != 0);
	cudaEventRecord(start);
	transposeThreadRowToCol<<<blocks1, threads1>>>(gpu_inMat, gpu_outMat, redici, koloni);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
	printf("red niski, red vo kol\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, blocks1, threads1, vreme);
	proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);

	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	blocks1 = koloni / threads1 + (koloni % threads1 != 0);
	cudaEventRecord(start);
	transposeThreadColToRow<<<blocks1, threads1>>>(gpu_inMat, gpu_outMat, redici, koloni);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
	printf("kol niski, kol vo red\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, blocks1, threads1, vreme);
	proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);


	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	int threads2 = threads1;
	int blocks2 = arraySize / threads2 + (arraySize % threads2 != 0);
	if (blocks2 < 65536) {
		cudaEventRecord(start);
		transposeFullParralel1D<<<blocks2, threads2>>>(gpu_inMat, gpu_outMat, redici, koloni);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
		printf("red*kol niski 1d\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, blocks2, threads2, vreme);
		proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	}

	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	dim3 threads3(blockX, blockY, 1);
	dim3 blocks3(koloni/threads3.x + (koloni%threads3.x != 0),
		redici/threads3.y + (redici%threads3.y != 0),
		1);
	if (blocks3.x * blocks3.y < 65536) {
		cudaEventRecord(start);
		transposeFullParralel2DBlocking<<<blocks3, threads3>>>(gpu_inMat, gpu_outMat, redici, koloni);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
		printf("red*kol niski blocking\tredici=%d koloni=%d blokovi=%dx%d=%d niski=%dx%d=%d vreme=%f\n", redici, koloni, blocks3.x, blocks3.y, blocks3.x*blocks3.y, threads3.x, threads3.y, threads3.x*threads3.y, vreme);
		proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	}

	return 0;
}