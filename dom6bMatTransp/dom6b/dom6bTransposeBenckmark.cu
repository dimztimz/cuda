#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>

#include <cuda_runtime.h>

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

__global__ void transposeFullParralel1DColToRow(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int globalIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int outR = globalIdx / redici;
	int outK = globalIdx % redici;
	if (outK >= redici || outR >= koloni) {
		return;
	}
	outMatrix[outR*redici + outK] = inMatrix[outK*koloni + outR];
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

__global__ void transposeFullParralel2DBlockingColToRow(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int outR = blockDim.y * blockIdx.y + threadIdx.y;
	int outK = blockDim.x * blockIdx.x + threadIdx.x;
	if (outK >= redici || outR >= koloni) {
		return;
	}
	outMatrix[outR*redici + outK] = inMatrix[outK*koloni + outR];
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

void proveriGreski(int argc, char * argv[])
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
	transposeCpuColToRow(cpu_inMat, cpu_outMatReference, redici, koloni);
	cpuTime = omp_get_wtime() - cpuTime;
	printf("Cpu col to row\tredici=%d koloni=%d vreme=%lf\n", redici, koloni, cpuTime*1000.0);
	
	cudaMemcpy(gpu_inMat, cpu_inMat, arraySize*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	int threads1 = blockX * blockY;
	int blocks1 = redici / threads1 + (redici % threads1 != 0);

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
		printf("red vo kol 1d\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, blocks2, threads2, vreme);
		proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	}

	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	threads2 = threads1;
	blocks2 = (arraySize + threads2 - 1) / threads2;
	if (blocks2 < 65536) {
		cudaEventRecord(start);
		transposeFullParralel1DColToRow<<<blocks2, threads2>>>(gpu_inMat, gpu_outMat, redici, koloni);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
		printf("kol vo red 1d\tredici=%d koloni=%d blokovi=%d niski=%d vreme=%f\n", redici, koloni, blocks2, threads2, vreme);
		proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	}

	{
	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	dim3 threads3(blockX, blockY, 1);
	dim3 blocks3((koloni+threads3.x-1)/threads3.x,
		(redici+threads3.y-1)/threads3.y,
		1);
	if (blocks3.x * blocks3.y < 65536) {
		cudaEventRecord(start);
		transposeFullParralel2DBlocking<<<blocks3, threads3>>>(gpu_inMat, gpu_outMat, redici, koloni);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
		printf("red vo kol 2d blocking\tredici=%d koloni=%d blokovi=%dx%d=%d niski=%dx%d=%d vreme=%f\n", redici, koloni, blocks3.x, blocks3.y, blocks3.x*blocks3.y, threads3.x, threads3.y, threads3.x*threads3.y, vreme);
		proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	}
	}

	{
	cudaMemset(gpu_outMat, 0, arraySize*sizeof(float));
	dim3 threads4(blockX, blockY, 1);
	dim3 blocks4((redici + threads4.x - 1) / threads4.x,
		(koloni + threads4.y - 1) / threads4.y,
		1);
	if (blocks4.x * blocks4.y < 65536) {
		cudaEventRecord(start);
		transposeFullParralel2DBlockingColToRow<<<blocks4, threads4>>>(gpu_inMat, gpu_outMat, redici, koloni);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaMemcpy(cpu_outMat, gpu_outMat, arraySize*sizeof(float), cudaMemcpyDeviceToHost);
		printf("col vo red 2d blocking\tredici=%d koloni=%d blokovi=%dx%d=%d niski=%dx%d=%d vreme=%f\n", redici, koloni, blocks4.x, blocks4.y, blocks4.x*blocks4.y, threads4.x, threads4.y, threads4.x*threads4.y, vreme);
		proveriGreska(cpu_outMatReference, cpu_outMat, arraySize);
	}
	}
}

void benchmark()
{
	
	float * cpu_inMat, * cpu_outMat;
	float * gpu_inMat, * gpu_outMat;


	printf("CPU kol vo red\n");
	printf("N;vreme(ms)\n");
	for (int N = 32; N < 1500; N+=1) {
		cpu_inMat = (float*)calloc(N*N, sizeof(float));
		cpu_outMat = (float*)calloc(N*N, sizeof(float));
		double cpuTime = omp_get_wtime();
		transposeCpuColToRow(cpu_inMat, cpu_outMat, N, N);
		cpuTime = omp_get_wtime() - cpuTime;
		free(cpu_inMat);
		free(cpu_outMat);
		printf("%d;%lf\n", N, cpuTime*1000.0);
	}
	
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	printf("GPU celo paralelno 1d red vo kol\n");
	printf("N;blokovi;niski;vreme\n");
	for (int numThreads0 = 8; numThreads0 <= 32; numThreads0 +=4) {
		int numThreads = numThreads0 * numThreads0;
		for (int N = 32; N < 1500; N+=1) {
			int len = N*N;
			int numBlocks = (len+numThreads-1)/numThreads;
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
				printf("%d;%d;%d;%f\n", N, numBlocks, numThreads, vreme);
			}
		}
	}

	printf("GPU celo paralelno 1d kol vo red\n");
	printf("N;blokovi;niski;vreme\n");
	for (int numThreads0 = 8; numThreads0 <= 32; numThreads0 +=4) {
		int numThreads = numThreads0 * numThreads0;
		for (int N = 32; N < 1500; N+=1) {
			int len = N*N;
			int numBlocks = (len+numThreads-1)/numThreads;
			if (numBlocks < 65536) {
				cudaMalloc(&gpu_inMat, len*sizeof(float));
				cudaMalloc(&gpu_outMat, len*sizeof(float));
				cudaEventRecord(start);
				transposeFullParralel1DColToRow<<<numBlocks, numThreads>>>(gpu_inMat, gpu_outMat, N, N);
				cudaEventRecord(end);
				cudaEventSynchronize(end);
				cudaEventElapsedTime(&vreme, start, end);
				cudaFree(gpu_inMat);
				cudaFree(gpu_outMat);
				printf("%d;%d;%d;%f\n", N, numBlocks, numThreads, vreme);
			}
		}
	}

	printf("GPU celo paralelno 2d blocking red vo kol\n");
	printf("N;blokovi;blokoviX;blokoviY;niski;niskiX;niskiY;vreme\n");
	for (int numThreads0 = 8; numThreads0 <= 32; numThreads0 +=4) {
		int numThreads = numThreads0 * numThreads0;
		dim3 blockDim(numThreads0, numThreads0, 1);
		for (int N = 32; N < 1500; N+=1) {
			int len = N*N;
			dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
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
				printf("%d;%d;%d;%d;%d;%d;%d;%f\n", N, gridDim.x*gridDim.y, gridDim.x, gridDim.y, numThreads, blockDim.x, blockDim.y, vreme);
			}
		}
	}

	printf("GPU celo paralelno 2d blocking kol vo red\n");
	printf("N;blokovi;blokoviX;blokoviY;niski;niskiX;niskiY;vreme\n");
	for (int numThreads0 = 8; numThreads0 <= 32; numThreads0 +=4) {
		int numThreads = numThreads0 * numThreads0;
		dim3 blockDim(numThreads0, numThreads0, 1);
		for (int N = 32; N < 1500; N+=1) {
			int len = N*N;
			dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
			int numBlocks = gridDim.x * gridDim.y;
			if (numBlocks < 65536) {
				cudaMalloc(&gpu_inMat, len*sizeof(float));
				cudaMalloc(&gpu_outMat, len*sizeof(float));
				cudaEventRecord(start);
				transposeFullParralel2DBlockingColToRow<<<gridDim, blockDim>>>(gpu_inMat, gpu_outMat, N, N);
				cudaEventRecord(end);
				cudaEventSynchronize(end);
				cudaEventElapsedTime(&vreme, start, end);
				cudaFree(gpu_inMat);
				cudaFree(gpu_outMat);
				printf("%d;%d;%d;%d;%d;%d;%d;%f\n", N, gridDim.x*gridDim.y, gridDim.x, gridDim.y, numThreads, blockDim.x, blockDim.y, vreme);
			}
		}
	}
}

int main(int argc, char * argv[])
{
	//proveriGreski(argc, argv);
	benchmark();
	return 0;
}