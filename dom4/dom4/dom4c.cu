#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void checkCudaError(cudaError_t err);

__global__ void cestotaBezAtomic(int * buckets, int bucketsLen, int totalThreads)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= totalThreads) {
		return;
	}
	int bid = tid % bucketsLen;
	buckets[bid] += 1;

}

__global__ void cestota(int * buckets, int bucketsLen, int totalThreads)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= totalThreads) {	
		return;
	}
	int bid = tid % bucketsLen;
	atomicAdd(&buckets[bid], 1);

}

int main(int argc, char* argv[])
{
	int bucketsSize, totalThreads, threadsPerBlock;
	int * hostBuckets, * devBuckets;

	if (argc < 4) {
		bucketsSize = 10;
		totalThreads = 10000;
		threadsPerBlock = 100;
	} else {
		sscanf(argv[1], "%d", &totalThreads);
		sscanf(argv[2], "%d", &threadsPerBlock);
		sscanf(argv[3], "%d", &bucketsSize);
	}

	hostBuckets = (int*)calloc(bucketsSize, sizeof(int));
	checkCudaError(cudaMalloc(&devBuckets, sizeof(int) * bucketsSize));

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	int numBlocks = totalThreads / threadsPerBlock + (totalThreads%threadsPerBlock != 0);

	checkCudaError(cudaMemset(devBuckets, 0, sizeof(int) * bucketsSize));
	cudaEventRecord(start);
	cestota<<<numBlocks, threadsPerBlock>>>(devBuckets, bucketsSize, totalThreads);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(hostBuckets, devBuckets, sizeof(int)*bucketsSize, cudaMemcpyDeviceToHost);
	printf("So atomic:\telementi=%d blokovi=%d niski=%d koficki=%d vreme=%f\n", totalThreads, numBlocks, threadsPerBlock, bucketsSize, vreme);

	checkCudaError(cudaMemset(devBuckets, 0, sizeof(int) * bucketsSize));
	cudaEventRecord(start);
	cestotaBezAtomic<<<numBlocks, threadsPerBlock>>>(devBuckets, bucketsSize, totalThreads);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(hostBuckets, devBuckets, sizeof(int)*bucketsSize, cudaMemcpyDeviceToHost);
	printf("Bez atomic:\telementi=%d blokovi=%d niski=%d koficki=%d vreme=%f\n", totalThreads, numBlocks, threadsPerBlock, bucketsSize, vreme);

	
	
	/*for(int i = 0; i<bucketsSize; i++) {
		printf("%d ", hostBuckets[i]);
	}
	printf("\n");*/

	return 0;
}

void checkCudaError(cudaError_t err)
{
	if (err != cudaSuccess)
		printf("%s\n", cudaGetErrorString(err));
}