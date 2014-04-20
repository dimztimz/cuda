#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <omp.h>

__global__ void dotKernel1(float * in1, float * in2, float * out, int len)
{
	extern __shared__ float cache[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int cidx = threadIdx.x;
	if (i >= len) {
		cache[cidx] = 0.0f;
		return;
	}
	cache[cidx] = in1[i] * in2[i];
	__syncthreads();

	i = blockDim.x >> 1;
	while (i != 0) {
		if (cidx < i) {
			cache[cidx] += cache[cidx + i];
		}
		__syncthreads();
		i >>= 1;
	}
	if (cidx == 0) {
		out[blockIdx.x] = cache[0];
	}
}

__global__ void dotKernel2(float * partialSumsIn, float * partialSumsOut, int len)
{
	extern __shared__ float cache[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int cidx = threadIdx.x;
	cache[cidx] = 0.0;
	if (i >= len) {
		
		return;
	} 
		cache[cidx] = partialSumsIn[i];
	
	
	__syncthreads();

	i = blockDim.x >> 1;
	while (i != 0) {
		if (cidx < i) {
			//printf("c[%d]=%f += c[%d]=%f, ", cidx, cache[cidx], cidx+i, cache[cidx + i]);
			cache[cidx] += cache[cidx + i];
		}
		__syncthreads();
		i >>= 1;
	}
	if (cidx == 0) {
		partialSumsOut[blockIdx.x] = cache[0];
	}
}

float dot(const float * v1, const float * v2, int len, int numThreads)
{
	float * d_v1, * d_v2, * d_out;
	//float * h_outDebug;

	float ret;
	int numBlocks = (len + numThreads - 1)/numThreads;
	cudaMalloc(&d_v1, len*sizeof(float));
	cudaMalloc(&d_v2, len*sizeof(float));
	cudaMalloc(&d_out, numBlocks*sizeof(float));

	cudaMemcpy(d_v1, v1, len*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v2, v2, len*sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	dotKernel1<<<numBlocks, numThreads, numThreads*sizeof(float)>>>(d_v1, d_v2, d_out, len);
	float * d_partSumsIn = d_v1, * d_partSumsOut = d_out;
	while(numBlocks > 1) {
		int len = numBlocks;
		numBlocks = (numBlocks + numThreads - 1)/numThreads;
		float * tmp = d_partSumsIn;
		d_partSumsIn = d_partSumsOut;
		d_partSumsOut = tmp;
		dotKernel2<<<numBlocks, numThreads, numThreads*sizeof(float)>>>(d_partSumsIn, d_partSumsOut, len);
	}
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("gpu dot reduce: elem=%d threadsPerBlk=%d vreme=%f\n", len, numThreads, vreme);
	cudaMemcpy(&ret, d_partSumsOut, sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_out);

	return ret;
}

float cpuDot(const float * v1, const float * v2, int len) {
	float sum = 0.0f;
	for(int i = 0; i < len; i++) {
		sum += v1[i] * v2[i];
	}
	return sum;
}

int main(int argc, char * argv[])
{
	float * h_v1, * h_v2;
	float dotCpu, dotGpu;
	int len, numThreads;

	if (argc < 3) {
		len = 1024;
		numThreads = 128;
	} else {
		sscanf(argv[1], "%d", &len);
		sscanf(argv[2], "%d", &numThreads);
	}

	h_v1 = (float*)calloc(len, sizeof(float));
	h_v2 = (float*)calloc(len, sizeof(float));

	for(int i = 0; i < len; i++) {
		h_v1[i] = 1.0f;
		h_v2[i] = (float)(i+1);
	}
	double vreme = omp_get_wtime();
	dotCpu = cpuDot(h_v1, h_v2, len);
	vreme = omp_get_wtime() -  vreme;
	printf("cpu dot: vreme=%lf\n", vreme*1000.0);

	dotGpu = dot(h_v1, h_v2, len, numThreads);
	if (dotCpu != dotGpu) {
		printf("greska %f != %f\n", dotCpu, dotGpu);
	} else {
		printf("tocno %f\n", dotGpu);
	}

	free(h_v1);
	free(h_v2);
	return 0;
}