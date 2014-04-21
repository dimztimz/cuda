#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>
#include <omp.h>

__global__ void suma1NiskaGPU(const float * in, float * out, int len)
{
	float sum = 0.0f;
	for(int i = 0; i<len; i++) {
		sum += in[i];
	}
	out[0] = sum;
}


template <int BLOCK_WIDTH>
__global__ void reduceSosedniElementi(const float * in, float * out, int len)
{
	__shared__ float cache[BLOCK_WIDTH];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	float vrednost = 0;
	if (idx < len) {
		vrednost = in[idx];
	}
	cache[tid] = vrednost;
	__syncthreads();

	for(int pomestuvanje = 1; pomestuvanje < blockDim.x; pomestuvanje <<= 1) {
		int i = tid * 2 * pomestuvanje;
		if (i < blockDim.x) {
			cache[i] += cache[i+pomestuvanje];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		out[blockIdx.x] = cache[0];
	}
}

template <int BLOCK_WIDTH>
__global__ void reduceSosedniElementiZapisiSosedno(const float * in, float * out, int len)
{
	__shared__ float cache[BLOCK_WIDTH];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	float vrednost = 0;
	if (idx < len) {
		vrednost = in[idx];
	}
	cache[tid] = vrednost;
	__syncthreads();

	for(int povtoruvanje = blockDim.x >> 1; povtoruvanje != 0; povtoruvanje >>= 1) {
		int i = tid * 2;
		if (tid < povtoruvanje) {
			float sum = cache[i] + cache[i+1];
			//__syncthreads(); //raboti i bez ova, iako teoretski ne treba da raboti
			cache[tid] = sum;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		out[blockIdx.x] = cache[0];
	}
}

template <int BLOCK_WIDTH>
__global__ void reduceSoSkok(const float * in, float * out, int len)
{
	__shared__ float cache[BLOCK_WIDTH];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	float vrednost = 0;
	if (idx < len) {
		vrednost = in[idx];
	}
	cache[tid] = vrednost;
	__syncthreads();

	for(int skok = blockDim.x >> 1; skok != 0; skok >>= 1) {
		if (tid < skok) {
			cache[tid] += cache[tid+skok];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		out[blockIdx.x] = cache[0];
	}
}

float cpuSum(const float * niza, int len)
{
	float suma = 0.0f;
	for (int i = 0; i<len; i++) {
		suma += niza[i];
	}
	return suma;
}

bool proveriGreska(float cpuRefSum, float sum)
{
	if (cpuRefSum != sum) {
		printf("greska %f != %f\n", cpuRefSum, sum);
		return true;
	}
	//printf("tocno %f == %f\n", cpuRefSum, sum);
	return false;
}

int main()
{
	const int len = 512;
	float h_niza[len];
	float refSum, reduceSum;
	srand((unsigned int)time(NULL));
	for (int i = 0; i<len; i++) {
		h_niza[i] = (float)(rand()%5000);
	}
	refSum = cpuSum(h_niza, len);

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	int numThreads = len;
	int numBlocks = (len + numThreads - 1)/numThreads;
	float * d_niza, *d_nizaOut;
	cudaMalloc(&d_niza, sizeof(float)*len);
	cudaMalloc(&d_nizaOut, sizeof(float)*numBlocks);
	cudaMemcpy(d_niza, h_niza, sizeof(float)*len, cudaMemcpyHostToDevice);

	//zagrevanje
	reduceSosedniElementi<len><<<numBlocks, numThreads>>>(d_niza, d_nizaOut, len);

	cudaEventRecord(start);
	suma1NiskaGPU<<<1, 1>>>(d_niza, d_nizaOut, len);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(&reduceSum, d_nizaOut, sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreska(refSum, reduceSum);
	printf("suma 1 nis vreme=%f\n", vreme);
	
	cudaEventRecord(start);
	reduceSosedniElementi<len><<<numBlocks, numThreads>>>(d_niza, d_nizaOut, len);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(&reduceSum, d_nizaOut, sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreska(refSum, reduceSum);
	printf("reduce sosedni vreme=%f\n", vreme);

	cudaEventRecord(start);
	reduceSosedniElementiZapisiSosedno<len><<<numBlocks, numThreads>>>(d_niza, d_nizaOut, len);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(&reduceSum, d_nizaOut, sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreska(refSum, reduceSum);
	printf("reduce sosedni zapisi sosedno vreme=%f\n", vreme);

	cudaEventRecord(start);
	reduceSoSkok<len><<<numBlocks, numThreads>>>(d_niza, d_nizaOut, len);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	cudaMemcpy(&reduceSum, d_nizaOut, sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreska(refSum, reduceSum);
	printf("reduce so skok zapisi sosedno vreme=%f\n", vreme);

}