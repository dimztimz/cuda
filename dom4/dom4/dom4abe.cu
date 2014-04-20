#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <cfloat>

#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;

__global__ void vectorSum(float * d_niza1, float * d_niza2, float * d_nizaSuma)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	d_nizaSuma[i] = d_niza1[i] + d_niza2[i];
}

__global__ void vectorSumSharedMem(float * d_niza1, float * d_niza2, float * d_nizaSuma)
{
	extern __shared__ float buff[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int shIdx1 = threadIdx.x;
	int shIdx2 = blockDim.x + threadIdx.x;
	buff[shIdx1] = d_niza1[i];
	buff[shIdx2] = d_niza2[i];
	//__syncthreads();
	//buff[shIdx1] += buff[shIdx2];
	d_nizaSuma[i] = buff[shIdx1] + buff[shIdx2];
}

__global__ void vectorRazlikaSosedi(const float * const vlez, float * const izlez, int lenVlez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez - 1) {
		return;
	}
	izlez[i] = vlez[i+1] - vlez[i];
}

__global__ void vectorRazlikaSosediSharedMem(const float * const vlez, float * const izlez, int lenVlez)
{
	extern __shared__ float buff[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int buffIdx = threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	buff[buffIdx] = vlez[i];
	if (buffIdx == blockDim.x - 1) {
		if (i != lenVlez - 1)
			izlez[i] = vlez[i+1] - buff[buffIdx];
		return;
	}
	__syncthreads();
	izlez[i] = buff[buffIdx+1] - buff[buffIdx];
}

__global__ void vectorRazlikaSosediSharedMemv2(const float * const vlez, float * const izlez, int lenVlez)
{
	extern __shared__ float buff[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	int buffIdx = threadIdx.x;
	buff[buffIdx] = vlez[i];
	if (i == lenVlez - 1) {
		return;
	}
	if (buffIdx == blockDim.x - 1) {
		izlez[i] = vlez[i+1] - buff[buffIdx];
		return;
	}
	__syncthreads();
	izlez[i] = buff[buffIdx+1] - buff[buffIdx];
}

__global__ void vectorRazlikaSosediSharedMemv3(const float * const vlez, float * const izlez, int lenVlez)
{
	extern __shared__ float buff[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	int buffIdx = threadIdx.x;
	buff[buffIdx] = vlez[i];
	__syncthreads();
	if (buffIdx == blockDim.x - 1) {
		if (i != lenVlez - 1)
			izlez[i] = vlez[i+1] - buff[buffIdx];
		return;
	}
	izlez[i] = buff[buffIdx+1] - buff[buffIdx];
}

__global__ void vectorRazlikaSosediSharedMemv4Faza1(const float * const vlez, float * const izlez, int lenVlez)
{
	extern __shared__ float buff[];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	int buffIdx = threadIdx.x;
	buff[buffIdx] = vlez[i];
	__syncthreads();
	izlez[i] = buff[buffIdx+1] - buff[buffIdx];
}

__global__ void vectorRazlikaSosediSharedMemv4Faza2(const float * const vlez, float * const izlez, int lenVlez)
{
	int prevBlockIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int i = prevBlockIdx * blockDim.x + blockDim.x - 1;
	if (i >= lenVlez - 1) {
		return;
	}
	izlez[i] = vlez[i+1] - vlez[i];
}

__global__ void vectorRazlikaSosediSharedMemv5Faza1(const float * const vlez, float * const izlez, int lenVlez)
{
	extern __shared__ float buff[];
	int i = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	int buffIdx1 = threadIdx.x;
	buff[buffIdx1] = vlez[i];
	int i2 = i + blockDim.x;
	if (i2 >= lenVlez) {
		__syncthreads();
		izlez[i] = buff[buffIdx1+1] - buff[buffIdx1];
		return;
	}
	int buffIdx2 = blockDim.x + buffIdx1;
	buff[buffIdx2] = vlez[i2];
	__syncthreads();
	izlez[i] = buff[buffIdx1+1] - buff[buffIdx1];
	izlez[i2] = buff[buffIdx2+1] - buff[buffIdx2];
}

__global__ void vectorRazlikaSosediSharedMemv5Faza2(const float * const vlez, float * const izlez, int lenVlez)
{
	int prevBlockIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int i = prevBlockIdx * 2 * blockDim.x + 2*blockDim.x - 1;
	if (i >= lenVlez - 1) {
		return;
	}
	izlez[i] = vlez[i+1] - vlez[i];
}

__global__ void vectorMax(const float vlez[], float * izlezMax, int lenVlez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	int tmp = *((int*)(&vlez[i])); //bezvezna finta, sporedba na pozitiven float ieee754 e ista kako signed int
	atomicMax((int*)izlezMax, tmp);
}

__global__ void vectorMaxv2(const float vlez[], float * izlezMax, int lenVlez)
{
	__shared__ int blockMax[1];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= lenVlez) {
		return;
	}
	blockMax[0] = 0;
	//__syncthreads();
	int tmp = *((int*)(&vlez[i])); //bezvezna finta, sporedba na pozitiven float ieee754 e ista kako signed int
	atomicMax((int*)blockMax, tmp);
	if (threadIdx.x == 0) {
		__syncthreads();
		atomicMax((int*)izlezMax, *((int*)blockMax));
	}
}

int proveriGreskaVectorSum(float * h_niza1, float * h_niza2, float * h_nizaRez, int len)
{
	int greska = 0, i;
	for (i = 0; i<len; i++) {
		if (h_niza1[i] + h_niza2[i] != h_nizaRez[i]) {
			greska = 1;
			printf("Greska vo index %d, %.0f + %.0f != %.0f\n", i, h_niza1[i], h_niza2[i], h_nizaRez[i]);
		}
	}
	return greska;
}

int proveriGreskaRazlikaSosedi(float * h_niza1, float * h_nizaRez, int len) {
	int greska = 0, i;
	for (i = 0; i<len-1; i++) {
		if (h_nizaRez[i] != h_niza1[i+1] - h_niza1[i]) {
			greska = 1;
			printf("Greska vo index %d, %.0f - %.0f != %.0f\n", i, h_niza1[i+1], h_niza1[i], h_nizaRez[i]);
		}
	}
	return greska;
}

int proveriGreskaCpuMax(float * h_niza1, float maxIn, int len)
{
	int greska = 0, i;
	float max = -FLT_MIN;
	for (i = 0; i<len; i++) {
		if (h_niza1[i] > max) {
			max = h_niza1[i];
		}
	}
	if (max != maxIn) {
		greska = 1;
		printf("Greska max: %f != %f\n", maxIn, max);
	}
	return greska;
}

int main(int argc, char * argv[])
{
	int blocks, threads;

	if (argc < 3) {
		blocks = 1;
		threads = 32;
	} else {
		sscanf(argv[1] ,"%d", &blocks);
		sscanf(argv[2] ,"%d", &threads);
	}

	int ARRAY_SIZE = blocks * threads;

	float * h_niza1 = (float*)calloc(ARRAY_SIZE, sizeof(float));
	float * h_niza2 = (float*)calloc(ARRAY_SIZE, sizeof(float));
	float * h_nizaRez = (float*)calloc(ARRAY_SIZE, sizeof(float));

	srand((unsigned int)time(NULL));

	for (int i = 0; i<ARRAY_SIZE; i++)
	{
		h_niza1[i] = (float)(i) + 0.5f;
		h_niza2[i] = h_niza1[i];
	}

	float * d_niza1, * d_niza2, * d_nizaRez;
	cudaMalloc(&d_niza1, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_niza2, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_nizaRez, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_niza1, h_niza1, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_niza2, h_niza2, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMemset(d_nizaRez, 0, sizeof(float)*ARRAY_SIZE);
	cudaEventRecord(start);
	vectorSum<<<blocks, threads>>>(d_niza1, d_niza2, d_nizaRez);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Vector sum naive:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_nizaRez, d_nizaRez, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaVectorSum(h_niza1, h_niza2, h_nizaRez, ARRAY_SIZE);
	
	cudaMemset(d_nizaRez, 0, sizeof(float)*ARRAY_SIZE);
	cudaEventRecord(start);
	vectorSumSharedMem<<<blocks, threads, threads*2*sizeof(float)>>>(d_niza1, d_niza2, d_nizaRez);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Vector sum shared:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_nizaRez, d_nizaRez, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaVectorSum(h_niza1, h_niza2, h_nizaRez, ARRAY_SIZE);

	cudaMemset(d_nizaRez, 0, sizeof(float)*ARRAY_SIZE);
	cudaEventRecord(start);
	vectorRazlikaSosedi<<<blocks, threads>>>(d_niza1, d_nizaRez, ARRAY_SIZE);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Razlika sosedi naive:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_nizaRez, d_nizaRez, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaRazlikaSosedi(h_niza1, h_nizaRez, ARRAY_SIZE);

	cudaMemset(d_nizaRez, 0, sizeof(float)*ARRAY_SIZE);
	cudaEventRecord(start);
	vectorRazlikaSosediSharedMem<<<blocks, threads, (threads)*sizeof(float)>>>(d_niza1, d_nizaRez, ARRAY_SIZE);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Razlika sosedi shared:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_nizaRez, d_nizaRez, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaRazlikaSosedi(h_niza1, h_nizaRez, ARRAY_SIZE);

	cudaMemset(d_nizaRez, 0, sizeof(float)*ARRAY_SIZE);
	int blocksV4f2 = blocks/threads + (blocks%threads != 0);
	cudaEventRecord(start);
	vectorRazlikaSosediSharedMemv4Faza1<<<blocks, threads, (threads+1)*sizeof(float)>>>(d_niza1, d_nizaRez, ARRAY_SIZE);
	vectorRazlikaSosediSharedMemv4Faza2<<<blocksV4f2, threads>>>(d_niza1, d_nizaRez, ARRAY_SIZE);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Razlika sosedi shared 2 fazi:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_nizaRez, d_nizaRez, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaRazlikaSosedi(h_niza1, h_nizaRez, ARRAY_SIZE);

	cudaMemset(d_nizaRez, 0, sizeof(float)*ARRAY_SIZE);
	int blocksV5f1 = ARRAY_SIZE / (threads*2) + (ARRAY_SIZE%(threads*2) != 0);
	int blocksV5f2 = blocksV5f1 / threads + (blocksV5f1 % threads != 0);
	cudaEventRecord(start);
	vectorRazlikaSosediSharedMemv5Faza1<<<blocksV5f1, threads, (2*threads+1)*sizeof(float)>>>(d_niza1, d_nizaRez, ARRAY_SIZE);
	vectorRazlikaSosediSharedMemv5Faza2<<<blocksV5f2, threads>>>(d_niza1, d_nizaRez, ARRAY_SIZE);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Razlika sosedi shared duplo 2 fazi:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_nizaRez, d_nizaRez, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaRazlikaSosedi(h_niza1, h_nizaRez, ARRAY_SIZE);

	float * d_max, h_max[1];
	cudaMalloc(&d_max, sizeof(float));
	h_max[0] = 0;
	cudaMemcpy(d_max, h_max, sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	vectorMax<<<blocks, threads>>>(d_niza1, d_max, ARRAY_SIZE);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Vector max:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaCpuMax(h_niza1, h_max[0], ARRAY_SIZE);

	h_max[0] = 0;
	cudaMemcpy(d_max, h_max, sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	vectorMaxv2<<<blocks, threads>>>(d_niza1, d_max, ARRAY_SIZE);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	printf("Vector max shared mem:\tblokovi=%d niski=%d elem=%d, vreme=%f\n", blocks, threads, ARRAY_SIZE, vreme);
	cudaMemcpy(h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
	proveriGreskaCpuMax(h_niza1, h_max[0], ARRAY_SIZE);

	/*long cpustart = clock();
	for (int i = 0; i < ARRAY_SIZE-1; i++) {
		h_nizaRez[i] = h_niza1[i+1] - h_niza1[i];
	}
	long cpuend = clock();
	printf("CPU sequential razlika sosedi vreme=%ld\n", cpuend - cpustart);*/
	

	cudaFree(d_niza1);
	cudaFree(d_niza2);
	cudaFree(d_nizaRez);

	free(h_niza1);
	free(h_niza2);
	free(h_nizaRez);
}
