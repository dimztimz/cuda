#include <cstdio>
#include <cstdlib>
#include <cfloat>

#include <cuda_runtime.h>
#include <omp.h>

template <int THREADSBLOCK, typename T>
__global__ void inclusiveScanHillisSteele1InBlock(const T * in, T * out, int len)
{
	__shared__ T cache[THREADSBLOCK*2];

	int shIdx = threadIdx.x;
	cache[shIdx] = 0;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	T val = 0;
	if (idx < len) {
		val = in[idx];
	}
	shIdx += blockDim.x;
	cache[shIdx] = val;

#pragma unroll
	for (int pomestuvanje = 1; pomestuvanje < THREADSBLOCK; pomestuvanje <<= 1) {
		__syncthreads();
		val += cache[shIdx - pomestuvanje];
		__syncthreads();
		cache[shIdx] = val;
	}
	if (idx < len) {
		out[idx] = val;
	}
}

template <int THREADSBLOCK, typename T>
__global__ void inclusiveScanHillisSteele2OutBlock(T * out, T * outerBlockScan, int len)
{
	__shared__ T cache[THREADSBLOCK*2];

	int shIdx = threadIdx.x;
	cache[shIdx] = 0;

	int idx = blockDim.x * threadIdx.x + blockDim.x - 1;
	T val = 0;
	if (idx < len) {
		val = out[idx];
	}
	shIdx += blockDim.x;
	cache[shIdx] = val;

	T scanVal = val;
#pragma unroll
	for (int pomestuvanje = 1; pomestuvanje < THREADSBLOCK; pomestuvanje <<= 1) {
		__syncthreads();
		scanVal += cache[shIdx - pomestuvanje];
		__syncthreads();
		cache[shIdx] = scanVal;
	}
	outerBlockScan[threadIdx.x] = scanVal - val;
}

template <int THREADSBLOCK, typename T>
__global__ void inclusiveScanHillisSteele3Merge(T * out, T * outerBlockScan, int len)
{
	__shared__ T toAdd;
	if (threadIdx.x == 0) {
		toAdd = outerBlockScan[blockIdx.x];
	}
	__syncthreads();
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < len) {
		out[idx] += toAdd;
	}
}


void cpuInclusiveScan(float * in, float * out, int len)
{
	float suma = 0;
	for (int i = 0; i < len; i++) {
		suma += in[i];
		out[i] = suma;
	}
}

void cpuExclusiveScan(float * in, float * out, int len)
{
	float suma = 0;
	for (int i = 0; i < len; i++) {
		out[i] = suma;
		suma += in[i];
	}
}

void proveriTocnost() {
	float * h_in, * h_scan, * h_refScan;
	int maxN = 256*10+10;

	h_in = new float[maxN];
	h_scan = new float[maxN];
	h_refScan = new float[maxN];

	srand(2008);
	for(int i = 0; i<maxN; i++) {
		h_in[i] = (rand() % 4000) + 0.5f;
	}
	cpuInclusiveScan(h_in, h_refScan, maxN);

	float * d_in, * d_scan, *d_buffOutBLockScan;
	cudaMalloc(&d_in, maxN * sizeof(float));
	cudaMalloc(&d_scan, maxN * sizeof(float));
	cudaMalloc(&d_buffOutBLockScan, 256 * sizeof(float));
	cudaMemcpy(d_in, h_in, maxN * sizeof(float), cudaMemcpyHostToDevice);

	inclusiveScanHillisSteele1InBlock<256, float><<<(maxN+255)/256, 256>>>(d_in, d_scan, maxN);
	inclusiveScanHillisSteele2OutBlock<256, float><<<1, 256>>>(d_scan, d_buffOutBLockScan, maxN);
	inclusiveScanHillisSteele3Merge<256, float><<<(maxN+255)/256, 256>>>(d_scan, d_buffOutBLockScan, maxN);
	cudaMemcpy(h_scan, d_scan, maxN * sizeof(float), cudaMemcpyDeviceToHost);
	/*for(int i = 0; i<maxN; i++) {
		printf("index %d: %f\n", i, h_scan[i]);
	}*/

	for(int i = 0; i<maxN; i++) {
		if (h_refScan[i] != h_scan[i]) {
			printf("Greska vo index %d: %f != %f\n", i, h_refScan[i], h_scan[i]);
			
		}
	}

	cudaFree(d_in);
	cudaFree(d_scan);
	cudaFree(d_buffOutBLockScan);

	delete[] h_in;
	delete[] h_scan;
	delete[] h_refScan;
}

int main()
{
	proveriTocnost();
}