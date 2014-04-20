#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

template <int THREADS, int BINS>
__global__ void perTheadHistogram(int * data, int len, int * outHist)
{
	__shared__ int hist[BINS][THREADS];

	int gridSize = blockDim.x * gridDim.x;
	

	for (int i = 0; i<BINS; i++) {
		hist[i][threadIdx.x] = 0;
	}

	for(int i = threadIdx.x; i<len; i+=gridSize) {
		int bin = data[i] % BINS;
		hist[bin][threadIdx.x]++;
	}

	/*
	//v1 najbavna
	for (int i = 0; i<BINS; i++) {
		atomicAdd(&outHist[i], hist[i][threadIdx.x]);
	}*/

	/*
	//v2 izbegnuvame atomicno zapisuvanje, imame konflikti na bankti
	for (int binIdx = threadIdx.x; binIdx < BINS; binIdx += blockDim.x) {
		for(int thIdx = 0; thIdx < blockDim.x; thIdx++) {
			outHist[binIdx] += hist[binIdx][thIdx];
		}
	}*/


	
	//v3 izbegnuvame atomicno zapisuvanje, iscisteni konflikti od bankti i pri citanje i pri zapisuvanje
	for (int binIdx = threadIdx.x; binIdx < BINS; binIdx += blockDim.x) {
		for(int thIdx = 0; thIdx < blockDim.x-1; thIdx++) {
			int tmpThIdx = (thIdx + threadIdx.x + 1) % blockDim.x;
			hist[binIdx][threadIdx.x] += hist[binIdx][tmpThIdx]; //moze slobodbo sumite da se cuvat vo lokalna, dopolnitelna optimizacija
		}
		outHist[binIdx] = hist[binIdx][threadIdx.x];
	}
	

	/*
	//v4 redukcija
	for (int skok = blockDim.x >> 1; skok != 0; skok >>= 1) {
		if (threadIdx.x < skok) {
			for (int bin = 0; bin<BINS; bin++) {
				hist[bin][threadIdx.x] += hist[bin][threadIdx.x + skok];
			}
		}
	}
	if (threadIdx.x == 0) {
		for (int bin = 0; bin<BINS; bin++) {
				outHist[bin] = hist[bin][0];
			}
	}*/


}

int main()
{
	int * h_data, * h_hist;
	int * d_data, * d_hist;

	h_data = (int*)calloc(1000000, sizeof(int));
	h_hist = (int*)calloc(100, sizeof(int));

	for(int i = 0; i<1000000; i++) {
		h_data[i] = i+1;
	}

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMalloc(&d_data, 1000000*sizeof(int));
	cudaMalloc(&d_hist, 100*sizeof(int));
	cudaMemcpy(d_data, h_data, 1000000*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(d_hist, 0, 100*sizeof(int));

	cudaEventRecord(start);
	perTheadHistogram<96, 100><<<1, 96>>>(d_data, 1000000, d_hist);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);

	cudaMemcpy(h_hist, d_hist, 100*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i<100; i++) {
		printf("%d ", h_hist[i]);
	}
	puts("");

	printf("vreme=%f\n", vreme);
	return 0;
}