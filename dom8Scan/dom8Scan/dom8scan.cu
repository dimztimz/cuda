#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>

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
__global__ void inclusiveScanHillisSteele2OutBlock(const T * out, T * outerBlockScan, int len)
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
__global__ void inclusiveScanHillisSteele3Merge(T * out, const T * outerBlockScan, int len)
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

template <int THREADSBLOCK, typename T>
__global__ void exclusiveScanBlelloch1InBlock(const T * in, T * out, int len)
{
	__shared__ T cache[THREADSBLOCK];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	T vrednost = 0;
	if (idx < len) {
		vrednost = in[idx];
	}
	cache[tid] = vrednost;
	__syncthreads();

	//upsweep
#pragma unroll
	for(int pomestuvanje = 1; pomestuvanje < THREADSBLOCK; pomestuvanje <<= 1) {
		int iDesno = pomestuvanje * (tid * 2 + 2) - 1;
		if (iDesno < THREADSBLOCK) {
			cache[iDesno] += cache[iDesno-pomestuvanje];
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		cache[THREADSBLOCK-1] = 0;
	}

	//downsweep
#pragma unroll
	for(int pomestuvanje = THREADSBLOCK >> 1; pomestuvanje != 0; pomestuvanje >>= 1) {
		int iDesno = pomestuvanje * (tid * 2 + 2) - 1;
		if (iDesno < THREADSBLOCK) {
			T levo = cache[iDesno - pomestuvanje];
			T desno = cache[iDesno];
			cache[iDesno] = levo + desno;
			cache[iDesno - pomestuvanje] = desno;
		}
		__syncthreads();
	}

	if (idx < len) {
		out[idx] = cache[tid];
	}
}

template <int THREADSBLOCK, typename T>
__global__ void exclusiveScanBlelloch2OutBlock(const T * in, const T * out, T * buff, int len)
{
	__shared__ T cache[THREADSBLOCK];
	int tid = threadIdx.x;
	int idx = blockDim.x * threadIdx.x + blockDim.x - 1;
	

	T vrednost = 0;
	if (idx < len) {
		vrednost = out[idx] + in[idx];
	}
	cache[tid] = vrednost;
	__syncthreads();

	//upsweep
#pragma unroll
	for(int pomestuvanje = 1; pomestuvanje < THREADSBLOCK; pomestuvanje <<= 1) {
		int iDesno = pomestuvanje * (tid * 2 + 2) - 1;
		if (iDesno < THREADSBLOCK) {
			cache[iDesno] += cache[iDesno-pomestuvanje];
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0) {
		cache[THREADSBLOCK-1] = 0;
	}

	//downsweep
#pragma unroll
	for(int pomestuvanje = THREADSBLOCK >> 1; pomestuvanje != 0; pomestuvanje >>= 1) {
		int iDesno = pomestuvanje * (tid * 2 + 2) - 1;
		if (iDesno < THREADSBLOCK) {
			T levo = cache[iDesno - pomestuvanje];
			T desno = cache[iDesno];
			cache[iDesno] = levo + desno;
			cache[iDesno - pomestuvanje] = desno;
		}
		__syncthreads();
	}

	buff[tid] = cache[tid];
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

void proveriTocnostInclusive() {
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

void proveriTocnostExclusive() {
	float * h_in, * h_scan, * h_refScan;
	int maxN = 5000;

	h_in = new float[maxN];
	h_scan = new float[maxN];
	h_refScan = new float[maxN];

	srand(2008);
	for(int i = 0; i<maxN; i++) {
		h_in[i] = (rand() % 4000) + 0.5f;
	}
	cpuExclusiveScan(h_in, h_refScan, maxN);

	float * d_in, * d_scan, *d_buffOutBLockScan;
	cudaMalloc(&d_in, maxN * sizeof(float));
	cudaMalloc(&d_scan, maxN * sizeof(float));
	cudaMalloc(&d_buffOutBLockScan, 256 * sizeof(float));
	cudaMemcpy(d_in, h_in, maxN * sizeof(float), cudaMemcpyHostToDevice);

	exclusiveScanBlelloch1InBlock<256, float><<<(maxN+255)/256, 256>>>(d_in, d_scan, maxN);
	exclusiveScanBlelloch2OutBlock<256, float><<<1, 256>>>(d_in, d_scan, d_buffOutBLockScan, maxN);
	inclusiveScanHillisSteele3Merge<256, float><<<(maxN+255)/256, 256>>>(d_scan, d_buffOutBLockScan, maxN);
	cudaMemcpy(h_scan, d_scan, maxN * sizeof(float), cudaMemcpyDeviceToHost);

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

template <int THREADSBLOCK>
void benchmarkInclusiveHillisSteele(FILE * outFile)
{
	float * d_array, * d_scan, * d_buff;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	fprintf(outFile, "GPU hillis steele scan\n");
	fprintf(outFile, "N;tile width;vreme\n");
	for (int N = 32; N <= THREADSBLOCK*THREADSBLOCK; N+=32) {
		cudaMalloc(&d_array, N*sizeof(float));
		cudaMalloc(&d_scan, N*sizeof(float));
		cudaMalloc(&d_buff, THREADSBLOCK*sizeof(float));
		cudaEventRecord(start);
		inclusiveScanHillisSteele1InBlock<THREADSBLOCK, float>
			<<<(N+THREADSBLOCK-1)/THREADSBLOCK, THREADSBLOCK>>>(d_array, d_scan, N);
		if (N > THREADSBLOCK) {
			inclusiveScanHillisSteele2OutBlock<THREADSBLOCK, float><<<1, THREADSBLOCK>>>
				(d_scan, d_buff, N);
			inclusiveScanHillisSteele3Merge<THREADSBLOCK, float>
				<<<(N+THREADSBLOCK-1)/THREADSBLOCK, THREADSBLOCK>>>(d_scan, d_buff, N);
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_array);
		cudaFree(d_scan);
		cudaFree(d_buff);
		fprintf(outFile, "%d;%d;%f\n", N, THREADSBLOCK, vreme);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template <int THREADSBLOCK>
void benchmarkExclusiveBlelloch(FILE * outFile)
{
	float * d_array, * d_scan, * d_buff;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	fprintf(outFile, "GPU blelloch scan\n");
	fprintf(outFile, "N;tile width;vreme\n");
	for (int N = 32; N <= THREADSBLOCK*THREADSBLOCK; N+=32) {
		cudaMalloc(&d_array, N*sizeof(float));
		cudaMalloc(&d_scan, N*sizeof(float));
		cudaMalloc(&d_buff, THREADSBLOCK*sizeof(float));
		cudaEventRecord(start);
		exclusiveScanBlelloch1InBlock<THREADSBLOCK, float>
			<<<(N+THREADSBLOCK-1)/THREADSBLOCK, THREADSBLOCK>>>(d_array, d_scan, N);
		if (N > THREADSBLOCK) {
			exclusiveScanBlelloch2OutBlock<THREADSBLOCK, float><<<1, THREADSBLOCK>>>
				(d_array, d_scan, d_buff, N);
			inclusiveScanHillisSteele3Merge<THREADSBLOCK, float>
				<<<(N+THREADSBLOCK-1)/THREADSBLOCK, THREADSBLOCK>>>(d_scan, d_buff, N);
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_array);
		cudaFree(d_scan);
		cudaFree(d_buff);
		fprintf(outFile, "%d;%d;%f\n", N, THREADSBLOCK, vreme);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void benchmark(const char * filenamePrefix)
{
	float * h_array, * h_scan;

	FILE * cpuInclusiveOut = stdout, * cpuExclusiveOut = stdout;
	FILE * gpuInclusiveOut = stdout, * gpuExclusiveOut = stdout;
	if (filenamePrefix != NULL) {
		char filename[256];
		strncpy(filename, filenamePrefix, 256);
		strncat(filename, "_cpuIncusive.csv", 256);
		cpuInclusiveOut = fopen(filename, "w");
		strncpy(filename, filenamePrefix, 256);
		strncat(filename, "_cpuExclusive.csv", 256);
		cpuExclusiveOut = fopen(filename, "w");
		strncpy(filename, filenamePrefix, 256);
		strncat(filename, "_gpuInclusive.csv", 256);
		gpuInclusiveOut = fopen(filename, "w");
		strncpy(filename, filenamePrefix, 256);
		strncat(filename, "_gpuExclusive.csv", 256);
		gpuExclusiveOut = fopen(filename, "w");
	}

	fprintf(cpuInclusiveOut, "CPU inclusive scan\n");
	fprintf(cpuInclusiveOut, "N;vreme\n");
	for (int N = 32; N < 512*512; N+=32) {
		h_array = (float *)calloc(N, sizeof(float));
		h_scan = (float *)calloc(N, sizeof(float));

		double vreme = omp_get_wtime();
		cpuInclusiveScan(h_array, h_scan, N);
		vreme = omp_get_wtime() - vreme;
		fprintf(cpuInclusiveOut, "%d;%lf\n", N, vreme*1000.0);

		free(h_array);
		free(h_scan);
	}

	fprintf(cpuExclusiveOut ,"CPU exclusive scan\n");
	fprintf(cpuExclusiveOut, "N;vreme\n");
	for (int N = 32; N < 512*512; N+=32) {
		h_array = (float *)calloc(N, sizeof(float));
		h_scan = (float *)calloc(N, sizeof(float));

		double vreme = omp_get_wtime();
		cpuExclusiveScan(h_array, h_scan, N);
		vreme = omp_get_wtime() - vreme;
		fprintf(cpuExclusiveOut, "%d;%lf\n", N, vreme*1000.0);

		free(h_array);
		free(h_scan);
	}
	
	benchmarkInclusiveHillisSteele<32>(gpuInclusiveOut);
	benchmarkInclusiveHillisSteele<64>(gpuInclusiveOut);
	benchmarkInclusiveHillisSteele<128>(gpuInclusiveOut);
	benchmarkInclusiveHillisSteele<256>(gpuInclusiveOut);
	benchmarkInclusiveHillisSteele<512>(gpuInclusiveOut);
	benchmarkInclusiveHillisSteele<1024>(gpuInclusiveOut);

	benchmarkExclusiveBlelloch<32>(gpuExclusiveOut);
	benchmarkExclusiveBlelloch<64>(gpuExclusiveOut);
	benchmarkExclusiveBlelloch<128>(gpuExclusiveOut);
	benchmarkExclusiveBlelloch<256>(gpuExclusiveOut);
	benchmarkExclusiveBlelloch<512>(gpuExclusiveOut);
	benchmarkExclusiveBlelloch<1024>(gpuExclusiveOut);

	fclose(cpuInclusiveOut);
	fclose(cpuExclusiveOut);
	fclose(gpuInclusiveOut);
	fclose(gpuExclusiveOut);
}

int main(int argc, char * argv[])
{
	if (argc > 1) {
		benchmark(argv[1]);
	} else {
		benchmark((const char *)NULL);
	}
}