#include <iostream>
#include <fstream>

#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define repeat2(x) x;x;
#define repeat4(x) repeat2(repeat2(x))
#define repeat8(x) repeat2(repeat4(x))
#define repeat16(x) repeat4(repeat4(x))
#define repeat32(x) repeat2(repeat16(x))
#define repeat64(x) repeat2(repeat32(x))
#define repeat128(x) repeat2(repeat64(x))
#define repeat256(x) repeat2(repeat128(x))
#define repeatN(x,n) {if (n & 1) x; \
if (n & 2) { repeat2(x) } \
if (n & 4) { repeat4(x) } \
if (n & 8) { repeat8(x) } \
if (n & 16) { repeat16(x) } \
if (n & 32) { repeat32(x) } \
if (n & 64) { repeat64(x) } \
if (n & 128) { repeat128(x) } \
}

using namespace std;

__device__ clock_t d_clocks[1];

__global__ void kernel(int ** a, int n)
{
	int **j = a;
	int povt = (2*n) >> 8 /* 2*n/256*/;
	int osatanti = (2*n) & 255 /*(2*n)%256*/;
	clock_t c1 = clock();
	for (int i = 0; i < povt; i++) {
		repeat256(j = (int**)*j)
	}
	repeatN(j = (int**)*j, osatanti)
		clock_t c2 = clock();
	//printf("%d,%ld,%ld,%ld\n", n, c1, c2, c2 - c1);
	d_clocks[0] = c2-c1;
}


int main(int argc, char* argv[])
{
	const int goleminaKes = 16384;
	const int goleminaLinija = 128;
	const int brojNaLinii = goleminaKes / goleminaLinija; /*128*/
	const int elementiVoLinija = goleminaLinija / sizeof(int*); /*32*/
	int stride = elementiVoLinija/2; //=16. mora da bide pomalo na elementi vo linija, po moznost da bide delitel

	cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferShared);

	const int min_array_size = brojNaLinii*elementiVoLinija;
	const int max_array_size = 2 * min_array_size;
	for (int array_size = min_array_size; array_size <= max_array_size; array_size++) {
		long long int totalClocks = 0;
		for (int j = 0; j < 100; j++) {

			int ** host_array = new int*[array_size];
			int ** device_array;
			cudaMalloc(&device_array, sizeof(int*)*array_size);
			for (int i = 0; i < array_size; i++) {
				int t = i + stride;
				if (t >= array_size) {
					t %= stride;
				}
				host_array[i] = (int*)(device_array + t);
			}
			cudaMemcpy(device_array, host_array, sizeof(int*)*array_size, cudaMemcpyHostToDevice);
			//cout << cudaGetErrorString(cudaGetLastError()) << endl;
			kernel <<< 1, 1 >>>(device_array, array_size);
			//cout << cudaGetErrorString(cudaGetLastError()) << endl;
			cudaDeviceSynchronize();
			clock_t clocks;
			cudaMemcpyFromSymbol(&clocks, d_clocks, sizeof(d_clocks));
			totalClocks += clocks;
			//cout << array_size << ",pov=" << j /*<< ",c1=" << clocks[0] << ",c2=" << clocks[1]*/ << ",del_c=" << clocks << endl;
			delete[] host_array;
			cudaFree(device_array);
		}
		double avg_clocks_program = totalClocks / 100.0;
		double avg_clocks_per_load = avg_clocks_program / (2.0*array_size); //prosecna latentnost po load
		cout << array_size << ',' << avg_clocks_per_load << endl;
	}

	

	//system("pause");
	return 0;
}

