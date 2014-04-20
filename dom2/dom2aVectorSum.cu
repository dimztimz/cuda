#include <cstdlib>
#include <cstdio>


using namespace std;

__global__ void vectorSum(float * d_niza1, float * d_niza2, float * d_nizaSuma)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	d_nizaSuma[i] = d_niza1[i] + d_niza2[i];
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

	float * h_niza1 = new float[ARRAY_SIZE];
	float * h_niza2 = new float[ARRAY_SIZE];
	float * h_nizaSum = new float[ARRAY_SIZE];

	srand(5);

	for (int i = 0; i<ARRAY_SIZE; i++)
	{
		h_niza1[i] = rand() % 4000000;
		h_niza2[i] = rand() % 4000000;
	}

	float * d_niza1, * d_niza2, * d_nizaSum;
	cudaMalloc(&d_niza1, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_niza2, ARRAY_SIZE * sizeof(float));
	cudaMalloc(&d_nizaSum, ARRAY_SIZE * sizeof(float));

	cudaMemcpy(d_niza1, h_niza1, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_niza2, h_niza2, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);


	//cudaEvent_t start1_32, end1_32, start32_1, end32_1, start2_16, end2_16, start2_512, end2_512;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);
	vectorSum<<<blocks, threads>>>(d_niza1, d_niza2, d_nizaSum);
	cudaEventRecord(end);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(&vreme, start, end);


	printf("blokovi=%d niski=%d, vreme=%f\n", blocks, threads, vreme);

	cudaMemcpy(h_nizaSum, d_nizaSum, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i<ARRAY_SIZE; i++)
	{
		//cout << setprecision(0) << h_niza1[i] << " + " << h_niza2[i] << " = " << h_nizaSum[i] << endl;
		printf("%.0f + %.0f = %.0f\n", h_niza1[i], h_niza2[i], h_nizaSum[i]);
	}*/

	cudaFree(d_niza1);
	cudaFree(d_niza2);
	cudaFree(d_nizaSum);

	delete[] h_niza1;
	delete[] h_niza2;
	delete[] h_nizaSum;

}
