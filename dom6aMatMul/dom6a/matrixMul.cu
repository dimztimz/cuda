#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>

const int TILE_WIDTH = 32;

void proveriOdGreski();
void benchmark();

__global__  void matrixMulNaiveKernel(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	int kol = blockDim.x * blockIdx.x + threadIdx.x;
	int red = blockDim.y * blockIdx.y + threadIdx.y;

	if (kol >= kolB || red >= redA) {
		return;
	}

	float s = 0.0f;
	for(int k = 0; k < kolARedB; k++) {
		s+= A[red*kolARedB + k] * B[k*kolB + kol];
	}
	C[red*kolB + kol] = s;
}

__global__ void matixMulTilesKernel(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	__shared__ float Acache[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bcache[TILE_WIDTH][TILE_WIDTH];

	int CkolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int CredIdx = blockDim.y * blockIdx.y + threadIdx.y;
	float rez = 0.0f;

	for (int p = 0; p<((kolARedB+TILE_WIDTH-1)/TILE_WIDTH); p++) {
		int AredIdx = CredIdx;
		int AkolIdx = p*TILE_WIDTH + threadIdx.x;
		int Aidx = AredIdx * kolARedB + AkolIdx;
		//int Aidx = CredIdx * kolARedB + p*TILE_WIDTH + threadIdx.x;
		
		int BredIdx = p*TILE_WIDTH + threadIdx.y;
		int BkolIdx = CkolIdx;
		int Bidx = BredIdx * kolB + BkolIdx;
		//int Bidx = (p*TILE_WIDTH + threadIdx.y)*kolB + CkolIdx;

		
		float Aval = 0.0f;
		float Bval = 0.0f;
		if (AredIdx < redA && AkolIdx < kolARedB) {
			Aval = A[Aidx];
		}
		if (BredIdx < kolARedB && BkolIdx < kolB) {
			Bval = B[Bidx];
		}
		Acache[threadIdx.y][threadIdx.x] = Aval;
		Bcache[threadIdx.y][threadIdx.x] = Bval;
		__syncthreads();
		for(int k = 0; k<TILE_WIDTH; k++) {
			rez += Acache[threadIdx.y][k] * Bcache[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (CkolIdx < kolB && CredIdx < redA) {
		C[CredIdx*kolB + CkolIdx] = rez;
	}	
}

__global__ void matixMulTilesPoramnetoKernel(const float * A, const float * B, float * C, int kolARedB)
{
	__shared__ float Acache[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bcache[TILE_WIDTH][TILE_WIDTH];

	int CkolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int CredIdx = blockDim.y * blockIdx.y + threadIdx.y;

	int kolB = gridDim.x * blockDim.x;

	float rez = 0.0f;

	for (int p = 0; p<kolARedB/TILE_WIDTH; p++) {
		int AredIdx = CredIdx;
		int AkolIdx = p*TILE_WIDTH + threadIdx.x;
		int Aidx = AredIdx * kolARedB + AkolIdx;
		//int Aidx = CredIdx * kolARedB + p*TILE_WIDTH + threadIdx.x;
		
		int BredIdx = p*TILE_WIDTH + threadIdx.y;
		int BkolIdx = CkolIdx;
		int Bidx = BredIdx * kolB + BkolIdx;
		//int Bidx = (p*TILE_WIDTH + threadIdx.y)*kolB + CkolIdx;

		Acache[threadIdx.y][threadIdx.x] = A[Aidx];
		Bcache[threadIdx.y][threadIdx.x] = B[Bidx];
		__syncthreads();
		for(int k = 0; k<TILE_WIDTH; k++) {
			rez += Acache[threadIdx.y][k] * Bcache[k][threadIdx.x];
		}
		__syncthreads();
	}
	C[CredIdx*kolB + CkolIdx] = rez;

}

void matMulCpuNaive(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	for (int i = 0; i < redA; i++) {
		for (int j = 0; j< kolB; j++) {
			float s = 0.0f;
			const float * Apok = A + i*kolARedB;
			const float * Bpok = B + j;
			for (int k = 0; k<kolARedB; k++) {
				s += *Apok * *Bpok;
				Apok++;
				Bpok += kolB;
			}
			C[i * kolB + j] = s;
		}
	}
}

bool proveriGreska(const float * referenceMat, const float * presmetanaMat, int len)
{
	bool greska = false;
	for (int i = 0; i<len; i++) {
		if (referenceMat[i] != presmetanaMat[i]) {
			greska = true;
			//printf("Greska vo index %d, %f != %f\n", i, referenceMat[i], presmetanaMat[i]);
		}
	}
	return greska;
}

int main(int argc, char * argv[])
{
	//proveriOdGreski();
	benchmark();
	return 0;
}

void benchmark()
{
	float * h_Amat, * h_Bmat, * h_Cmat;
	float * d_Amat, * d_Bmat, * d_Cmat;

	/*printf("CPU mat mul naive\n");
	printf("N;vreme\n");
	for (int N = 4; N < 750; N+=1) {
		h_Amat = (float *)calloc(N*N, sizeof(float));
		h_Bmat = (float *)calloc(N*N, sizeof(float));
		h_Cmat = (float *)calloc(N*N, sizeof(float));

		double vreme = omp_get_wtime();
		matMulCpuNaive(h_Amat, h_Bmat, h_Cmat, N, N, N);
		vreme = omp_get_wtime() - vreme;
		printf("%d;%lf\n", N, vreme*1000.0);

		free(h_Amat);
		free(h_Bmat);
		free(h_Cmat);
	}*/

	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);


	printf("GPU mat mul naive\n");
	printf("N;tile width;vreme\n");
	for (int N = 4; N < 750; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matrixMulNaiveKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}

	printf("GPU mat mul tiled\n");
	printf("N;tile width;vreme\n");
	for (int N = 4; N < 750; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matixMulTilesKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}

	printf("GPU mat mul tiled poramneto\n");
	printf("N;tile width;vreme\n");
	for (int N = TILE_WIDTH; N < 750; N+=TILE_WIDTH) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matixMulTilesKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}


}

void proveriOdGreski()
{
	float * h_Amat, * h_Bmat, * h_Cmat, * h_CmatRef;
	float * d_Amat, * d_Bmat, * d_Cmat;
	int maxN = 600;
	int maxLen = maxN * maxN;
	h_Amat = (float*)calloc(maxLen, sizeof(float));
	h_Bmat = (float*)calloc(maxLen, sizeof(float));
	h_Cmat = (float*)calloc(maxLen, sizeof(float));
	h_CmatRef = (float*)calloc(maxLen, sizeof(float));

	cudaMalloc(&d_Amat, maxLen * sizeof(float));
	cudaMalloc(&d_Bmat, maxLen * sizeof(float));
	cudaMalloc(&d_Cmat, maxLen * sizeof(float));

	srand(5);
	for (int i = 0; i<maxLen; i++) {
		h_Amat[i] = h_Bmat[i] = (float)(rand() % 100);
	}
	cudaMemcpy(d_Amat, h_Amat, maxLen * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bmat, h_Bmat, maxLen * sizeof(float), cudaMemcpyHostToDevice);

	for (int N = 1; N<maxN; N++) {
		matMulCpuNaive(h_Amat, h_Bmat, h_CmatRef, N, N, N);

		dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);

		matrixMulNaiveKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("naive N=%d greska\n", N);
		} else {
			printf("naive N=%d tocno\n", N);
		}

		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matixMulTilesKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("tiled N=%d greska\n", N);
		} else {
			printf("tiled N=%d tocno\n", N);
		}

		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matixMulTilesPoramnetoKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("tiled poramneto N=%d greska\n", N);
		} else {
			printf("tiled poramneto N=%d tocno\n", N);
		}
	}

	cudaFree(d_Amat);
	cudaFree(d_Bmat);
	cudaFree(d_Cmat);

	free(h_Amat);
	free(h_Bmat);
	free(h_Cmat);
	free(h_CmatRef);
}