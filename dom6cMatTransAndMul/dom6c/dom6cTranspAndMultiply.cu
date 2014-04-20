#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>

const int TILE_WIDTH1 = 12;

__global__ void transposeFullParralel2DBlocking(float * inMatrix, float * outMatrix, int redici, int koloni)
{
	int r = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	if (r >= redici || k >= koloni) {
		return;
	}
	outMatrix[k*redici + r] = inMatrix[r*koloni + k];
}

__global__ void matrixMulBtrNaiveKernel(const float * A, const float * Btr, float * C, int redA, int kolARedB, int kolB)
{
	int kol = blockDim.x * blockIdx.x + threadIdx.x;
	int red = blockDim.y * blockIdx.y + threadIdx.y;

	if (kol >= kolB || red >= redA) {
		return;
	}

	float s = 0.0f;
	for(int k = 0; k < kolARedB; k++) {
		s+= A[red*kolARedB + k] * Btr[kol*kolARedB + k];
	}
	C[red*kolB + kol] = s;
}

/**
* na vlez se prakja matrica A normalna i matrica B transponirana
* rezultatot e C e mnozenje na A * B kako B da ne e transponirana
* argumentite za golemina na matricite se prakjaat kako B da NE e transponirana
* vnatre, posledovatelni niski citaat posledovatelni elementi od A koga prefrlaat od globalna vo spodelena
* dodeka pak posledovatelni niski citaat po koloni za B koga prefrlaat vo shared, a vo shared zapisuvat po redici
* sto znaci blokceto od B se odtransponira pred da se mnozi.
* FOR-ot sto pravi skalaren prozivod e kako kaj obicno mnozenje odnosno za A izminuva redici a za B koloni.
*/
template <int TILE_WIDTH>
__global__ void matrixMulBtrTilesKernelV1(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
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
		
		int BtrRedIdx = CkolIdx;
		int BtrKolIdx = p*TILE_WIDTH + threadIdx.y;
		int Bidx = BtrRedIdx * kolARedB + BtrKolIdx;

		
		float Aval = 0.0f;
		float Bval = 0.0f;
		if (AredIdx < redA && AkolIdx < kolARedB) {
			Aval = A[Aidx];
		}
		if (BtrRedIdx < kolB && BtrKolIdx < kolARedB) {
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

/**
* na vlez se prakja matrica A normalna i matrica B transponirana
* rezultatot e C e mnozenje na A * B kako B da ne e transponirana
* argumentite za golemina na matricite se prakjaat kako B da NE e transponirana
* vnatre, posledovatelni niski citaat posledovatelni elementi od A koga prefrlaat od globalna vo spodelena
* dodeka pak posledovatelni niski citaat po koloni za B koga prefrlaat vo shared, a vo shared zapisuvat po koloni
* sto znaci blokceto od B ostanuva transponirano pred da se mnozi.
* FOR-ot sto pravi skalaren prozivod e kako za B transponirana odnosno i na dvete matrici izminuva po redici.
*/
template <int TILE_WIDTH>
__global__ void matrixMulBtrTilesKernelV2(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
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
		
		int BtrRedIdx = CkolIdx;
		int BtrKolIdx = p*TILE_WIDTH + threadIdx.y;
		int Bidx = BtrRedIdx * kolARedB + BtrKolIdx;

		
		float Aval = 0.0f;
		float Bval = 0.0f;
		if (AredIdx < redA && AkolIdx < kolARedB) {
			Aval = A[Aidx];
		}
		if (BtrRedIdx < kolB && BtrKolIdx < kolARedB) {
			Bval = B[Bidx];
		}

		Acache[threadIdx.y][threadIdx.x] = Aval;
		Bcache[threadIdx.x][threadIdx.y] = Bval;
		__syncthreads();
		for(int k = 0; k<TILE_WIDTH; k++) {
			rez += Acache[threadIdx.y][k] * Bcache[threadIdx.x][k];
		}
		__syncthreads();
	}
	if (CkolIdx < kolB && CredIdx < redA) {
		C[CredIdx*kolB + CkolIdx] = rez;
	}	
}

/**
* na vlez se prakja matrica A normalna i matrica B transponirana
* rezultatot e C e mnozenje na A * B kako B da ne e transponirana
* argumentite za golemina na matricite se prakjaat kako B da NE e transponirana
* vnatre, posledovatelni niski citaat posledovatelni elementi od A koga prefrlaat od globalna vo spodelena
* isto posledovatelni niski citaat po redici za B koga prefrlaat vo shared, a vo shared zapisuvat po redici
* sto znaci blokceto od B ostanuva transponirano pred da se mnozi.
* FOR-ot sto pravi skalaren prozivod e kako za B transponirana odnosno i na dvete matrici izminuva po redici.
*/
template <int TILE_WIDTH>
__global__ void matrixMulBtrTilesKernelV3(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	__shared__ float Acache[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bcache[TILE_WIDTH][TILE_WIDTH];

	int CkolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int CredIdx = blockDim.y * blockIdx.y + threadIdx.y;
	float rez = 0.0f;

	for (int p = 0; p<((kolARedB+TILE_WIDTH-1)/TILE_WIDTH); p++) {
		int AredIdx = CredIdx;
		int AkolIdx = p*TILE_WIDTH + threadIdx.x;
		int Aidx = CredIdx * kolARedB + p*TILE_WIDTH + threadIdx.x;
		
		int BtrRedIdx = blockDim.x * blockIdx.x + threadIdx.y;
		int BtrKolIdx = p*TILE_WIDTH + threadIdx.x;
		int Bidx = BtrRedIdx * kolARedB + BtrKolIdx;
		
		float Aval = 0.0f;
		float Bval = 0.0f;
		if (AredIdx < redA && AkolIdx < kolARedB) {
			Aval = A[Aidx];
		}
		if (BtrRedIdx < kolB && BtrKolIdx < kolARedB) {
			Bval = B[Bidx];
		}

		Acache[threadIdx.y][threadIdx.x] = Aval;
		Bcache[threadIdx.y][threadIdx.x] = Bval;
		__syncthreads();
		for(int k = 0; k<TILE_WIDTH; k++) {
			rez += Acache[threadIdx.y][k] * Bcache[threadIdx.x][k];
		}

		__syncthreads();
	}
	if (CkolIdx < kolB && CredIdx < redA) {
		C[CredIdx*kolB + CkolIdx] = rez;
	}	
}

/**
* na vlez se prakja matrica A normalna i matrica B transponirana
* rezultatot e C e mnozenje na A * B kako B da ne e transponirana
* argumentite za golemina na matricite se prakjaat kako B da NE e transponirana
* vnatre, posledovatelni niski citaat posledovatelni elementi od A koga prefrlaat od globalna vo spodelena
* isto posledovatelni niski citaat po redici za B koga prefrlaat vo shared, a vo shared zapisuvat po KOLONI
* sto znaci blokceto od B se odtransponira pred da se mnozi.
* FOR-ot sto pravi skalaren prozivod e kako kaj obicno mnozenje odnosno za A izminuva redici a za B koloni.
*/
template <int TILE_WIDTH>
__global__ void matrixMulBtrTilesKernelV4(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	__shared__ float Acache[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bcache[TILE_WIDTH][TILE_WIDTH];

	int CkolIdx = blockDim.x * blockIdx.x + threadIdx.x;
	int CredIdx = blockDim.y * blockIdx.y + threadIdx.y;
	float rez = 0.0f;

	for (int p = 0; p<((kolARedB+TILE_WIDTH-1)/TILE_WIDTH); p++) {
		int AredIdx = CredIdx;
		int AkolIdx = p*TILE_WIDTH + threadIdx.x;
		int Aidx = CredIdx * kolARedB + p*TILE_WIDTH + threadIdx.x;
		
		int BtrRedIdx = blockDim.x * blockIdx.x + threadIdx.y;
		int BtrKolIdx = p*TILE_WIDTH + threadIdx.x;
		int Bidx = BtrRedIdx * kolARedB + BtrKolIdx;
		
		float Aval = 0.0f;
		float Bval = 0.0f;
		if (AredIdx < redA && AkolIdx < kolARedB) {
			Aval = A[Aidx];
		}
		if (BtrRedIdx < kolB && BtrKolIdx < kolARedB) {
			Bval = B[Bidx];
		}

		Acache[threadIdx.y][threadIdx.x] = Aval;
		Bcache[threadIdx.x][threadIdx.y] = Bval;
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

/**
* na vlez se prakja matrica A normalna i matrica B normalna
* rezultatot e C e mnozenje na A * B
* vnatre, posledovatelni niski citaat posledovatelni elementi od A koga prefrlaat od globalna vo spodelena
* isto posledovatelni niski citaat po redici za B koga prefrlaat vo shared, a vo shared zapisuvat po koloni
* sto znaci blokceto od B se transponira pred da se mnozi.
* FOR-ot sto pravi skalaren prozivod e kako za B transponirana odnosno i na dvete matrici izminuva po redici.
*/
template <int TILE_WIDTH>
__global__ void matixMulTilesInsideTransposeKernel(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
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
		
		int BredIdx = p*TILE_WIDTH + threadIdx.y;
		int BkolIdx = CkolIdx;
		int Bidx = BredIdx * kolB + BkolIdx;
		
		float Aval = 0.0f;
		float Bval = 0.0f;
		if (AredIdx < redA && AkolIdx < kolARedB) {
			Aval = A[Aidx];
		}
		if (BredIdx < kolARedB && BkolIdx < kolB) {
			Bval = B[Bidx];
		}
		Acache[threadIdx.y][threadIdx.x] = Aval;
		Bcache[threadIdx.x][threadIdx.y] = Bval;
		__syncthreads();
		for(int k = 0; k<TILE_WIDTH; k++) {
			rez += Acache[threadIdx.y][k] * Bcache[threadIdx.x][k];
		}
		__syncthreads();
	}
	if (CkolIdx < kolB && CredIdx < redA) {
		C[CredIdx*kolB + CkolIdx] = rez;
	}	
}

void transposeCpuColToRow(const float * inMatrix, float * outMatrix, int redici, int koloni)
{
	for (int j = 0; j<koloni; j++) {
		for (int i = 0; i<redici; i++) {
			outMatrix[j*redici + i] = inMatrix[i*koloni + j];
		}
	}
}

void matMulCpuBTransp(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	for (int i = 0; i < redA; i++) {
		for (int j = 0; j< kolB; j++) {
			float s = 0.0f;
			const float * Apok = A + i*kolARedB;
			const float * Bpok = B + j*kolARedB;
			for (int k = 0; k<kolARedB; k++) {
				s += *Apok * *Bpok;
				Apok++;
				Bpok++;
			}
			C[i * kolB + j] = s;
		}
	}
}

void matMulCpuOptimisedTransp(const float * A, const float * B, float * C, int redA, int kolARedB, int kolB)
{
	float * Btr = (float *)calloc(kolARedB*kolB, sizeof(float));
	transposeCpuColToRow(B, Btr, kolARedB, kolB);
	matMulCpuBTransp(A, Btr, C, redA, kolARedB, kolB);
	free(Btr);
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

void proveriOdGreski()
{
	float * h_Amat, * h_Bmat, * h_Cmat, * h_CmatRef, * h_BtrMat;
	float * d_Amat, * d_Bmat, * d_Cmat, * d_BtrMat;
	int maxN = 1200;
	int maxLen = maxN * maxN;
	h_Amat = (float*)calloc(maxLen, sizeof(float));
	h_Bmat = (float*)calloc(maxLen, sizeof(float));
	h_Cmat = (float*)calloc(maxLen, sizeof(float));
	h_CmatRef = (float*)calloc(maxLen, sizeof(float));
	h_BtrMat = (float*)calloc(maxLen, sizeof(float));

	cudaMalloc(&d_Amat, maxLen * sizeof(float));
	cudaMalloc(&d_Bmat, maxLen * sizeof(float));
	cudaMalloc(&d_Cmat, maxLen * sizeof(float));
	cudaMalloc(&d_BtrMat, maxLen * sizeof(float));

	srand(5);
	for (int i = 0; i<maxLen; i++) {
		h_Amat[i] = h_Bmat[i] = (float)(rand() % 100);
	}
	cudaMemcpy(d_Amat, h_Amat, maxLen * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Bmat, h_Bmat, maxLen * sizeof(float), cudaMemcpyHostToDevice);

	for (int N = 5; N <= maxN; N++) {
		transposeCpuColToRow(h_Bmat, h_BtrMat, N, N);
		matMulCpuBTransp(h_Amat, h_BtrMat, h_CmatRef, N, N, N);

		dim3 blockDim(TILE_WIDTH1, TILE_WIDTH1, 1);
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		transposeFullParralel2DBlocking<<<gridDim, blockDim>>>(d_Bmat, d_BtrMat, N, N);
		matrixMulBtrNaiveKernel<<<gridDim, blockDim>>>(d_Amat, d_BtrMat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("gpu tranp mul naive N=%d greska\n", N);
		} else {
			printf("gpu tranp mul naive N=%d tocno\n", N);
		}

		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matrixMulBtrTilesKernelV1<TILE_WIDTH1><<<gridDim, blockDim>>>(d_Amat, d_BtrMat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("gpu tranp mul tiled v1 N=%d greska\n", N);
		} else {
			printf("gpu tranp mul tiled v1 N=%d tocno\n", N);
		}

		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matrixMulBtrTilesKernelV2<TILE_WIDTH1><<<gridDim, blockDim>>>(d_Amat, d_BtrMat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("gpu tranp mul tiled v2 N=%d greska\n", N);
		} else {
			printf("gpu tranp mul tiled v2 N=%d tocno\n", N);
		}
		
		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matrixMulBtrTilesKernelV3<TILE_WIDTH1><<<gridDim, blockDim>>>(d_Amat, d_BtrMat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("gpu tranp mul tiled v3 N=%d greska\n", N);
		} else {
			printf("gpu tranp mul tiled v3 N=%d tocno\n", N);
		}

		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matrixMulBtrTilesKernelV4<TILE_WIDTH1><<<gridDim, blockDim>>>(d_Amat, d_BtrMat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("gpu tranp mul tiled v4 N=%d greska\n", N);
		} else {
			printf("gpu tranp mul tiled v4 N=%d tocno\n", N);
		}

		cudaMemset(d_Cmat, 0, maxLen*sizeof(float));
		matixMulTilesInsideTransposeKernel<TILE_WIDTH1><<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaMemcpy(h_Cmat, d_Cmat, N*N*sizeof(float), cudaMemcpyDeviceToHost);
		if (proveriGreska(h_CmatRef, h_Cmat, N*N)) {
			printf("gpu tiled inside transp N=%d greska\n", N);
		} else {
			printf("gpu tiled inside transp N=%d tocno\n", N);
		}
	}

	cudaFree(d_Amat);
	cudaFree(d_Bmat);
	cudaFree(d_Cmat);
	cudaFree(d_BtrMat);

	free(h_Amat);
	free(h_Bmat);
	free(h_Cmat);
	free(h_CmatRef);
	free(h_BtrMat);
}

template <int TILE_WIDTH>
void benchMatMulBtrNaive(int startN, int endN)
{
	float * d_Amat, * d_Bmat, * d_Cmat;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	printf("GPU mat mul Btr Naive\n");
	printf("N;tile width;vreme\n");
	for (int N = startN; N < endN; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matrixMulBtrNaiveKernel<<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template <int TILE_WIDTH>
void benchMatMulBtrTiledV1(int startN, int endN)
{
	float * d_Amat, * d_Bmat, * d_Cmat;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	printf("GPU mat mul Btr Tiled v1\n");
	printf("N;tile width;vreme\n");
	for (int N = startN; N < endN; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matrixMulBtrTilesKernelV1<TILE_WIDTH><<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template <int TILE_WIDTH>
void benchMatMulBtrTiledV2(int startN, int endN)
{
	float * d_Amat, * d_Bmat, * d_Cmat;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	printf("GPU mat mul Btr Tiled v2\n");
	printf("N;tile width;vreme\n");
	for (int N = startN; N < endN; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matrixMulBtrTilesKernelV2<TILE_WIDTH><<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template <int TILE_WIDTH>
void benchMatMulBtrTiledV3(int startN, int endN)
{
	float * d_Amat, * d_Bmat, * d_Cmat;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	printf("GPU mat mul Btr Tiled v3\n");
	printf("N;tile width;vreme\n");
	for (int N = startN; N < endN; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matrixMulBtrTilesKernelV3<TILE_WIDTH><<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template <int TILE_WIDTH>
void benchMatMulBtrTiledV4(int startN, int endN)
{
	float * d_Amat, * d_Bmat, * d_Cmat;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	printf("GPU mat mul Btr Tiled v4\n");
	printf("N;tile width;vreme\n");
	for (int N = startN; N < endN; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matrixMulBtrTilesKernelV4<TILE_WIDTH><<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

template <int TILE_WIDTH>
void benchMatMulTiledBInsideTransp(int startN, int endN)
{
	float * d_Amat, * d_Bmat, * d_Cmat;
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	printf("GPU mat mul Tiled Inside Transpose\n");
	printf("N;tile width;vreme\n");
	for (int N = startN; N < endN; N+=1) {
		cudaMalloc(&d_Amat, N*N*sizeof(float));
		cudaMalloc(&d_Bmat, N*N*sizeof(float));
		cudaMalloc(&d_Cmat, N*N*sizeof(float));
		dim3 gridDim((N+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y, 1);
		cudaEventRecord(start);
		matixMulTilesInsideTransposeKernel<TILE_WIDTH><<<gridDim, blockDim>>>(d_Amat, d_Bmat, d_Cmat, N, N, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&vreme, start, end);
		cudaFree(d_Amat);
		cudaFree(d_Bmat);
		cudaFree(d_Cmat);
		printf("%d;%d;%f\n", N, TILE_WIDTH, vreme);
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void benchmark()
{
	const int startN = 32, endN = 750;
	benchMatMulBtrNaive<8>(startN, endN);
	benchMatMulBtrNaive<12>(startN, endN);
	benchMatMulBtrNaive<16>(startN, endN);
	benchMatMulBtrNaive<20>(startN, endN);
	benchMatMulBtrNaive<24>(startN, endN);
	benchMatMulBtrNaive<28>(startN, endN);
	benchMatMulBtrNaive<32>(startN, endN);

	benchMatMulBtrTiledV1<8>(startN, endN);
	benchMatMulBtrTiledV1<12>(startN, endN);
	benchMatMulBtrTiledV1<16>(startN, endN);
	benchMatMulBtrTiledV1<20>(startN, endN);
	benchMatMulBtrTiledV1<24>(startN, endN);
	benchMatMulBtrTiledV1<28>(startN, endN);
	benchMatMulBtrTiledV1<32>(startN, endN);

	benchMatMulBtrTiledV2<8>(startN, endN);
	benchMatMulBtrTiledV2<12>(startN, endN);
	benchMatMulBtrTiledV2<16>(startN, endN);
	benchMatMulBtrTiledV2<20>(startN, endN);
	benchMatMulBtrTiledV2<24>(startN, endN);
	benchMatMulBtrTiledV2<28>(startN, endN);
	benchMatMulBtrTiledV2<32>(startN, endN);

	benchMatMulBtrTiledV3<8>(startN, endN);
	benchMatMulBtrTiledV3<12>(startN, endN);
	benchMatMulBtrTiledV3<16>(startN, endN);
	benchMatMulBtrTiledV3<20>(startN, endN);
	benchMatMulBtrTiledV3<24>(startN, endN);
	benchMatMulBtrTiledV3<28>(startN, endN);
	benchMatMulBtrTiledV3<32>(startN, endN);

	benchMatMulBtrTiledV4<8>(startN, endN);
	benchMatMulBtrTiledV4<12>(startN, endN);
	benchMatMulBtrTiledV4<16>(startN, endN);
	benchMatMulBtrTiledV4<20>(startN, endN);
	benchMatMulBtrTiledV4<24>(startN, endN);
	benchMatMulBtrTiledV4<28>(startN, endN);
	benchMatMulBtrTiledV4<32>(startN, endN);

	benchMatMulTiledBInsideTransp<8>(startN, endN);
	benchMatMulTiledBInsideTransp<12>(startN, endN);
	benchMatMulTiledBInsideTransp<16>(startN, endN);
	benchMatMulTiledBInsideTransp<20>(startN, endN);
	benchMatMulTiledBInsideTransp<24>(startN, endN);
	benchMatMulTiledBInsideTransp<28>(startN, endN);
	benchMatMulTiledBInsideTransp<32>(startN, endN);
}

int main()
{
	benchmark();
}