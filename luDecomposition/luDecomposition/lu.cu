#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;

template <typename T>
__global__ void lu(T * matrica, T * lMat, T * uMat, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= n*n) {
		return;
	}

	int redica = idx / n;
	int kolona = idx % n;

	uMat[idx] = matrica[idx];
	//lMat[idx] = 0;
	for (int i = 0; i < n; i++) {
		// presmetaj koeficienti
		if (idx < i) {
			lMat[idx * n + i] = 0;
		} else if (idx < n) {
			lMat[idx * n + i] = uMat[idx * n + i] / uMat[i * n + i];
			//printf("%f ", lMat[idx * n + i]);
		}

		//eliminacija
		if (redica > i) {
			uMat[redica * n + kolona] -= lMat[redica * n + i] * uMat[i * n + kolona];
		}
		
		__threadfence();

	}
}


template <typename T>
void printMatrix(T * mat, int red, int kol)
{
	for(int i = 0; i < red; i++) {
		for(int j = 0; j < red; j++) {
			cout << mat[i*kol + j] << ' ';
		}
		cout << endl;
	}
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

int main()
{
	float A[16] = {1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 10, 11, 11, 12, 13, 15};
	float L[16], U[16], novoA[16];

	float *d_A, *d_U, *d_L;
	cudaMalloc(&d_A, sizeof(A));
	cudaMalloc(&d_L, sizeof(L));
	cudaMalloc(&d_U, sizeof(U));
	cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);
	lu<float><<<1,16>>>(d_A, d_L, d_U, 4);
	cudaMemcpy(L, d_L, sizeof(L), cudaMemcpyDeviceToHost);
	cudaMemcpy(U, d_U, sizeof(U), cudaMemcpyDeviceToHost);

	printMatrix<float>(L, 4, 4);
	cout << '*' << endl;
	printMatrix<float>(U, 4, 4);
	cout << '=' << endl;
	matMulCpuNaive(L, U, novoA, 4,4,4);
	printMatrix<float>(novoA, 4, 4);
	bool greska = proveriGreska(A, novoA, 16);
	if (greska) {
		cout << "!= so ref, GRESKA" << endl;
		printMatrix<float>(A, 4, 4);
	} else {
		cout << "== so ref, TOCNO" << endl;
	}
	return 0;
}
