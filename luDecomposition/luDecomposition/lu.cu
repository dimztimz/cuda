#include <iostream>
#include <algorithm>

#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

template <typename T>
__global__ void luStaro(T * matrica, T * lMat, T * uMat, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx >= n*n) {
		return;
	}

	int redica = idx / n;
	int kolona = idx % n;

	uMat[idx] = matrica[idx];

	lMat[idx] = 0;
	for (int i = 0; i < n; i++) {
		// presmetaj koeficienti
		
		/*
		//so sledniov kod prvite n posledovatelni niski od celata mreza zapisuvaat vo edna kolona od L
		if (idx < i) { //vo ovoj if vleguvaat prvite i niski od celata mreza t.e. od 0 do i-1
			lMat[idx * n + i] = 0;
		} else if (idx < n) { // a vo ovoj ostanatite do n t.e od i do n-1
			lMat[idx * n + i] = uMat[idx * n + i] / uMat[i * n + i];
			//printf("%f ", lMat[idx * n + i]);
		}
		*/
		if (kolona == i && redica >= i) {
			lMat[redica*n + kolona] = uMat[redica*n + kolona] / uMat[kolona*n + kolona];
		}
		//problemot e tuka, nekoi blokovi pocnuva eliminacija pred da se presmeta koeficietot koj go stavame vo L
		//na mestovo kade st ostoi komentarov idealno treba da stoi bariera na celata mreza, a ne samo na vo blok
		//edno resenie da presmetuvame koef vo L, a potoa sekoja niska posebno uste pri samata eliminacija da si presmetuva

		__threadfence(); // ne vrsi rabota
		//eliminacija
		if (redica > i) {
			uMat[redica * n + kolona] -= lMat[redica * n + i] * uMat[i * n + kolona];
		}
		
		__threadfence();

	}
}

template <typename T>
__global__ void lu1(T * L, T * U, int n, int i)
{
	//vo ovoj cekor presmetivame koef so koj kje mnozime gorna redica i kje gi eliminrame redicite pod nea
	//idx ni e redica pocnuvanjki od A[i], i ni e kolona
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int redica = idx + i;
	if (redica < n) {
		L[redica*n + i] = U[redica*n + i] / U[i*n + i];
	}
}

template <typename T>
__global__ void lu2(T * L, T * U, int n, int i)
{
	//ovde idx ni e globalen indeks na niskata i globalen indeks vo matricata (row major) pocnuvajki od U[i+1][i]
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= (n-i-1)*(n-i)) {
		return;
	}
	//(n - i) sirina na redica sto ja obrabotivame. delime so sorinata i gi dodavame pomestuvanjeata t.e. offsetite
	int redica = idx / (n - i) + i + 1;
	int kolona = idx % (n - i) + i;
	U[redica * n + kolona] -= L[redica * n + i] * U[i * n + kolona];
}

template <typename T>
void printMatrix(T * mat, int red, int kol)
{
	for (int i = 0; i < red; i++) {
		for (int j = 0; j < red; j++) {
			cout << mat[i*kol + j] << ' ';
		}
		cout << endl;
	}
}

template <typename T>
void lu(T * d_L, T * d_U, int n)
{
	const int NUMTHREAD = 128;

	float * L = new float[n*n];
	float * U = new float[n*n];

	/*cudaMemcpy(U, d_U, sizeof(U[0])*n*n, cudaMemcpyDeviceToHost);
	printMatrix<float>(U, n, n);
	cout << "======" << endl;*/

	cudaMemset(d_L, 0, sizeof(T)*n*n);
	for (int i = 0; i < n-1; i++) {
		int lu1blocks = (n-i + NUMTHREAD - 1) / NUMTHREAD;
		int lu2blocks = ((n-i-1)*(n-i) + NUMTHREAD - 1) / NUMTHREAD;
		lu1<T> << <lu1blocks, NUMTHREAD >> >(d_L, d_U, n, i);
		lu2<T> << <lu2blocks, NUMTHREAD >> >(d_L, d_U, n, i);

		/*cudaMemcpy(L, d_L, sizeof(L[0])*n*n, cudaMemcpyDeviceToHost);
		cudaMemcpy(U, d_U, sizeof(U[0])*n*n, cudaMemcpyDeviceToHost);

		printMatrix<float>(L, n, n);
		cout << '*' << endl;
		printMatrix<float>(U, n, n);
		cout << "======" << endl;*/
	}
	lu1<T> << <1, 1 >> >(d_L, d_U, n, n-1);

	delete[] L; delete[] U;
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

bool test1()
{
	float A[16] = { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 10, 11, 11, 12, 13, 15 };
	float L[16], U[16], novoA[16];
	int n = 4;
	float *d_U, *d_L;
	cudaMalloc(&d_L, sizeof(L));
	cudaMalloc(&d_U, sizeof(U));
	cudaMemcpy(d_U, A, sizeof(A), cudaMemcpyHostToDevice);
	lu<float>(d_L, d_U, n);
	cudaMemcpy(L, d_L, sizeof(L), cudaMemcpyDeviceToHost);
	cudaMemcpy(U, d_U, sizeof(U), cudaMemcpyDeviceToHost);
	cudaFree(d_L);
	cudaFree(d_U);
	printMatrix<float>(L, 4, 4);
	cout << '*' << endl;
	printMatrix<float>(U, 4, 4);
	cout << '=' << endl;
	matMulCpuNaive(L, U, novoA, 4, 4, 4);
	printMatrix<float>(novoA, 4, 4);
	bool greska = proveriGreska(A, novoA, 16);
	if (greska) {
		cout << "razlicno so referentna matrica, GRESKA" << endl;
		printMatrix<float>(A, 4, 4);
	}
	else {
		cout << "ednakvo so referentna matrica, TOCNO" << endl;
	}
	return greska;
}

bool randomMatrixTest()
{
	srand(time(NULL));
	int n = 6;
	float * A = new float[n*n];
	float * L = new float[n*n];
	float * U = new float[n*n];
	float * novoA = new float[n*n];

	for (int i = 0; i < n*n; i++) {
		A[i] = (float)(rand() % 1000);
	}

	float *d_U, *d_L;
	cudaMalloc(&d_L, sizeof(L[0])*n*n);
	cudaMalloc(&d_U, sizeof(U[0])*n*n);
	cudaMemcpy(d_U, A, sizeof(A[0])*n*n, cudaMemcpyHostToDevice);
	lu<float>(d_L, d_U, n);
	cudaMemcpy(L, d_L, sizeof(L[0])*n*n, cudaMemcpyDeviceToHost);
	cudaMemcpy(U, d_U, sizeof(U[0])*n*n, cudaMemcpyDeviceToHost);
	cudaFree(d_L);
	cudaFree(d_U);

	printMatrix<float>(L, n, n);
	cout << '*' << endl;
	printMatrix<float>(U, n, n);
	cout << '=' << endl;
	matMulCpuNaive(L, U, novoA, n, n, n);
	printMatrix<float>(novoA, n, n);
	bool greska = proveriGreska(A, novoA, n*n);
	if (greska) {
		cout << "razlicno so referentna matrica, GRESKA" << endl;
		printMatrix<float>(A, n, n);
	}
	else {
		cout << "ednakvo so referentna matrica, TOCNO" << endl;
	}
	delete[] A;
	delete[] L;
	delete[] U;
	delete[] novoA;
	return greska;
}

int main()
{
	test1();
	cout << endl;
	randomMatrixTest();
	system("pause");
	return 0;
}
