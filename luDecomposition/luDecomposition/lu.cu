#include <iostream>
#include <fstream>

#include <time.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <omp.h>

using namespace std;

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
void printMatrix(T * mat, int red, int kol, ostream& out = std::cout)
{
	for (int i = 0; i < red; i++) {
		for (int j = 0; j < red; j++) {
			out << mat[i*kol + j] << ' ';
		}
		out << endl;
	}
}

template <typename T>
void lu(T * d_L, T * d_U, int n, const int NUMTHREAD = 128)
{
	cudaMemset(d_L, 0, sizeof(T)*n*n);
	for (int i = 0; i < n-1; i++) {
		int lu1blocks = (n-i + NUMTHREAD - 1) / NUMTHREAD;
		int lu2blocks = ((n-i-1)*(n-i) + NUMTHREAD - 1) / NUMTHREAD;
		lu1<T> << <lu1blocks, NUMTHREAD >> >(d_L, d_U, n, i);
		lu2<T> << <lu2blocks, NUMTHREAD >> >(d_L, d_U, n, i);
	}
	lu1<T> << <1, 1 >> >(d_L, d_U, n, n-1);
}

template <typename T>
void matMulCpuNaive(const T * A, const T * B, T * C, int redA, int kolARedB, int kolB)
{
	for (int i = 0; i < redA; i++) {
		for (int j = 0; j< kolB; j++) {
			T s = 0.0f;
			const T * Apok = A + i*kolARedB;
			const T * Bpok = B + j;
			for (int k = 0; k<kolARedB; k++) {
				s += *Apok * *Bpok;
				Apok++;
				Bpok += kolB;
			}
			C[i * kolB + j] = s;
		}
	}
}

template <typename T>
T proveriGreska(const T * referenceMat, const T * presmetanaMat, int len)
{
	T maxGreska = 0.0f;
	for (int i = 0; i<len; i++) {
		if (abs(referenceMat[i] - presmetanaMat[i]) > maxGreska) {
			maxGreska = abs(referenceMat[i] - presmetanaMat[i]);
		}
	}
	return maxGreska;
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
	printMatrix<float>(L, n, n);
	cout << '*' << endl;
	printMatrix<float>(U, n, n);
	cout << '=' << endl;
	matMulCpuNaive<float>(L, U, novoA, n, n, n);
	printMatrix<float>(novoA, n, n);
	bool greska = proveriGreska<float>(A, novoA, n*n) > 0.0001;
	if (greska) {
		cout << "razlicno so referentna matrica, GRESKA" << endl;
		printMatrix<float>(A, n, n);
	}
	else {
		cout << "ednakvo so referentna matrica, TOCNO" << endl;
	}
	return greska;
}

bool randomMatrixTest()
{
	srand((unsigned int)time(NULL));
	int n = rand()%1000;
	for (int i = n; i--;) {
		n = rand() % 1000;
	}
	cout << "Test na A = LU, A e so slucajni broevi i ima golemina " << n << endl;
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

	matMulCpuNaive<float>(L, U, novoA, n, n, n);
	float maxGreska = proveriGreska<float>(A, novoA, n*n);
	bool greska = maxGreska > 0.01;
	if (greska) {
		cout << "razlicno so referentna matrica, najgolema greska e " << maxGreska << endl;
		/*
		ofstream izlezL("L.txt");
		ofstream izlezU("U.txt");
		ofstream izlezRefA("refA.txt");
		ofstream izlezNovoA("novoA.txt");
		printMatrix(L, n, n, izlezL);
		printMatrix(U, n, n, izlezU);
		printMatrix(A, n, n, izlezRefA);
		printMatrix(novoA, n, n, izlezNovoA);
		//datotekite avotamatski se zatvoraat preku destruktorite koi avotmatski se povikuvaat tuka
		*/
	} else {
		cout << "priblizno ednakvo so referentna matrica, TOCNO, greksa: " << maxGreska  << endl;
	}

	delete[] A;
	delete[] L;
	delete[] U;
	delete[] novoA;
	return greska;
}

template <typename T>
void cpuLUDecomposition(T * L, T * U, int n)
{
	memset(L, 0, n*n*sizeof(T));
	for (int i = 0; i < n; i++) {
		L[i*n + i] = 1;
		for (int j = i + 1; j < n; j++) {
			T koef = U[j*n + i] / U[i*n + i];
			L[j*n + i] = koef;
			for (int k = i; k < n; k++) {
				U[j*n + k] -= koef * U[i*n + k];
			}
		}
	}
}

bool test1cpuLU()
{
	float A[16] = { 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 10, 11, 11, 12, 13, 15 };
	float L[16], U[16], novoA[16];
	int n = 4;
	memcpy(U, A, sizeof(A));
	cpuLUDecomposition<float>(L, U, n);
	printMatrix<float>(L, n, n);
	cout << '*' << endl;
	printMatrix<float>(U, n, n);
	cout << '=' << endl;
	matMulCpuNaive<float>(L, U, novoA, n, n, n);
	printMatrix<float>(novoA, n, n);
	bool greska = proveriGreska<float>(A, novoA, n*n) > 0.0001;
	if (greska) {
		cout << "razlicno so referentna matrica, GRESKA" << endl;
		printMatrix<float>(A, n, n);
	}
	else {
		cout << "ednakvo so referentna matrica, TOCNO" << endl;
	}
	return greska;
}

bool randomMatrixTestCPU()
{
	srand((unsigned int)time(NULL));
	int n = rand() % 1000;
	for (int i = n; i--;) {
		n = rand() % 1000;
	}
	cout << "Test na A = LU, A e so slucajni broevi i ima golemina " << n << endl;
	float * A = new float[n*n];
	float * L = new float[n*n];
	float * U = new float[n*n];
	float * novoA = new float[n*n];

	for (int i = 0; i < n*n; i++) {
		A[i] = (float)(rand() % 1000);
	}
	memcpy(U, A, sizeof(float)*n*n);
	cpuLUDecomposition<float>(L, U, n);
	matMulCpuNaive<float>(L, U, novoA, n, n, n);
	float maxGreska = proveriGreska<float>(A, novoA, n*n);
	bool greska = maxGreska > 0.01;
	if (greska) {
		cout << "razlicno so referentna matrica, najgolema greska e " << maxGreska << endl;
	}
	else {
		cout << "priblizno ednakvo so referentna matrica, TOCNO, greksa: " << maxGreska << endl;
	}

	delete[] A;
	delete[] L;
	delete[] U;
	delete[] novoA;
	return greska;
}

void benchmark()
{
	int MIN_SIZE = 32, MAX_SIZE = 2100;
	float * AA = new float[MAX_SIZE*MAX_SIZE];
	srand((unsigned int)time(NULL));
	for (int i = 0; i < MAX_SIZE*MAX_SIZE; i++) {
		AA[i] = (float)(rand() % 1000);
	}
	/*
	Pravime benchmark taka sto za razlicni golemini na matrici, t.e. za razlicno N merime vreme T_N_i
	kade T_N_i ni e vreme na edna LU za matrica so golemina N. Ova go smetame za eden eksperiment.
	Za sekoe N, ekperimentot go povtoruvame povt(N) pati i potoa presmetuvame prosecno vreme T_N.
	T_N = Suma(T_N_i, i=1..povt(N))/N
	povt kje ni bude negativna stepenska funkcija povt(N)=a/N^3+b.
	za MIN_SIZE kje imame POVT_MIN_SIZE = 100000
	za MAX_SIZE kje imame POVT_MAX_SIZE = 100
	a i b od povt(N) gi procenuvame preku sistem lieanrni ravenki
	a/MIN_SIZE^3 + b = POVT_MIN_SIZE
	a/MAX_SIZE^3 + b = POVT_MAX_SIZE
	*/
	int POVT_MAX_SIZE = 100, POVT_MIN_SIZE = 100000;
	double povtA = (POVT_MAX_SIZE - POVT_MIN_SIZE)*1.0 / (1.0 / MAX_SIZE / MAX_SIZE / MAX_SIZE - 1.0 / MIN_SIZE / MIN_SIZE / MIN_SIZE);
	double povtB = 1.0*POVT_MAX_SIZE*MAX_SIZE*MAX_SIZE*MAX_SIZE - 1.0*POVT_MIN_SIZE*MIN_SIZE*MIN_SIZE*MIN_SIZE;
	povtB /= 1.0*MAX_SIZE*MAX_SIZE*MAX_SIZE - 1.0*MIN_SIZE*MIN_SIZE*MIN_SIZE;
	cout << "LU on CPU" << endl;
	cout << "N,povtoruvanja,vkupno vreme (s),prosecno vreme (s)" << endl;
	for (int N = MIN_SIZE; N < MAX_SIZE; N += 32) {
		float * A = new float[N*N];
		float * L = new float[MAX_SIZE*MAX_SIZE * 4];
		float * U = new float[MAX_SIZE*MAX_SIZE * 4];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i*N + j] = AA[i*MAX_SIZE + j];
			}
		}
		int povt = (int)(povtA / N / N / N + povtB + 0.5);
		double totalTime = 0.0;
		//Offsetot ni e za sekoe povtoruvanej na eksprimetot da rabotime so "necepnata" meorija, memorija sto ne se naogja vo kesovite
		for (int j = 0, offset = 0; j < povt; j++, offset += N*N) {
			offset %= MAX_SIZE*MAX_SIZE * 4;
			if (offset + N*N > MAX_SIZE*MAX_SIZE * 4) {
				offset = 0;
			}
			
			memcpy(U+offset, A, sizeof(float)*N*N);
			double start = omp_get_wtime();
			cpuLUDecomposition<float>(L+offset, U+offset, N);
			double end = omp_get_wtime();
			totalTime += end - start;
		}
		cout << N << ',' << povt << ',' << totalTime << ',' << totalTime / povt << endl;
		delete[] L;
		delete[] U;
		delete[] A;
	}
	
	cout << "cuda LU on GPU" << endl;
	cout << "N,povtoruvanja,vkupno vreme (ms),prosecno vreme (ms)" << endl;
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	for (int N = MIN_SIZE; N < MAX_SIZE; N += 32) {

		float * d_L, *d_U;
		cudaMalloc(&d_L, sizeof(float)*MAX_SIZE*MAX_SIZE * 4);
		cudaMalloc(&d_U, sizeof(float)*MAX_SIZE*MAX_SIZE * 4);
		float * A = new float[N*N];
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i*N + j] = AA[i*MAX_SIZE + j];
			}
		}
		int povt = (int)(povtA / N / N / N + povtB + 0.5);
		float totalTime = 0.0;
		for (int j = 0, offset = 0; j < povt; j++, offset += N*N) {
			offset %= MAX_SIZE*MAX_SIZE * 4;
			if (offset + N*N > MAX_SIZE*MAX_SIZE * 4) {
				offset = 0;
			}

			cudaMemcpy(d_U + offset, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
			cudaEventRecord(start);
			lu<float>(d_L + offset, d_U + offset, N);
			cudaEventRecord(end);
			cudaEventSynchronize(end);
			float time;
			cudaEventElapsedTime(&time, start, end);
			totalTime += time;
		}
		cout << N << ',' << povt << ',' << totalTime << ',' << totalTime / povt << endl;
		cudaFree(d_L);
		cudaFree(d_U);
		delete[] A;
	}
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	delete[] AA;
}

int main()
{
	benchmark();
	system("pause");
	return 0;
}
