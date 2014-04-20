#include "cestotaBoi.h"
#include <iostream>
#include <algorithm>


__global__ void presmetajCestotaNaBoiCuda(const uchar4 * pixeli, const int len,
										  int *  _3dKoficki, const int brojKofickiPoBoja, const int boiVoKoficka)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= len) {
		return;
	}
	int r = pixeli[i].x;
	int g = pixeli[i].y;
	int b = pixeli[i].z;
	r /= boiVoKoficka;
	g /= boiVoKoficka;
	b /= boiVoKoficka;
	int koficka = r * brojKofickiPoBoja * brojKofickiPoBoja + g * brojKofickiPoBoja + b;
	atomicAdd(&_3dKoficki[koficka], 1);
}

void presmetajCestotaNaBoiCpu(const uchar4 * pixeli, const int len, int *  _3dKoficki, const int brojKofickiPoBoja,
							  const int boiVoKoficka)
{
	std::fill(_3dKoficki, _3dKoficki + (brojKofickiPoBoja*brojKofickiPoBoja*brojKofickiPoBoja), 0);
	for (int i = 0; i<len; i++) {
		int r = pixeli[i].x;
		int g = pixeli[i].y;
		int b = pixeli[i].z;
		r /= boiVoKoficka;
		g /= boiVoKoficka;
		b /= boiVoKoficka;
		int koficka = r * brojKofickiPoBoja * brojKofickiPoBoja + g * brojKofickiPoBoja + b;
		_3dKoficki[koficka]++;
	}
}

bool proveriGreskaCestotaBoi(int * kofPresmetani, int * kofReference, int vkupnoKoficki, int kofickiPoBoja)
{
	bool greska = false;
	for(int i = 0; i < vkupnoKoficki; i++) {
		if (kofPresmetani[i] != kofReference[i]) {
			greska = true;
			std::cout << "Greska vo koficka so idx " << i << std::endl;
		}
	}
	return greska;
}

/**
*	Presmetuva cestota na boi. Vleznata niza se ocekuva da bide matrica pretstavena kako niza vo row-major podreduvanje
*	Kofickata se presmetuva kako redKof = redPixel / brojKofickiPoBoja. slicko e za zelenata i sinata komponenta
*	potoa kofickata vo linearnata niza se dobiva so
*	koficka[redKof * brojKofickiPoBoja * brojKofickiPoBoja + greenKof * brojKofickiPoBoja + blueKof]
*	Za najdetalna cestota brojNaKoficki se stava 256.
*/
void presmetajCestotaNaBoi(const uchar4 * pixeli, const int sirina, const int visina,
						   int *  _3dKoficki, const int brojKofickiPoBoja)
{
	int len = sirina * visina;
	int brojKoficki = brojKofickiPoBoja * brojKofickiPoBoja * brojKofickiPoBoja;
	int boiVoKoficka = 256 / brojKofickiPoBoja;
	uchar4 * d_pixeli;
	int * d_koficki;
	cudaMalloc(&d_pixeli, len * sizeof(uchar4));
	cudaMalloc(&d_koficki, brojKoficki * sizeof(int));
	cudaMemcpy(d_pixeli, pixeli, len*sizeof(uchar4), cudaMemcpyHostToDevice);
	cudaMemset(d_koficki, 0, brojKoficki * sizeof(int));
	int threads = 512;
	int blocks = len / threads + (len % threads != 0);
	cudaEvent_t start, end;
	float vreme;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
	presmetajCestotaNaBoiCuda<<<blocks, threads>>>(d_pixeli, len, d_koficki, brojKofickiPoBoja, boiVoKoficka);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&vreme, start, end);
	std::cout << "pixeli=" << len << " koficki=" << brojKoficki << " thr=" << threads << " blk=" << blocks
		<< " vreme=" << vreme << std::endl;
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cudaMemcpy(_3dKoficki, d_koficki, brojKoficki*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_koficki);
	cudaFree(d_pixeli);

	int * cpuKofickiReference = new int[brojKoficki];
	presmetajCestotaNaBoiCpu(pixeli, len, cpuKofickiReference, brojKofickiPoBoja, boiVoKoficka);
	proveriGreskaCestotaBoi(_3dKoficki, cpuKofickiReference, brojKoficki, brojKofickiPoBoja);
	delete[] cpuKofickiReference;
}
