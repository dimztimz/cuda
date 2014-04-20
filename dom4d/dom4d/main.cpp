#include <iostream>
#include "EasyBMP\EasyBMP.h"
#include "cestotaBoi.h"

using namespace std;

int main(int argc, char * argv[])
{
	char * datotekaIme;
	if (argc < 2) {
		return 0;
	} else {
		datotekaIme = argv[1];
	}
	BMP slika;
	slika.ReadFromFile(datotekaIme);
	int brojPixeli = slika.TellHeight() * slika.TellWidth();
	uchar4 * pixeli = new uchar4[brojPixeli];
	for (int y = 0, i = 0; y < slika.TellHeight(); y++) {
		for (int x = 0; x < slika.TellWidth(); x++, i++) {
			pixeli[i].x = slika(x, y)->Red;
			pixeli[i].y = slika(x, y)->Green;
			pixeli[i].z = slika(x, y)->Blue;
			pixeli[i].w = slika(x, y)->Alpha;
		}
	}
	int brojKofickiPoBoja = 32;
	int * _3dKoficki = new int[brojKofickiPoBoja*brojKofickiPoBoja*brojKofickiPoBoja];
	presmetajCestotaNaBoi(pixeli, slika.TellWidth(), slika.TellHeight(), _3dKoficki, brojKofickiPoBoja);
	return 0;
}