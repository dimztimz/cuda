#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <complex>

#include "julia.h"

using namespace std;

typedef struct textura
{
	unsigned int id;
	unsigned int sirina;
	unsigned int visina;
	unsigned char * niza;
} textura, * pTextura;

unsigned int texturaIDs[1] = {-1};
textura texturi[1];
float scale = 1.0;

int julia(float zReal, float zImag)
{
	complex<float> z(zReal, zImag);
	complex<float> c(-0.8f, 0.156f);

	for (int i = 0; i<256; i++) {
		z = z*z + c;
		if (norm(z) > 4.0f) {
			return i;
		}
	}
	return 0;
}

void fillJulia(textura * t, float xMin, float xMax, float yMin, float yMax)
{
	float xDelce = (xMax - xMin) / (float)t->sirina;
	float yDelce = (yMax - yMin) / (float)t->visina;
	unsigned char * niza = t->niza;
	int s = t->sirina;
	int v = t->visina;
	
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int yStart = tid * v / omp_get_num_threads();
		int yEnd = (tid+1) * v / omp_get_num_threads();
		int y;
		float zReal, zImag;
		for (y = yStart, zImag = yMin + y*yDelce; y < yEnd; y++) {
			zReal = xMin;
			for (int x = 0; x < s; x++) {
				int i = 4*(y*s + x);
				niza[i] = julia(zReal, zImag);
				niza[i+1] = 0;
				niza[i+2] = 0;
				niza[i+3] = 255;
				zReal += xDelce;
			}
			zImag += yDelce;
		}
	}
}

void updateJulia() {
	int sir = glutGet(GLUT_WINDOW_WIDTH);
	int vis = glutGet(GLUT_WINDOW_HEIGHT);
	float ar = (float) sir / (float) vis;
	if (texturi[0].niza != NULL && (int)(texturi[0].sirina*texturi[0].visina) < sir*vis) {
		delete[] texturi[0].niza;
		texturi[0].niza = new unsigned char[4*sir*vis];
	} else if (texturi[0].niza == NULL) {
		texturi[0].niza = new unsigned char[4*sir*vis];
	}
	texturi[0].sirina = sir;
	texturi[0].visina = vis;
	/*double vreme = omp_get_wtime();
	fillJulia(&texturi[0], -ar*scale, ar*scale, -scale, scale);
	vreme = omp_get_wtime() - vreme;
	printf("julia cpu: sir=%d vis=%d vreme=%lf\n", sir, vis, vreme*1000.0);*/
	fillJuliaGPU(texturi[0].niza, sir, vis, -ar*scale, ar*scale, -scale, scale);
	glBindTexture(GL_TEXTURE_2D, texturi[0].id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sir, vis, 0, GL_RGBA, GL_UNSIGNED_BYTE, texturi[0].niza);
}

static void resize(int width, int height)
{
    float ar = (float) width / (float) height;

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
	glOrtho(-ar, ar, -1.0, 1.0, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	updateJulia();
}

static void display(void)
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	float ar = (float)texturi[0].sirina / texturi[0].visina;

	glPushMatrix();
	glBindTexture(GL_TEXTURE_2D, texturi[0].id);
    glBegin(GL_QUADS);
    {
    	glTexCoord2f(1, 0);glVertex2f(-ar, -1.0);
    	glTexCoord2f(1, 1);glVertex2f(-ar, 1.0);
    	glTexCoord2f(0, 1);glVertex2f(ar, 1.0);
    	glTexCoord2f(0, 0);glVertex2f(ar, -1.0);
    }
    glEnd();
	/*glBegin(GL_POINTS);
	for (int y = 0; y<texturi[0].visina; y++) {
		for (int x = 0; x<texturi[0].sirina; x++) {
			int i = 4 * (y * texturi[0].sirina + x);
			glColor3ub(texturi[0].niza[i], texturi[0].niza[i+1], texturi[0].niza[i+2]);
			glVertex2f(x * 2 * ar / texturi[0].sirina - ar, y * 2.0 / texturi[0].visina - 1.0);
		}
	}
	glEnd();*/
    glPopMatrix();

    glutSwapBuffers();
}


static void key(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27 :
        case 'q':
            exit(0);
            break;

        case '+':
			scale /= 1.1f;
			updateJulia();
            break;

        case '-':
			scale *= 1.1f;
			updateJulia();
            break;
    }

    glutPostRedisplay();
}

bool keyArray[256];

void specialKey(int key, int x, int y)
{
	keyArray[key] = true;
}

void specialKeyUp(int key, int x, int y)
{
	keyArray[key] = false;
}

static void idle(void)
{
    glutPostRedisplay();
}



/* Program entry point */

int main(int argc, char *argv[])
{
    glutInit(&argc, argv);
    glutInitWindowSize(640,480);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);

    glutCreateWindow("Julia");

    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    glutSpecialFunc(specialKey);
    glutSpecialUpFunc(specialKeyUp);
    //glutIdleFunc(idle);

    glClearColor(0.4f,0.4f,0.4f,1.0f);

	glGenTextures(1, texturaIDs);
	texturi[0].id = texturaIDs[0];
	glBindTexture(GL_TEXTURE_2D, texturaIDs[0]);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glEnable(GL_TEXTURE_2D);
    glutMainLoop();

    return EXIT_SUCCESS;
}
