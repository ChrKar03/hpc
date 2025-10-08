#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

char horiz_operator[3][3] = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
char vert_operator[3][3] = {{1, 2, 1},
                            {0, 0, 0},
                            {-1, -2, -1}};

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);

unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];

int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]) {
	int dy, dx, res = 0;

	for (dy = -1; dy <= 1; dy++) {
		for (dx = -1; dx <= 1; dx++) {
			res += input[(posy + dy) * SIZE + posx + dx] * operator[dy + 1][dx + 1];
		}
	}
	return res;
}

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	int row, col;
	unsigned int p;
	int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (row = 1; row < SIZE-1; row++) {
		output[row*SIZE] = 0;
		output[row*SIZE + SIZE - 1] = 0;
	}

	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}

	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}

	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);

	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	const int inner_limit = SIZE - 1;
	for (row = 1; row < inner_limit; row++) {
		for (col = 1; col <= inner_limit - 4; col += 4) {
			p = pow(convolution2D(row, col, input, horiz_operator), 2) +
			    pow(convolution2D(row, col, input, vert_operator), 2);
			res = (int)sqrt(p);
			output[row * SIZE + col] = (unsigned char)(res > 255 ? 255 : res);

			p = pow(convolution2D(row, col + 1, input, horiz_operator), 2) +
			    pow(convolution2D(row, col + 1, input, vert_operator), 2);
			res = (int)sqrt(p);
			output[row * SIZE + col + 1] = (unsigned char)(res > 255 ? 255 : res);

			p = pow(convolution2D(row, col + 2, input, horiz_operator), 2) +
			    pow(convolution2D(row, col + 2, input, vert_operator), 2);
			res = (int)sqrt(p);
			output[row * SIZE + col + 2] = (unsigned char)(res > 255 ? 255 : res);

			p = pow(convolution2D(row, col + 3, input, horiz_operator), 2) +
			    pow(convolution2D(row, col + 3, input, vert_operator), 2);
			res = (int)sqrt(p);
			output[row * SIZE + col + 3] = (unsigned char)(res > 255 ? 255 : res);
		}
		for (; col < inner_limit; col++) {
			p = pow(convolution2D(row, col, input, horiz_operator), 2) +
			    pow(convolution2D(row, col, input, vert_operator), 2);
			res = (int)sqrt(p);
			output[row * SIZE + col] = (unsigned char)(res > 255 ? 255 : res);
		}
	}

	for (row = 1; row < inner_limit; row++) {
		for (col = 1; col < inner_limit; col++) {
			t = pow((output[row * SIZE + col] - golden[row * SIZE + col]), 2);
			PSNR += t;
		}
	}

	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);

	return PSNR;
}

int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}
