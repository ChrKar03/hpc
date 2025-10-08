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

unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];

#define PROCESS_PIXEL(COL_INDEX) do { \
		const int c = (COL_INDEX); \
		int gx = \
			input[(row - 1) * SIZE + (c - 1)] * horiz_operator[0][0] + \
			input[(row - 1) * SIZE + c] * horiz_operator[0][1] + \
			input[(row - 1) * SIZE + (c + 1)] * horiz_operator[0][2] + \
			input[row * SIZE + (c - 1)] * horiz_operator[1][0] + \
			input[row * SIZE + c] * horiz_operator[1][1] + \
			input[row * SIZE + (c + 1)] * horiz_operator[1][2] + \
			input[(row + 1) * SIZE + (c - 1)] * horiz_operator[2][0] + \
			input[(row + 1) * SIZE + c] * horiz_operator[2][1] + \
			input[(row + 1) * SIZE + (c + 1)] * horiz_operator[2][2]; \
		int gy = \
			input[(row - 1) * SIZE + (c - 1)] * vert_operator[0][0] + \
			input[(row - 1) * SIZE + c] * vert_operator[0][1] + \
			input[(row - 1) * SIZE + (c + 1)] * vert_operator[0][2] + \
			input[row * SIZE + (c - 1)] * vert_operator[1][0] + \
			input[row * SIZE + c] * vert_operator[1][1] + \
			input[row * SIZE + (c + 1)] * vert_operator[1][2] + \
			input[(row + 1) * SIZE + (c - 1)] * vert_operator[2][0] + \
			input[(row + 1) * SIZE + c] * vert_operator[2][1] + \
			input[(row + 1) * SIZE + (c + 1)] * vert_operator[2][2]; \
		p = pow(gx, 2) + pow(gy, 2); \
		res = (int)sqrt(p); \
		const int out_index = row * SIZE + c; \
		output[out_index] = (unsigned char)(res > 255 ? 255 : res); \
		diff = output[out_index] - golden[out_index]; \
		PSNR += diff * diff; \
	} while (0)

double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0;
	int row, col;
	unsigned int p;
	int res;
	double diff;
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
			PROCESS_PIXEL(col);
			PROCESS_PIXEL(col + 1);
			PROCESS_PIXEL(col + 2);
			PROCESS_PIXEL(col + 3);
		}
		for (; col < inner_limit; col++) {
			PROCESS_PIXEL(col);
		}
	}
#undef PROCESS_PIXEL

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
