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

static const char horiz_operator[3][3] = {{-1, 0, 1},
                                          {-2, 0, 2},
                                          {-1, 0, 1}};
static const char vert_operator[3][3] = {{1, 2, 1},
                                         {0, 0, 0},
                                         {-1, -2, -1}};

double sobel(unsigned char * restrict input, unsigned char * restrict output, unsigned char * restrict golden);

unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];

#define PROCESS_PIXEL(COL_INDEX) do { \
		const int c = (COL_INDEX); \
		const int cm1 = c - 1; \
		const int cp1 = c + 1; \
		const int up_l = upper[cm1]; \
		const int up_m = upper[c]; \
		const int up_r = upper[cp1]; \
		const int mid_l = middle[cm1]; \
		const int mid_m = middle[c]; \
		const int mid_r = middle[cp1]; \
		const int low_l = lower[cm1]; \
		const int low_m = lower[c]; \
		const int low_r = lower[cp1]; \
		int gx = \
			up_l * horiz_operator[0][0] + \
			up_m * horiz_operator[0][1] + \
			up_r * horiz_operator[0][2] + \
			mid_l * horiz_operator[1][0] + \
			mid_m * horiz_operator[1][1] + \
			mid_r * horiz_operator[1][2] + \
			low_l * horiz_operator[2][0] + \
			low_m * horiz_operator[2][1] + \
			low_r * horiz_operator[2][2]; \
		const int gy = \
			up_l * vert_operator[0][0] + \
			up_m * vert_operator[0][1] + \
			up_r * vert_operator[0][2] + \
			mid_l * vert_operator[1][0] + \
			mid_m * vert_operator[1][1] + \
			mid_r * vert_operator[1][2] + \
			low_l * vert_operator[2][0] + \
			low_m * vert_operator[2][1] + \
			low_r * vert_operator[2][2]; \
		const unsigned int mag_sq = (unsigned int)(gx * gx) + (unsigned int)(gy * gy); \
		res = (int)sqrt((double)mag_sq); \
		const unsigned char pixel_value = (unsigned char)(res > 255 ? 255 : res); \
		out_row[c] = pixel_value; \
		diff = out_row[c] - gold_row[c]; \
		PSNR += diff * diff; \
	} while (0)

double sobel(unsigned char * restrict input, unsigned char * restrict output, unsigned char * restrict golden)
{
	double PSNR = 0;
	int row, col;
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
		const int row_base = row * SIZE;
		const unsigned char * restrict upper = input + row_base - SIZE;
		const unsigned char * restrict middle = input + row_base;
		const unsigned char * restrict lower = input + row_base + SIZE;
		unsigned char * restrict out_row = output + row_base;
		const unsigned char * restrict gold_row = golden + row_base;

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
