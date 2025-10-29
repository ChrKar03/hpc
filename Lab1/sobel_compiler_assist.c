// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	    4096
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"

/* The horizontal and vertical operators to be used in the sobel filter */
const char horiz_operator[3][3] = {{-1, 0, 1}, 
                                   {-2, 0, 2}, 
                                   {-1, 0, 1}};
const char vert_operator[3][3] = {{1, 2, 1}, 
                                  {0, 0, 0}, 
                                  {-1, -2, -1}};

double sobel(unsigned char *restrict input, unsigned char *restrict output, unsigned char *restrict golden);
int convolution2D(int posy, int posx, const unsigned char *restrict input, char operator[][3]);

/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */
int convolution2D(int posy, int posx, const unsigned char *restrict input, char operator[][3]) {
	register int i, j, res;
  
	res = 0;
	for (i = -1; i <= 1; i++) {
		for (j = -1; j <= 1; j++) {
			res += input[(posy + i)*SIZE + posx + j] * operator[i+1][j+1];
		}
	}
	return(res);
}

#define INLINE_CONVOLUTION2D(operator, res) \
	u0 = input[upper  - 1]; \
	u1 = input[upper     ]; \
	u2 = input[upper  + 1]; \
	m0 = input[middle - 1]; \
	m1 = input[middle    ]; \
	m2 = input[middle + 1]; \
	l0 = input[lower  - 1]; \
	l1 = input[lower     ]; \
	l2 = input[lower  + 1]; \
	temp0 = u0 * operator[0][0]; \
	temp1 = u1 * operator[0][1]; \
	temp2 = u2 * operator[0][2]; \
	temp3 = m0 * operator[1][0]; \
	temp4 = m1 * operator[1][1]; \
	temp5 = m2 * operator[1][2]; \
	temp6 = l0 * operator[2][0]; \
	temp7 = l1 * operator[2][1]; \
	temp8 = l2 * operator[2][2]; \
	res = temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7 + temp8;

#define UNROLL_FACTOR 8

#define UNROLLED(off) \
	upper = upper_row_base + j + off; \
	middle = row_base + j + off; \
	lower = lower_row_base + j + off; \
	INLINE_CONVOLUTION2D(horiz_operator, ch); \
	INLINE_CONVOLUTION2D(vert_operator, cv); \
	p1 = ch * ch; \
	p2 = cv * cv; \
	p = p1 + p2; \
	res = (int)sqrt(p); \
	output[middle] = (res > 255) ? 255 : (unsigned char)res; \
	diff = (int)(output[middle] - golden[middle]); \
	t = diff * diff; \
	PSNR += t;

/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *restrict input, unsigned char *restrict output, unsigned char *restrict golden)
{
	double PSNR = 0, t;
	register int i, j;
	register unsigned int p;
	register int res;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	register int ch, cv; // convolution horizontal and vertical
	register int row_base, upper_row_base, lower_row_base; // base indices for rows
	register int upper, middle, lower; // indices for convolution
	register int u0, u1, u2, m0, m1, m2, l0, l1, l2; // pixels for inlined convolution
	register int temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8; // temps for inlined convolution
	register int p1, p2; // squared convolutions
	register int diff;    // difference between computed and golden pixel

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
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
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);

	row_base = SIZE;
	
    /* For each pixel of the output image */
    for (i = 1; i < SIZE - 1; i++) {
		upper_row_base = row_base - SIZE;
		lower_row_base = row_base + SIZE;

        for (j = 1; j < SIZE - 1 - UNROLL_FACTOR + 1; j += UNROLL_FACTOR) {
			UNROLLED(0);
			UNROLLED(1);
			UNROLLED(2);
			UNROLLED(3);
			UNROLLED(4);
			UNROLLED(5);
			UNROLLED(6);
			UNROLLED(7);
			// UNROLLED(8);
			// UNROLLED(9);
			// UNROLLED(10);
			// UNROLLED(11);
			// UNROLLED(12);
			// UNROLLED(13);
			// UNROLLED(14);
			// UNROLLED(15);
        }

        /* Handle remaining pixels if SIZE-2 is not divisible by UNROLL_FACTOR */
        for (; j < SIZE - 1; j++) {
            UNROLLED(0);
        }

		row_base += SIZE;
    }
  
	PSNR /= (double)(SIZE*SIZE);
	PSNR = 10*log10(65536/PSNR);

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	const double PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

