#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>

#include "kmeans.h"

int _debug;

static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -c centers     : file containing initial centers (default: filename)\n"
        "       -b             : input file is in binary format (default: no)\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -p nproc       : number of OpenMP threads (default: runtime)\n"
        "       -a             : use atomic updates (default: privatized reductions)\n"
        "       -o             : output timing results (default: no)\n"
        "       -q             : quiet mode\n"
        "       -d             : enable debug mode\n"
        "       -h             : print this help information\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

int main(int argc, char **argv) {
           int     opt;
    extern char   *optarg;
    extern int     optind;
           int     i, j, numThreads, isBinaryFile, is_output_timing, verbose;

           int     numClusters, numCoords, numObjs;
           int    *membership;
           char   *filename, *center_filename;
           float **objects;
           float **clusters;
           float   threshold;
           double  timing, io_timing, clustering_timing;
           int     use_atomic_updates;

    _debug             = 0;
    verbose            = 1;
    threshold          = 0.001f;
    numClusters        = 0;
    isBinaryFile       = 0;
    is_output_timing   = 0;
    filename           = NULL;
    center_filename    = NULL;
    numThreads         = 0;
    use_atomic_updates = 0;

    while ((opt = getopt(argc, argv, "p:i:c:n:t:abdohq")) != EOF) {
        switch (opt) {
            case 'p':
                numThreads = atoi(optarg);
                break;
            case 'i':
                filename = optarg;
                break;
            case 'c':
                center_filename = optarg;
                break;
            case 'b':
                isBinaryFile = 1;
                break;
            case 't':
                threshold = (float)atof(optarg);
                break;
            case 'n':
                numClusters = atoi(optarg);
                break;
            case 'o':
                is_output_timing = 1;
                break;
            case 'q':
                verbose = 0;
                break;
            case 'd':
                _debug = 1;
                break;
            case 'a':
                use_atomic_updates = 1;
                break;
            case 'h':
            default:
                usage(argv[0], threshold);
                break;
        }
    }
    if (center_filename == NULL)
        center_filename = filename;

    if (filename == NULL || numClusters <= 1) usage(argv[0], threshold);

    if (numThreads > 0) {
        omp_set_num_threads(numThreads);
    }

    if (is_output_timing) io_timing = wtime();

    printf("reading data points from file %s\n", filename);

    objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    if (objects == NULL) exit(1);

    if (numObjs < numClusters) {
        printf("Error: number of clusters must be larger than the number of data points to be clustered.\n");
        free(objects[0]);
        free(objects);
        return 1;
    }

    clusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc((size_t)numClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (i = 1; i < numClusters; i++)
        clusters[i] = clusters[i - 1] + numCoords;

    if (center_filename != filename) {
        printf("reading initial %d centers from file %s\n", numClusters, center_filename);
        read_n_objects(isBinaryFile, center_filename, numClusters, numCoords, clusters);
    } else {
        printf("selecting the first %d elements as initial centers\n", numClusters);
        for (i = 0; i < numClusters; i++)
            for (j = 0; j < numCoords; j++)
                clusters[i][j] = objects[i][j];
    }

    if (check_repeated_clusters(numClusters, numCoords, clusters) == 0) {
        printf("Error: some initial clusters are repeated. Please select distinct initial centers\n");
        free(objects[0]);
        free(objects);
        free(clusters[0]);
        free(clusters);
        return 1;
    }

    if (_debug) {
        printf("Sorted initial cluster centers:\n");
        for (i = 0; i < numClusters; i++) {
            printf("clusters[%d]=", i);
            for (j = 0; j < numCoords; j++)
                printf(" %6.2f", clusters[i][j]);
            printf("\n");
        }
    }

    if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }

    membership = (int*) malloc((size_t)numObjs * sizeof(int));
    assert(membership != NULL);

    if (!omp_kmeans(objects, numCoords, numObjs, numClusters, threshold,
                    membership, clusters, use_atomic_updates)) {
        fprintf(stderr, "Error: omp_kmeans failed\n");
        free(objects[0]);
        free(objects);
        free(membership);
        free(clusters[0]);
        free(clusters);
        return 1;
    }

    free(objects[0]);
    free(objects);

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
    }

    file_write(filename, numClusters, numObjs, numCoords, clusters,
               membership, verbose);

    free(membership);
    free(clusters[0]);
    free(clusters);

    if (is_output_timing) {
        io_timing += wtime() - timing;
        printf("\nPerforming **** Regular Kmeans (OpenMP version) ****\n");
        printf("Input file:     %s\n", filename);
        printf("numObjs       = %d\n", numObjs);
        printf("numCoords     = %d\n", numCoords);
        printf("numClusters   = %d\n", numClusters);
        printf("threshold     = %.4f\n", threshold);
        printf("Threads       = %d\n", (numThreads > 0) ? numThreads : omp_get_max_threads());

        printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", clustering_timing);
    }

    return 0;
}
