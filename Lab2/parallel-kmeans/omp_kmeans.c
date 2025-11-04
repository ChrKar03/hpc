#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "kmeans.h"

__inline static float euclid_dist_2(int numdims, float *coord1, float *coord2) {
    float ans = 0.0f;

    for (int i = 0; i < numdims; i++) {
        float diff = coord1[i] - coord2[i];
        ans += diff * diff;
    }

    return ans;
}

__inline static int find_nearest_cluster(int numClusters,
                                         int numCoords,
                                         float *object,
                                         float **clusters) {
    int   index    = 0;
    float min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (int i = 1; i < numClusters; i++) {
        float dist = euclid_dist_2(numCoords, object, clusters[i]);
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }

    return index;
}

// return an array of cluster centers of size [numClusters][numCoords]
int omp_kmeans(float **objects,
               int     numCoords,
               int     numObjs,
               int     numClusters,
               float   threshold,
               int    *membership,
               float **clusters) {
    if (objects == NULL || membership == NULL || clusters == NULL) return 0;

    /* Global accumulators for the new cluster sums and sizes */
    int *newClusterSize = (int*) calloc(numClusters, sizeof(int));
    if (newClusterSize == NULL) return 0;

    /* newClusters is allocated as a contiguous block for better locality:
     * newClusters points to an array of float*; newClusters[0] points to
     * numClusters * numCoords floats, and newClusters[i] = newClusters[0] + i*numCoords.
     */
    float **newClusters = (float**) malloc(numClusters * sizeof(float*));
    if (newClusters == NULL) {
        free(newClusterSize);
        return 0;
    }
    newClusters[0] = (float*) calloc((size_t)numClusters * numCoords, sizeof(float));
    if (newClusters[0] == NULL) {
        free(newClusters);
        free(newClusterSize);
        return 0;
    }
    for (int i = 1; i < numClusters; i++)
        newClusters[i] = newClusters[i - 1] + numCoords;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numObjs; i++)
        membership[i] = -1;

    int loop = 0, maxThreads = omp_get_max_threads(), nthreads   = maxThreads;
    double delta;

    /* Per-thread accumulators:
     * - partialClusterSize: for each thread, numClusters ints
     * - partialClusters: for each thread, numClusters * numCoords floats
     *
     * - Using maxThreads guarantees enough space even if fewer threads are used.
     */
    int   *partialClusterSize = (int*)   calloc((size_t)maxThreads * numClusters, sizeof(int));
    float *partialClusters    = (float*) calloc((size_t)maxThreads * numClusters * numCoords,
                                                sizeof(float));
    if (partialClusterSize == NULL || partialClusters == NULL) {
        free(partialClusterSize);
        free(partialClusters);
        return 0;
    }

    /* Main k-means loop: assign points, accumulate partial sums, reduce, recompute centers */
    do {
        delta = 0.0;

        /* Parallel region: each thread uses its own local accumulator slices
         * and contributes to a shared reduction variable delta.
         */
        #pragma omp parallel shared(nthreads, partialClusterSize, partialClusters) reduction(+:delta)
        {
            int tid = omp_get_thread_num();

            #pragma omp single
            {
                nthreads = omp_get_num_threads();
            }

            /* compute pointers to this thread's local accumulators */
            int   *localClusterSize = partialClusterSize + tid * numClusters;
            float *localClusters    = partialClusters + ((size_t)tid * numClusters * numCoords);

            memset(localClusterSize, 0, numClusters * sizeof(int));
            memset(localClusters, 0, (size_t)numClusters * numCoords * sizeof(float));

            /* distribute objects across threads; each thread updates its local accumulators */
            #pragma omp for schedule(static)
            for (int i = 0; i < numObjs; i++) {
                int index = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);

                /* count how many objects changed membership (for convergence check) */
                if (membership[i] != index) delta += 1.0;
                membership[i] = index;

                /* update local accumulators for the assigned cluster */
                localClusterSize[index]++;

                float *clusterAccum = localClusters + index * numCoords;
                for (int j = 0; j < numCoords; j++)
                    clusterAccum[j] += objects[i][j];
            }
        }

        memset(newClusterSize, 0, numClusters * sizeof(int));
        memset(newClusters[0], 0, (size_t)numClusters * numCoords * sizeof(float));

        /* Reduce per-thread accumulators into global accumulators.
         * Parallelizing over clusters is natural: each iteration aggregates the
         * contributions for one cluster across all threads.
         */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numClusters; i++) {
            int   clusterCount = 0;
            float *dest        = newClusters[i];

            for (int tid = 0; tid < nthreads; tid++) {
                int   *localClusterSize = partialClusterSize + tid * numClusters;
                float *localClusters    = partialClusters + ((size_t)tid * numClusters * numCoords);

                clusterCount += localClusterSize[i];

                float *src = localClusters + i * numCoords;
                for (int j = 0; j < numCoords; j++)
                    dest[j] += src[j];
            }

            newClusterSize[i] = clusterCount;
        }

        /* Recompute cluster centers from summed coordinates and sizes.
         * Also clear newClusters entries for the next iteration (we reuse the buffer).
         */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numClusters; i++) {
            if (newClusterSize[i] > 0) {
                float inv = 1.0f / newClusterSize[i];
                for (int j = 0; j < numCoords; j++) {
                    clusters[i][j] = newClusters[i][j] * inv;
                    newClusters[i][j] = 0.0f;
                }
            } else
                for (int j = 0; j < numCoords; j++)
                    newClusters[i][j] = 0.0f;

            newClusterSize[i] = 0;
        }

        /* fraction of objects that changed membership this iteration */
        delta /= numObjs;
    } while (delta > threshold && loop++ < 500); 

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    free(partialClusterSize);
    free(partialClusters);

    return 1;
}
