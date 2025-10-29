#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "kmeans.h"

/* square of Euclid distance between two multi-dimensional points */
__inline static float euclid_dist_2(int numdims, float *coord1, float *coord2) {
    float ans = 0.0f;

    for (int i = 0; i < numdims; i++) {
        float diff = coord1[i] - coord2[i];
        ans += diff * diff;
    }

    return ans;
}

/* find the cluster id that has minimum distance to the given object */
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

/* return an array of cluster centers of size [numClusters][numCoords]       */
int omp_kmeans(float **objects,
               int     numCoords,
               int     numObjs,
               int     numClusters,
               float   threshold,
               int    *membership,
               float **clusters,
               int     use_atomic_updates) {
    if (objects == NULL || membership == NULL || clusters == NULL) return 0;

    int *newClusterSize = (int*) calloc(numClusters, sizeof(int));
    if (newClusterSize == NULL) return 0;

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
    for (int i = 1; i < numClusters; i++) {
        newClusters[i] = newClusters[i - 1] + numCoords;
    }

    /* initialize membership[] */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < numObjs; i++) {
        membership[i] = -1;
    }

    int    loop = 0;
    double delta;

    if (use_atomic_updates) {
        do {
            delta = 0.0;

            memset(newClusterSize, 0, numClusters * sizeof(int));
            memset(newClusters[0], 0, numClusters * numCoords * sizeof(float));

            #pragma omp parallel for reduction(+:delta) schedule(static)
            for (int i = 0; i < numObjs; i++) {
                int index = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);

                if (membership[i] != index) delta += 1.0;
                membership[i] = index;

                #pragma omp atomic
                newClusterSize[index]++;

                for (int j = 0; j < numCoords; j++) {
                    #pragma omp atomic
                    newClusters[index][j] += objects[i][j];
                }
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < numClusters; i++) {
                if (newClusterSize[i] > 0) {
                    float inv = 1.0f / newClusterSize[i];
                    for (int j = 0; j < numCoords; j++) {
                        clusters[i][j] = newClusters[i][j] * inv;
                        newClusters[i][j] = 0.0f;
                    }
                } else {
                    for (int j = 0; j < numCoords; j++) {
                        newClusters[i][j] = 0.0f;
                    }
                }
                newClusterSize[i] = 0;
            }

            delta /= numObjs;
        } while (delta > threshold && loop++ < 500);

        free(newClusters[0]);
        free(newClusters);
        free(newClusterSize);

        return 1;
    }

    /* default path: per-thread privatization and reduction */
    int maxThreads = omp_get_max_threads();
    int nthreads   = maxThreads;

    int   *partialClusterSize = (int*)   calloc((size_t)maxThreads * numClusters, sizeof(int));
    float *partialClusters    = (float*) calloc((size_t)maxThreads * numClusters * numCoords,
                                                sizeof(float));
    if (partialClusterSize == NULL || partialClusters == NULL) {
        free(partialClusterSize);
        free(partialClusters);
        return 0;
    }

    do {
        delta = 0.0;

        #pragma omp parallel shared(nthreads, partialClusterSize, partialClusters) \
                             reduction(+:delta)
        {
            int tid = omp_get_thread_num();

            #pragma omp single
            {
                nthreads = omp_get_num_threads();
            }

            int   *localClusterSize = partialClusterSize + tid * numClusters;
            float *localClusters    = partialClusters + ((size_t)tid * numClusters * numCoords);

            memset(localClusterSize, 0, numClusters * sizeof(int));
            memset(localClusters, 0, (size_t)numClusters * numCoords * sizeof(float));

            #pragma omp for schedule(static)
            for (int i = 0; i < numObjs; i++) {
                int index = find_nearest_cluster(numClusters, numCoords, objects[i], clusters);

                if (membership[i] != index) delta += 1.0;
                membership[i] = index;

                localClusterSize[index]++;

                float *clusterAccum = localClusters + index * numCoords;
                for (int j = 0; j < numCoords; j++) {
                    clusterAccum[j] += objects[i][j];
                }
            }
        } /* end parallel region */

        memset(newClusterSize, 0, numClusters * sizeof(int));
        memset(newClusters[0], 0, (size_t)numClusters * numCoords * sizeof(float));

        /* merge partial sums per cluster in parallel before normalization */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numClusters; i++) {
            int   clusterCount = 0;
            float *dest        = newClusters[i];

            for (int tid = 0; tid < nthreads; tid++) {
                int   *localClusterSize = partialClusterSize + tid * numClusters;
                float *localClusters    = partialClusters + ((size_t)tid * numClusters * numCoords);

                clusterCount += localClusterSize[i];

                float *src = localClusters + i * numCoords;
                for (int j = 0; j < numCoords; j++) {
                    dest[j] += src[j];
                }
            }

            newClusterSize[i] = clusterCount;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numClusters; i++) {
            if (newClusterSize[i] > 0) {
                float inv = 1.0f / newClusterSize[i];
                for (int j = 0; j < numCoords; j++) {
                    clusters[i][j] = newClusters[i][j] * inv;
                    newClusters[i][j] = 0.0f;
                }
            } else {
                for (int j = 0; j < numCoords; j++) {
                    newClusters[i][j] = 0.0f;
                }
            }
            newClusterSize[i] = 0;
        }

        delta /= numObjs;
    } while (delta > threshold && loop++ < 500);

    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);
    free(partialClusterSize);
    free(partialClusters);

    return 1;
}
