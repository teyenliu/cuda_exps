#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define ITER_NUM 100

__global__ void iKernel(float *src, float *dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}

__global__ void reduceKernel(float *src, float *dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    src[idx] += dst[idx];
}

inline bool isCapableP2P(int ngpus) {
    cudaDeviceProp prop[ngpus];
    int iCount = 0;

    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaGetDeviceProperties(&prop[i], i));

        if (prop[i].major >= 2)
            iCount++;

        printf("> GPU%d: %s %s capable of Peer-to-Peer access\n", i,
                prop[i].name, (prop[i].major >= 2 ? "is" : "not"));
    }

    if (iCount != ngpus) {
        printf("> no enough device to run this application\n");
    }

    return (iCount == ngpus);
}

inline bool isUnifiedAddressingSupported(int ngpus) {
    cudaDeviceProp prop[ngpus];
    bool iuva;
    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaGetDeviceProperties(&prop[i], i));
        iuva &= prop[i].unifiedAddressing;
        printf("> GPU%i: %s %s unified addressing\n", i, prop[0].name,
                (prop[i].unifiedAddressing ? "supports" : "does not support"));
    }
    return iuva;
}

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later)).
 */
inline void enableP2P(int ngpus) {
    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));

        for (int j = 0; j < ngpus; j++) {
            if (i == j)
                continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) {
                CHECK(cudaDeviceEnablePeerAccess(j, 0));
                printf("> GPU%d enabled direct access to GPU%d\n", i, j);
            } else {
                printf("(%d, %d)\n", i, j);
            }
        }
    }
}

inline void disableP2P(int ngpus) {
    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));

        for (int j = 0; j < ngpus; j++) {
            if (i == j)
                continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) {
                CHECK(cudaDeviceDisablePeerAccess(j));
                printf("> GPU%d disabled direct access to GPU%d\n", i, j);
            }
        }
    }
}

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        ip[i] = (float) rand() / (float) RAND_MAX;
    }
}

void test_method1() {

}

int main(int argc, char **argv) {
    int ngpus;

    // check device count
    CHECK(cudaGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    // check p2p capability
    //isCapableP2P(ngpus);
    //isUnifiedAddressingSupported(ngpus);

    // get ngpus from command line
    if (argc > 1) {
        if (atoi(argv[1]) > ngpus) {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            return 1;
        } else if (atoi(argv[1]) < 1) {
            fprintf(stderr, "Invalid number of GPUs specified: %d is less  "
                    "than 1 in this platform (%d)\n", atoi(argv[1]), ngpus);
            return 1;
        }
        ngpus = atoi(argv[1]);
    }

    if (ngpus % 2) {
        fprintf(stderr, "The number of GPUs must be odd one\n");
        return 1;
    }
    enableP2P(ngpus);

    // Allocate buffers
    int iSize = 1024 * 1024 * 16;
    const size_t iBytes = iSize * sizeof(float);
    printf("\nAllocating buffers (%iMB on each GPU and CPU Host)...\n",
            int(iBytes / 1024 / 1024));

    float **d_src = (float **) malloc(sizeof(float) * ngpus);
    float **d_rcv = (float **) malloc(sizeof(float) * ngpus);
    float **h_src = (float **) malloc(sizeof(float) * ngpus);

    // We have n phases
    cudaStream_t *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t) * ngpus);

    // Create CUDA event handles
    cudaEvent_t start, stop;
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc(&d_src[i], iBytes));
        CHECK(cudaMalloc(&d_rcv[i], iBytes));
        CHECK(cudaStreamCreate(&stream[i]));
        CHECK(cudaMallocHost((void **) &h_src[i], iBytes));
    }


    for (int i = 0; i < ngpus; i++) {
        initialData(h_src[i], iSize);
    }

    const dim3 block(512);
    const dim3 grid(iSize / block.x);

    /*** Method 1 ***/
    // Asynchronous GPUmem copy by pairs
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(start, 0));

    for (int i = 0; i < ITER_NUM; i++)
    {
        // Phase 1
        for (int dev = 0; dev < ngpus; dev++)
        {
            if (!(dev % 2)) // even number, for instance: 0, 2, 4,...
            {
                CHECK(cudaMemcpyPeerAsync(d_src[dev+1], dev+1, d_rcv[dev], dev, iBytes, stream[dev]));
            }
        }
        // Do stream sync
        for (int dev = 0; dev < ngpus; dev++)
        {
            if (!(dev % 2)) // even number, for instance: 0, 2, 4,...
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }
        // Do kernel function
        for (int dev = 0; dev < ngpus; dev++) {
            if (!(dev % 2)) // even number, for instance: 0, 2, 4,...
            {
                CHECK(cudaSetDevice(dev+1));
                reduceKernel<<<grid, block, iBytes, stream[0]>>>(d_src[dev+1], d_rcv[dev+1]);
            }
        }
        // Do stream sync
        for (int dev = 0; dev < ngpus; dev++) {
            if (!(dev % 2)) // even number, for instance: 0, 2, 4,...
            {
                CHECK(cudaStreamSynchronize(stream[dev]));
            }
        }


        // Phase 2
        for (int dev = 0; dev < ngpus; dev++)
        {
            if ((dev % 2) && (dev != (ngpus - 1))) // even number, for instance: 1, 3, 5,...
            {
                CHECK(cudaMemcpyPeerAsync(d_src[dev], dev+2, d_rcv[dev + 2], dev, iBytes, stream[dev]));
            } /*else if ((dev % 2) && (dev == (ngpus - 1))) {
                CHECK(cudaMemcpyPeerAsync(d_src[dev], 1, d_rcv[0], 0, iBytes, stream[dev]));
            }*/
        }
        // Do stream sync
        for (int dev = 0; dev < ngpus; dev++)
        {
            if ((dev % 2) && (dev != (ngpus - 1))) // even number, for instance: 1, 3, 5,...
            {
                CHECK(cudaMemcpyPeerAsync(d_src[dev], dev+2, d_rcv[dev + 2], dev, iBytes, stream[dev]));
            }
        }
        // Do kernel function
        for (int dev = 0; dev < ngpus; dev++)
        {
            if ((dev % 2) && (dev != (ngpus - 1))) // even number, for instance: 1, 3, 5,...
            {
                CHECK(cudaSetDevice(dev+2));
                reduceKernel<<<grid, block, iBytes, stream[dev+2]>>>(d_src[dev+2], d_rcv[dev+2]);
            }
        }
        // Do stream sync
        for (int dev = 0; dev < ngpus; dev++)
        {
            if ((dev % 2) && (dev != (ngpus - 1))) // even number, for instance: 1, 3, 5,...
            {
                CHECK(cudaStreamSynchronize(stream[dev+2]));
            }
        }
    }

    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    float elapsed_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

    elapsed_time_ms /= (float)ITER_NUM;
    printf("2 phases cudaMemcpyPeerAsync time per cycle:\t %8.2fms\n", elapsed_time_ms);
    printf("performance: %8.2f GB/s\n", (float) iBytes * 4.0 / (elapsed_time_ms * 1e6f));

    /*** Method 2 ***/
    // Asynchronous GPUmem copy
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(start, 0));

    for (int i = 0; i < ITER_NUM; i++) {
        for (int dev = 0; dev < ngpus; dev++) {
            // Do ring async memory copy
            if ((dev != (ngpus - 1))) {
                CHECK(cudaMemcpyPeerAsync(d_src[dev], dev+1, d_rcv[dev + 1], dev, iBytes, stream[dev]));
            } else if ((dev == (ngpus - 1))) {
                CHECK(cudaMemcpyPeerAsync(d_src[dev], 0, d_rcv[0], dev, iBytes, stream[dev]));
            }
            CHECK(cudaStreamSynchronize(stream[dev]));
            // Do stream sync
            if ((dev != (ngpus - 1))) {
                CHECK(cudaSetDevice(dev+1));
                reduceKernel<<<grid, block, iBytes, stream[dev+1]>>>(d_src[dev+1], d_rcv[dev+1]);
                CHECK(cudaStreamSynchronize(stream[dev]));
            } else if ((dev == (ngpus - 1))) {
                CHECK(cudaSetDevice(0));
                reduceKernel<<<grid, block, iBytes, stream[0]>>>(d_src[0], d_rcv[0]);
                CHECK(cudaStreamSynchronize(stream[0]));
            }
        }
    }

    CHECK(cudaSetDevice(0));
    CHECK(cudaEventRecord(stop, 0));
    CHECK(cudaEventSynchronize(stop));

    elapsed_time_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_time_ms, start, stop));

    elapsed_time_ms /= (float)ITER_NUM;
    printf("Ring cudaMemcpyPeerAsync time per cycle:\t %8.2fms\n", elapsed_time_ms);
    printf("performance: %8.2f GB/s\n", (float) iBytes * 4.0 / (elapsed_time_ms * 1e6f));

    disableP2P(ngpus);

    // free
    CHECK(cudaSetDevice(0));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    for (int i = 0; i < ngpus; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaFree(h_src[i]));
        CHECK(cudaFree(d_src[i]));
        CHECK(cudaFree(d_rcv[i]));
        CHECK(cudaStreamDestroy(stream[i]));
        CHECK(cudaDeviceReset());
    }

    exit (EXIT_SUCCESS);
}
