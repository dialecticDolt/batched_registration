/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include<stdio.h>
#include<iostream>
#include <assert.h>
#include <cooperative_groups.h>
#include "helper_cuda.h"
#include<nppdefs.h>

namespace cg = cooperative_groups;
#include "convolutionSeparable.h"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDA Assert Failed: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

////////////////////////////////////////////////////////////////////////////////
// Convolution kernel storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH] = { 1.0 / 12, -2.0 / 3, 0.0, 2.0 / 3, -1.0 / 12 };

void setConvolutionKernel(float *h_Kernel)
{
    //cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));

}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define   ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel(
    float *d_Dst,
    Npp8u *d_Src,
    int imageW,
    int imageH,
    int pitch,
    double* quad
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ Npp8u s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    cg::sync(cta);
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }


        const int current_loc = baseY* pitch + baseX + i * ROWS_BLOCKDIM_X;
        const int x = current_loc % pitch;
        const int y = current_loc / pitch;
        bool flag = 0;

        //Check boundaries
        double eval = (quad[2] - quad[0]) * (y - quad[1]) - (x - quad[0]) * (quad[3] - quad[1]);
        flag = ( eval >= 0 )  ? flag : 1;

        eval = (quad[4] - quad[2]) * (y - quad[3]) - (x - quad[2]) * (quad[5] - quad[3]);
        flag = (eval >= 0) ? flag : 1;

        eval = (quad[6] - quad[4]) * (y - quad[5]) - (x - quad[4]) * (quad[7] - quad[5]);
        flag = (eval >= 0) ? flag : 1;

        eval = (quad[0] - quad[6]) * (y - quad[7]) - (x - quad[6]) * (quad[1] - quad[7]);
        flag = (eval >= 0) ? flag : 1;
        //flag = 0;

        d_Dst[i * ROWS_BLOCKDIM_X] = (flag) ? 0 : sum;
        //d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

void convolutionRowsGPU(
    float *d_Dst,
    Npp8u *d_Src,
    int imageW,
    int imageH,
    double* quad
)
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW, 
        quad
    );
    getLastCudaError("convolutionRowsKernel() execution failed\n");
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel(
    float *d_Dst,
    Npp8u *d_Src,
    int imageW,
    int imageH,
    int pitch, 
    double* quad
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (float) d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? (float) d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0.0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? (float) d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0.0;
    }

    //Compute and store results
    cg::sync(cta);
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        const int current_loc = baseY* pitch + baseX + i * COLUMNS_BLOCKDIM_Y * pitch;
        const int x = current_loc % pitch;
        const int y = current_loc / pitch;
        bool flag = 0;

        //Check boundaries
        double eval = (quad[2] - quad[0]) * (y - quad[1]) - (x - quad[0]) * (quad[3] - quad[1]);
        flag = ( eval >= 0 )  ? flag : 1;

        eval = (quad[4] - quad[2]) * (y - quad[3]) - (x - quad[2]) * (quad[5] - quad[3]);
        flag = (eval >= 0) ? flag : 1;

        eval = (quad[6] - quad[4]) * (y - quad[5]) - (x - quad[4]) * (quad[7] - quad[5]);
        flag = (eval >= 0) ? flag : 1;

        eval = (quad[0] - quad[6]) * (y - quad[7]) - (x - quad[6]) * (quad[1] - quad[7]);
        flag = (eval >= 0) ? flag : 1;
        //flag = 0;
        //if (flag) {
        //    printf("(q: %f %f %f %f) :: (%i, %i) : % f \n", quad[0], quad[1], quad[6], quad[7], x, y, eval);
        //}

        //d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = (flag) ? 0 : sum;
    }
}

void convolutionColumnsGPU(
    float *d_Dst,
    Npp8u *d_Src,
    int imageW,
    int imageH, 
    double* quad
)
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

    //std::cout << blocks.x << ":" << threads.x << std::endl;

    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        d_Src,
        imageW,
        imageH,
        imageW, 
        quad
    );
    getLastCudaError("convolutionColumnsKernel() execution failed\n");
}

