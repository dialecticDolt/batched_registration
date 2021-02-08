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

#ifndef CONVOLUTIONSEPARABLE_COMMON_H
#define CONVOLUTIONSEPARABLE_COMMON_H

#include<npp.h>
#include<nppi_geometry_transforms.h>
#include<nppdefs.h>

#define KERNEL_RADIUS 2
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

void setConvolutionKernel(float *h_Kernel);

void convolutionRowsGPU(
    float *d_Dst,
    Npp8u *d_Src,
    int imageW,
    int imageH, 
    double* quad
);

void convolutionColumnsGPU(
    float *d_Dst,
    Npp8u *d_Src,
    int imageW,
    int imageH,
    double* quad
);


#endif
