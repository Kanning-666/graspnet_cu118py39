/*#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, int* ind_dev, cudaStream_t stream);*/

#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

void knn_device(float *ref_dev, int ref_width,
                float *query_dev, int query_width,
                int height, int k, float *dist_dev, int *ind_dev, cudaStream_t stream);