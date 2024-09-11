#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

namespace CUDAKernal
{

#define GPU_BLOCK_THREADS 512

// dim3 grid_dims(int numJobs)
// {
//     int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
//     return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
// }

// dim3 block_dims(int numJobs) { return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS; }

void ConverHWC2CHWMeanStd(const unsigned char* src, int h, int w, int c, const float* mean, const float* scale,
                          float* dst);
void ConverHWC2CHWAlpahNormResizeKeepRatio(const unsigned char* src, int src_h, int src_w, int src_c, int dst_h,
                                           int dst_w, int dst_c, float alpha, float beta, float fill_value, int stride,
                                           float* dst);
}  // namespace CUDAKernal