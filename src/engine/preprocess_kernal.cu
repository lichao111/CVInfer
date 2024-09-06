#include <cuda.h>
#include <cuda_runtime.h>

#include "preprocess_kernal.cuh"
#include "timer_gpu.cuh"

using namespace cv_infer;
namespace CUDAKernal
{
__global__ void ConverHWC2CHWMeanStdKernel(const unsigned char* src, int h, int w, int c, const float* mean,
                                           const float* scale, float* dst)
{
    // from copilot :
    /*
    在 CUDA 编程中，计算线程索引的方式有多种，选择哪种方式取决于具体的应用场景和需求。以下是对两种索引计算方式的解释：
    方式一：直接计算三维索引
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;

    方式二：计算一维索引并转换为三维索引
        int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
        int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) +
(threadIdx.y * blockDim.x) + threadIdx.x;
        int x = threadId % width;
        int y = (threadId / width) % height;
        int z = threadId / (width * height);
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < c)
    {
        dst[z * h * w + y * w + x] = ((float)(src[y * w * c + x * c + z] - mean[z])) / scale[z];
    }
}

__global__ void ConverHWC2CHWNormKernel(const unsigned char* src, int h, int w, int c, float* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < c)
    {
        dst[z * h * w + y * w + x] = ((float)(src[y * w * c + x * c + z])) / 255.0;
    }
}

void ConverHWC2CHWMeanStd(const unsigned char* src, int h, int w, int c, const float* mean, const float* scale,
                          float* dst)
{
    // int jobs = h * w;
    // auto grid = grid_dims(jobs);
    // auto block = block_dims(jobs);
    dim3 block(16, 16, 1);
    auto grid_x = (w + block.x - 1) / block.x;
    auto grid_y = (h + block.y - 1) / block.y;
    auto grid_z = (c + block.z - 1) / block.z;
    dim3 grid(grid_x, grid_y, grid_z);

    // stream
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    TimerGPU timer;
    timer.start(stream);

    // copy src to device
    unsigned char* src_d;
    cudaMallocAsync(&src_d, h * w * c * sizeof(unsigned char), stream);
    cudaMemcpyAsync(src_d, src, h * w * c * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);
    // copy mean to device
    float* mean_d;
    cudaMallocAsync(&mean_d, 3 * sizeof(float), stream);
    cudaMemcpyAsync(mean_d, mean, c * sizeof(float), cudaMemcpyHostToDevice, stream);
    // copy scale to device
    float* scale_d;
    cudaMallocAsync(&scale_d, 3 * sizeof(float), stream);
    cudaMemcpyAsync(scale_d, scale, c * sizeof(float), cudaMemcpyHostToDevice, stream);

    ConverHWC2CHWMeanStdKernel<<<grid, block, 0, stream>>>(src_d, h, w, c, mean_d, scale_d, dst);

    // realize async
    cudaStreamSynchronize(stream);
    timer.stop("ConverHWC2CHWMeanStd", true);

    cudaFree(src_d);
    cudaFree(mean_d);
    cudaFree(scale_d);
    cudaStreamDestroy(stream);
}

void ConverHWC2CHWNorm(const unsigned char* src, int h, int w, int c, float* dst)
{
    dim3 block(16, 16, 1);
    auto grid_x = (w + block.x - 1) / block.x;
    auto grid_y = (h + block.y - 1) / block.y;
    auto grid_z = (c + block.z - 1) / block.z;
    dim3 grid(grid_x, grid_y, grid_z);

    // stream
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    TimerGPU timer;
    timer.start(stream);

    // copy src to device
    unsigned char* src_d;
    cudaMallocAsync(&src_d, h * w * c * sizeof(unsigned char), stream);
    cudaMemcpyAsync(src_d, src, h * w * c * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);

    ConverHWC2CHWNormKernel<<<grid, block, 0, stream>>>(src_d, h, w, c, dst);

    // realize async
    cudaStreamSynchronize(stream);
    timer.stop("ConverHWC2CHWNorm", true);

    cudaFree(src_d);
    cudaStreamDestroy(stream);
}

}  // namespace CUDAKernal