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

    /*  HWC->CHW得转换原理
        HWC: BGRBGRBGRBGR......BGRBGRBGR  (行优先)
        CHW: BBB...BBBGGG...GGGRRR...RRR
        source_index = (x, y, z)
        则source_index对应得是第 y*w*c + x*c + z 个元素
        其转换后得目标位置为 z*h*w + y*w + x
    */

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < w && y < h && z < c)
    {
        dst[z * h * w + y * w + x] = ((float)(src[y * w * c + x * c + z] - mean[z])) / scale[z];
    }
}

// 函数功能：
// 1. 将HWC格式的图像转换为CHW格式的图像
// 2. 进行alpha,beta归一化
// 3. 进行保持比例缩放, 使用固定值padding
__global__ void ConverHWC2CHWAlpahNormResizeKeepRatioKernel(const unsigned char* src, int src_h, int src_w, int src_c,
                                                            int dst_h, int dst_w, int dst_c, float alpha, float beta,
                                                            int fill_value, int stride, float* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // dst_index = (x,y,z)
    if (x < dst_w && y < dst_h && z < src_c)
    {
        // 计算缩放比例
        float scale = min((float)dst_w / src_w, (float)dst_h / src_h);
        int   new_w = (int)(src_w * scale);
        int   new_h = (int)(src_h * scale);

        // 计算填充
        int pad_x = (dst_w - new_w) / 2;
        int pad_y = (dst_h - new_h) / 2;

        // 检查是否在填充区域
        if (x < pad_x || x >= pad_x + new_w || y < pad_y || y >= pad_y + new_h)
        {
            // 填充颜色
            dst[z * dst_h * dst_w + y * dst_w + x] = fill_value * alpha + beta;
            // CHW格式 所以是z*h*w + y*w + x ，否则为y*w*c + x*c + z
        }
        else
        {
            // 计算源图像中的坐标
            int src_x = (int)((x - pad_x) / scale);
            int src_y = (int)((y - pad_y) / scale);
            // 将 BGR 转换为 RGB
            int rgb_index;
            if (z == 0)
            {
                rgb_index = src_y * src_w * src_c + src_x * src_c + 2;  // B -> R
            }
            else if (z == 1)
            {
                rgb_index = src_y * src_w * src_c + src_x * src_c + 1;  // G -> G
            }
            else
            {
                rgb_index = src_y * src_w * src_c + src_x * src_c + 0;  // R -> B
            }

            // 进行 HWC 到 CHW 的转换并归一化
            dst[z * dst_h * dst_w + y * dst_w + x] = ((float)(src[rgb_index])) * alpha + beta;
        }
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

void ConverHWC2CHWAlpahNormResizeKeepRatio(const unsigned char* src, int src_h, int src_w, int src_c, int dst_h,
                                           int dst_w, int dst_c, float alpha, float beta, float fill_value, int stride,
                                           float* dst)
{
    dim3 block(16, 16, 1);
    auto grid_x = (dst_w + block.x - 1) / block.x;
    auto grid_y = (dst_h + block.y - 1) / block.y;
    auto grid_z = (dst_c + block.z - 1) / block.z;
    dim3 grid(grid_x, grid_y, grid_z);

    // stream
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0);
    TimerGPU timer;
    timer.start(stream);

    // copy src to device
    unsigned char* src_d;
    cudaMallocAsync(&src_d, src_h * src_w * src_c * sizeof(unsigned char), stream);
    cudaMemcpyAsync(src_d, src, src_h * src_w * src_c * sizeof(unsigned char), cudaMemcpyHostToDevice, stream);

    ConverHWC2CHWAlpahNormResizeKeepRatioKernel<<<grid, block, 0, stream>>>(
        src_d, src_h, src_w, src_c, dst_h, dst_w, dst_c, alpha, beta, fill_value, stride, dst);

    // realize async
    cudaStreamSynchronize(stream);
    timer.stop("ConverHWC2CHWNorm", true);

    cudaFree(src_d);
    cudaStreamDestroy(stream);
}

}  // namespace CUDAKernal