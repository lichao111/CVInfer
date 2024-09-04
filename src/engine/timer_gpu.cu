#include <cuda.h>
#include <cuda_runtime.h>

#include "timer_gpu.cuh"
#include "tools/logger.h"

namespace cv_infer
{
TimerGPU::TimerGPU()
{
    checkRuntime(cudaEventCreate((cudaEvent_t *)&Start));
    checkRuntime(cudaEventCreate((cudaEvent_t *)&Stop));
}

TimerGPU::~TimerGPU()
{
    checkRuntime(cudaEventDestroy((cudaEvent_t)Start));
    checkRuntime(cudaEventDestroy((cudaEvent_t)Stop));
}

void TimerGPU::start(void *stream)
{
    Stream = stream;
    checkRuntime(cudaEventRecord((cudaEvent_t)Start, (cudaStream_t)stream));
}

float TimerGPU::stop(const char *prefix, bool print)
{
    checkRuntime(cudaEventRecord((cudaEvent_t)Stop, (cudaStream_t)Stream));
    checkRuntime(cudaEventSynchronize((cudaEvent_t)Stop));

    float latency = 0;
    checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t)Start, (cudaEvent_t)Stop));

    if (print)
    {
        LOGI("[%s]: %.5f ms", prefix, latency);
    }
    return latency;
}
}  // namespace cv_infer