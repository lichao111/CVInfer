#pragma once
#include <stdio.h>

namespace cv_infer
{

#define checkRuntime(call)                                                                                           \
    do                                                                                                               \
    {                                                                                                                \
        auto ___call__ret_code__ = (call);                                                                           \
        if (___call__ret_code__ != cudaSuccess)                                                                      \
        {                                                                                                            \
            printf("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call, cudaGetErrorString(___call__ret_code__), \
                   cudaGetErrorName(___call__ret_code__), ___call__ret_code__);                                      \
            abort();                                                                                                 \
        }                                                                                                            \
    } while (0)

class TimerGPU
{
public:
    TimerGPU();
    virtual ~TimerGPU();
    void  start(void *stream = nullptr);
    float stop(const char *prefix = "Timer", bool print = true);

private:
    void *Start, *Stop;
    void *Stream;
};
}  // namespace cv_infer