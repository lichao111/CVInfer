#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>

#include "tools/logger.h"
namespace cv_infer
{
class Timer
{
public:
    Timer(const std::string& name, bool open = true) : Name(name), Open(open), Start(std::chrono::steady_clock::now())
    {
    }
    Timer(const Timer&)            = delete;
    Timer& operator=(const Timer&) = delete;
    Timer(Timer&&)                 = delete;
    Timer& operator=(Timer&&)      = delete;
    ~Timer()                       = default;

    void StartTimer() { Start = std::chrono::steady_clock::now(); }

    void EndTimer(const std::string& name = "")
    {
        End     = std::chrono::steady_clock::now();
        Elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(End - Start).count();
        Total += Elapsed;
        Average = static_cast<float>(Total) / ++Times;
        Min     = std::min(Min, Elapsed);
        Max     = std::max(Max, Elapsed);
        if (Open)
        {
            LOGI("timer [%s] - [%s] cost: Average: [%f] ms, Min: [%lu] ms, Max: [%lu] ms", Name.c_str(), name.c_str(),
                 Average, Min, Max);
        }
        Start = std::chrono::steady_clock::now();
    }

    void Reset()
    {
        Elapsed = 0;
        Start   = std::chrono::steady_clock::now();
        Times   = 0;
        Total   = 0;
        Min     = std::numeric_limits<std::uint64_t>::max();
        Max     = std::numeric_limits<std::uint64_t>::min();
        Average = 0.0f;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> Start;
    std::chrono::time_point<std::chrono::steady_clock> End;
    std::uint64_t                                      Elapsed{0};
    std::uint64_t                                      Times{0};
    std::uint64_t                                      Total{0};
    float                                              Average{0.0f};
    std::uint64_t                                      Min{std::numeric_limits<std::uint64_t>::max()};
    std::uint64_t                                      Max{std::numeric_limits<std::uint64_t>::min()};
    std::string                                        Name{"Timer"};
    bool                                               Open{true};
};
}  // namespace cv_infer
