#pragma once

#include <functional>
#include <vector>

#include "signal/signal.h"

namespace cv_infer
{
// NodeBase* node = new NodeImpl(4,5);
// 具备的能力：
// 1. 有多个输入和输出
// 2. 每个输入和输出都是一个信号队列
// 3. start之后启动独立线程，不断从输入队列中取数据，处理后放入输出队列
// 4. 本身是一个线程池，可以批量处理数据， 线程数量可以配置
// 4. 有一个虚函数Worker，用于处理数据
class NodeBase
{
public:
    NodeBase()          = default;
    virtual ~NodeBase() = default;

    virtual bool Init() { return true; };
    bool         bind(std::vector<std::reference_wrapper<SignalQue>> input,
                      std::vector<std::reference_wrapper<SignalQue>> Outputs);
    void         SetInputCount(std::uint8_t count) { InputCount = count; }
    std::uint8_t GetInputCount() const { return InputCount; }

    void         SetOutputCount(std::uint8_t count) { OutputCount = count; }
    std::uint8_t GetOutputCount() const { return OutputCount; }

    virtual void Worker() = 0;

private:
    std::uint8_t                                   InputCount{0};
    std::uint8_t                                   OutputCount{0};
    std::vector<std::reference_wrapper<SignalQue>> InputList;
    std::vector<std::reference_wrapper<SignalQue>> OutputList;
};
}  // namespace cv_infer