#pragma once

#include <atomic>
#include <future>

#include "signal/signal.h"
#include "tools/timer.h"

namespace cv_infer
{
// NodeBase* node = new NodeImpl(4,5);
// 具备的能力：
// 1. 有多个输入和输出
// 2. 每个输入和输出都是一个信号队列
// 3. start之后启动独立线程，不断从输入队列中取数据，处理后放入输出队列
class NodeBase
{
public:
    NodeBase(std::size_t inputs, std::size_t outputs) : InputCount(inputs), OutputCount(outputs)
    {
        NodeName = GetName();
    };

    virtual ~NodeBase() { Stop(); };

    virtual bool Init() { return true; }

    bool AddInputs(SignalQuePtr input);
    bool AddOutputs(SignalQuePtr output);

    std::size_t GetInputsCount() const { return InputCount; }
    std::size_t GetOutputsCount() const { return OutputCount; }

    virtual bool Start();
    virtual bool Stop();
    virtual bool Run();         // 发起线程 执行worker函数 子类需要实现Worker函数
    virtual bool Worker() = 0;  // 消费输入队列，生产输出队列

    void        SetName(const std::string& node_name);
    std::string GetName();

protected:
    std::string Demangle(const char* name);

    std::string NodeName;
    std::size_t InputCount{0};
    std::size_t OutputCount{0};

    SignalQuePtrList InputList;
    SignalQuePtrList OutputList;

    std::future<bool> Future;
    std::atomic_bool  Running{false};

    std::chrono::milliseconds SleepTime{1};

    Timer CostTimer;
};
}  // namespace cv_infer