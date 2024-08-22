#pragma once

#include <atomic>
#include <memory>
#include <typeinfo>

#include "signal/signal.h"
#include "tools/threadpool.h"

namespace cv_infer
{
// NodeBase* node = new NodeImpl(4,5);
// 具备的能力：
// 1. 有多个输入和输出
// 2.
// 每个输入和输出都是一个信号队列(输入的信号队列是一个引用队列，输出的信号队列是一个普通队列)
// 3. start之后启动独立线程，不断从输入队列中取数据，处理后放入输出队列
// 4. 本身是一个线程池，可以批量处理数据， 线程数量可以配置
// 4. 有一个虚函数Worker，用于处理数据
class NodeBase
{
public:
    NodeBase(std::size_t inputs, std::size_t outputs, std::size_t thds = 1)
        : InputCount(inputs), OutputCount(outputs)
    {
        PoolPtr = std::make_unique<ThreadPool>(thds);
        OutputList.resize(outputs);
        NodeName = GetName();
    };

    virtual ~NodeBase() = default;

    virtual bool Init() { return true; }

    bool             SetInputs(SignalQueRefList Inputs);
    std::size_t      GetInputsCount() const { return InputCount; }
    std::size_t      GetOutputsCount() const { return OutputList.size(); }
    SignalQueRefList GetOutputList();

    virtual bool Start();
    virtual bool Stop();
    virtual bool Run();  // 不断的把Worker提交到线程池 为了输出的顺序性
                         // 需要控制最大提交数量不超过线程池大小

    virtual std::vector<SignalBasePtr> Worker(
        std::vector<SignalBasePtr> input_signals) = 0;  // 处理输出数据

    void        SetName(const std::string node_name) { NodeName = node_name; }
    std::string GetName()
    {
        if (NodeName.empty()) NodeName = Demangle(typeid(*this).name());
        return NodeName;
    }

protected:
    std::string Demangle(const char* name);

    std::string NodeName;
    std::size_t InputCount{0};
    std::size_t OutputCount{0};

    SignalQueRefList InputList;
    SignalQueList    OutputList;
    // std::vector<std::queue<std::shared_ptr<SignalBase>>> OutputList;

    std::unique_ptr<ThreadPool> PoolPtr;
    std::future<bool>           Future;
    std::atomic_bool            Running{false};
    std::chrono::milliseconds   SleepTime{10};

    std::vector<SignalBasePtr>
        InputSignals;  //[[inputs_port[0], inputs_port[1], ...]
};
}  // namespace cv_infer