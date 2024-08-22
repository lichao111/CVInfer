#include "node_base.h"

#include <cxxabi.h>

#include <future>
#include <thread>

#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
bool NodeBase::Start()
{
    Running = true;
    if (Running)
    {
        PoolPtr->Start();
        Future = std::async(std::launch::async, [this]() { return Run(); });
    }
    return true;
}
bool NodeBase::Stop()
{
    Running = false;
    if (Future.valid())
    {
        Future.wait();
        Future.get();
    }
    return true;
}

std::string NodeBase::Demangle(const char* name)
{
    int                                    status = 0;
    std::unique_ptr<char, void (*)(void*)> res{
        abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
    return (status == 0) ? res.get() : name;
}
SignalQueRefList NodeBase::GetOutputList()
{
    SignalQueRefList output_list;
    for (auto& output : OutputList)
    {
        output_list.push_back(std::ref(output));
    }
    return std::move(output_list);
}

bool NodeBase::SetInputs(SignalQueRefList Inputs)
{
    if (Inputs.size() != InputCount)
    {
        LOGE(
            "NodeBase::SetInputs() Inputs.size() != InputCount, excepted: "
            "[%d], actual: [%d]",
            InputCount, Inputs.size());
        return false;
    }
    InputList = Inputs;
    return true;
}

bool NodeBase::Run()
{
    // TODO: 使用线程池并发 现在每次只提交一个任务
    while (Running)
    {
        InputSignals.clear();
        bool all_input_is_prepared = false;

        while (Running and (not IsSignalQueRefListReady(InputList)))
        {
            std::this_thread::sleep_for(SleepTime);
        }
        if (not Running)
        {
            return true;
        }
        if (InputSignals = std::move(GetSignaList(InputList));
            InputSignals.size() != InputCount)
        {
            LOGE(
                "fatal !! NodeBase::Run() InputSignals.size() != InputCount, "
                "excepted: [%d], actual: [%d]",
                InputCount, InputSignals.size());
            continue;
        }

        auto ret = PoolPtr->Commit([this]() { return Worker(InputSignals); });
        ret.wait();
        auto output_signals = ret.get();
        if (output_signals.size() != OutputCount)
        {
            LOGE(
                "NodeBase::Run() output_signals.size() != OutputCount, "
                "excepted: [%d], actual: [%d]",
                OutputCount, output_signals.size());
            continue;
        }
        std::size_t output_index = 0;
        for (auto& sig : output_signals)
        {
            OutputList[output_index].Push(sig);
            output_index++;
        }
        LOGT("NodeBase::Run() finish one loop");
    }
    return true;
}
}  // namespace cv_infer