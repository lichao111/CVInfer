#include "node_base.h"

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
    Future.wait();
    Future.get();
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