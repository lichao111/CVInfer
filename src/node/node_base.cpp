#include "node_base.h"

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
    }
    return true;
}
bool NodeBase::Stop()
{
    Running = false;
    return true;
}
bool NodeBase::Run()
{
    while (Running)
    {
        InputSignals.clear();
        for (auto& input : InputList)
        {
            SignalBasePtr sig;
            if (input.get().Pop(sig))
            {
                InputSignals.push_back(sig);
            }
        }

        std::vector<std::future<std::vector<SignalBasePtr>>> futures;
        while (futures.size() < OutputCount)
        {
            auto ret =
                PoolPtr->Commit([this]() { return Worker(InputSignals); });
            futures.emplace_back(std::move(ret));
        }

        for (auto& future : futures)
        {
            future.wait();
            std::size_t output_index = 0;
            for (auto& sig : future.get())
            {
                OutputList[output_index].Push(sig);
                output_index++;
            }
        }
        LOGT("NodeBase::Run() finish one loop");
    }
    return true;
}
}  // namespace cv_infer