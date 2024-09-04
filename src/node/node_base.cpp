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
        Future = std::async(std::launch::async, &NodeBase::Run, this);
    }
    return true;
}
bool NodeBase::Stop()
{
    if (Running)
    {
        Running = false;
        if (Future.valid())
        {
            Future.wait();
            Future.get();
        }
    }
    return true;
}

bool NodeBase::Run()
{
    while (Running)
    {
        CostTimer.StartTimer();
        if (Worker())
        {
            CostTimer.EndTimer(GetName());
        }
        else
        {
            // LOGT("NodeBase::Run Worker failed");
            std::this_thread::sleep_for(SleepTime);
        }
    }
    return true;
}

std::string NodeBase::Demangle(const char* name)
{
    int                                    status = 0;
    std::unique_ptr<char, void (*)(void*)> res{abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};
    return (status == 0) ? res.get() : name;
}

void NodeBase::SetName(const std::string& node_name) { NodeName = node_name; }

std::string NodeBase::GetName()
{
    if (NodeName.empty()) NodeName = Demangle(typeid(*this).name());
    return NodeName;
}

bool NodeBase::AddInputs(SignalQuePtr input)
{
    if (InputList.size() < InputCount)
    {
        InputList.push_back(input);
        return true;
    }
    return false;
}

bool NodeBase::AddOutputs(SignalQuePtr output)
{
    if (OutputList.size() < OutputCount)
    {
        OutputList.push_back(output);
        return true;
    }
    return false;
}

}  // namespace cv_infer