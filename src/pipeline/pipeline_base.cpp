#include "pipeline_base.h"

#include "node/node_base.h"
#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
bool PipelineBase::Init() { return true; }
bool PipelineBase::Start()
{
    for (const auto& node : NodeList)
    {
        if (not node->Start())
        {
            LOGE("Node [%s] start failed", node->GetName().c_str());
            return false;
        }
    }
    return true;
}

bool PipelineBase::Stop()
{
    for (const auto& node : NodeList)
    {
        if (not node->Stop())
        {
            LOGE("Node [%s] stop failed", node->GetName().c_str());
            return false;
        }
    }
    return true;
}

bool PipelineBase::Check()
{
    auto size = NodeList.size();
    for (auto node_idx = 0; node_idx < size - 1; ++node_idx)
    {
        if (NodeList[node_idx]->GetOutputsCount() != NodeList[node_idx + 1]->GetInputsCount())
        {
            LOGE(
                "Node [%s] output count = [%d] is not equal to Node [%s] input "
                "count  = [%d]",
                NodeList[node_idx]->GetName().c_str(), NodeList[node_idx]->GetOutputsCount(),
                NodeList[node_idx + 1]->GetName().c_str()),
                NodeList[node_idx + 1]->GetInputsCount();
            return false;
        }
    }
    return true;
}

bool PipelineBase::BindAll(std::vector<std::shared_ptr<NodeBase>> node_list)
{
    NodeList  = node_list;
    auto size = NodeList.size();
    for (auto node_idx = 0; node_idx < size - 1; ++node_idx)
    {
        if (not Bind(NodeList[node_idx], NodeList[node_idx + 1]))
        {
            LOGE("Bind Node [%s] and Node [%s] failed", NodeList[node_idx]->GetName().c_str(),
                 NodeList[node_idx + 1]->GetName().c_str());
            return false;
        }
    }
    return true;
}

// TODO: 默认了所有节点都是一个输出!!!
bool PipelineBase::Bind(std::shared_ptr<NodeBase> pre, std::shared_ptr<NodeBase> next)
{
    SignalQuePtr signal_queue = std::make_shared<SignalQue>();
    if (not next->AddInputs(signal_queue))
    {
        LOGE("Node [%s] AddInputs failed", next->GetName().c_str());
        return false;
    }
    if (not pre->AddOutputs(signal_queue))
    {
        LOGE("Node [%s] AddOutputs failed", pre->GetName().c_str());
        return false;
    }
    return true;
}

bool PipelineBase::SetSource(const std::string& source)
{
    Source = source;
    return true;
};

bool PipelineBase::RegisterCallback(EventId event, EventCallbackFunc callback)
{
    CallBackMap[event] = callback;
    return true;
};
}  // namespace cv_infer