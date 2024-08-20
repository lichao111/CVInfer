#include "pipeline_base.h"

#include "node/node_base.h"
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
        if (NodeList[node_idx]->GetOutputsCount() !=
            NodeList[node_idx + 1]->GetInputsCount())
        {
            LOGE(
                "Node [%s] output count = [%d] is not equal to Node [%s] input "
                "count  = [%d]",
                NodeList[node_idx]->GetName().c_str(),
                NodeList[node_idx]->GetOutputsCount(),
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
            LOGE("Bind Node [%s] and Node [%s] failed",
                 NodeList[node_idx]->GetName().c_str(),
                 NodeList[node_idx + 1]->GetName().c_str());
            return false;
        }
    }
    return true;
}

bool PipelineBase::Bind(std::shared_ptr<NodeBase> pre,
                        std::shared_ptr<NodeBase> next)
{
    if (pre->GetOutputsCount() != next->GetInputsCount())
    {
        LOGE(
            "Node [%s] output count = [%d] is not equal to Node [%s] input "
            "count  = [%d]",
            pre->GetName().c_str(), pre->GetOutputsCount(),
            next->GetName().c_str(), next->GetInputsCount());
        return false;
    }
    return next->SetInputs(pre->GetOutputList());
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