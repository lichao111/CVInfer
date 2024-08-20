#pragma once
#include <initializer_list>
#include <string>
#include <vector>

#include "node/node_base.h"
#include "tools/defines.h"
namespace cv_infer
{
class PipelineBase
{
public:
    PipelineBase() = default;
    PipelineBase(const std::string& name) : PipelineName(name) {}

    PipelineBase(const PipelineBase&)            = delete;
    PipelineBase& operator=(const PipelineBase&) = delete;
    PipelineBase(PipelineBase&&)                 = delete;
    PipelineBase& operator=(PipelineBase&&)      = delete;

    virtual ~PipelineBase() = default;

    virtual bool Init();
    virtual bool Start();
    virtual bool Stop();
    virtual bool Check();
    virtual bool BindAll(std::vector<std::shared_ptr<NodeBase>> node_list);
    virtual bool Bind(std::shared_ptr<NodeBase> pre,
                      std::shared_ptr<NodeBase> next);
    virtual bool SetSource(const std::string& source);
    virtual bool RegisterCallback(EventId event, EventCallbackFunc callback);

    std::string GetName() const { return PipelineName; }

private:
    bool InitAllNode(
        std::initializer_list<std::shared_ptr<NodeBase>> node_list);

private:
    std::string      PipelineName = "Pipeline";
    std::string      Source;
    EventCallbackMap CallBackMap;

    std::vector<std::shared_ptr<NodeBase>> NodeList;
};
}  // namespace cv_infer