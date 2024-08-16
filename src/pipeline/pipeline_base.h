#pragma once
namespace cv_infer
{
class PipelineBase
{
public:
    PipelineBase()          = default;
    virtual ~PipelineBase() = default;

    virtual bool Init(std::size_t thread_count) = 0;
    virtual bool Start()                        = 0;
    virtual bool Stop()                         = 0;
    virtual void Worker()                       = 0;
};
}  // namespace cv_infer