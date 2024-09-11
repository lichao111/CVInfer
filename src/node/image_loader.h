#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#include "node/node_base.h"
namespace cv_infer
{

class ImageLoader : public NodeBase
{
public:
    ImageLoader();
    virtual ~ImageLoader();

    virtual bool Init(const std::string &src);
    virtual bool Run() override;
    virtual bool Worker() override { return true; };

private:
    std::string Src;

    bool IsDirectory{false};  // directory or file
};
}  // namespace cv_infer