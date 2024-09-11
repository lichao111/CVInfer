#include "image_loader.h"

#include <filesystem>

#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
ImageLoader::ImageLoader() : NodeBase(0, 1) { SetName("ImageLoader"); }

ImageLoader::~ImageLoader() {}

bool ImageLoader::Init(const std::string &src)
{
    if (src.empty())
    {
        LOGE("src is empty");
        return false;
    }
    if (std::filesystem::is_empty(src))
    {
        LOGE("src is empty");
        return false;
    }
    if (std::filesystem::is_directory(src))
    {
        IsDirectory = true;
        LOGI("src is a directory");
    }
    Src = src;
    return true;
}
bool ImageLoader::Run()
{
    if (not IsDirectory)
    {
        LOGI("src is a file");
        return false;
    }
    for (const auto &entry : std::filesystem::directory_iterator(Src))
    {
        if (entry.is_regular_file())
        {
            auto image = cv::imread(entry.path().string());
            if (image.empty())
            {
                LOGE("Read image failed: [%s]", entry.path().string().c_str());
                continue;
            }
            auto signal = std::make_shared<SignalImageBGR>(image);
            OutputList[0]->Push(signal);
        }
    }
    return true;
}
}  // namespace cv_infer