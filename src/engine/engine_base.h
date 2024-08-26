#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace cv_infer
{
class EngineBase
{
    virtual bool LoadModel(const std::string& model) = 0;
};
}  // namespace cv_infer
