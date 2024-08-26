#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace cv_infer
{
template <typename EngineType>
class ModelBase
{
public:
    virtual bool Init(const std::string& model_file) { return Engine.LoadModel(model_file); };
    //  virtual bool PreProcess(const std::vector<cv::Mat>& inputs, void* dst) = 0;
    //  virtual std::vector<std::vector<float>> PostProcess(const std::vector<std::vector<float>>& model_outputs) = 0;

public:
    std::string Model;
    EngineType  Engine;
};
}  // namespace cv_infer