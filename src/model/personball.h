#pragma once

#include <algorithm>
#include <optional>

#include "engine/preprocess_kernal.cuh"
#include "math/math.h"
#include "math/nms.h"
#include "model/model_base.h"
#include "signal/signal.h"
#include "tools/logger.h"
#include "tools/timer.h"

namespace cv_infer
{
template <typename EngineType>
class PersonBall : public ModelBase<EngineType>
{
public:
    virtual bool Init(const std::string& model)
    {
        if (not ModelBase<EngineType>::Init(model, DevicePreProcess))
        {
            LOGE("ModelBase<EngineType>::Init failed");
            return false;
        }
        if (not(this->Engine)
                   .RegisterPreProcessFunc(
                       std::bind(&PersonBall::PreProcess, this, std::placeholders::_1, std::placeholders::_2)))
        {
            LOGE("Engine.RegisterPreProcessFunc failed");
            return false;
        }
        if (not(this->Engine).RegisterPostProcessFunc(std::bind(&PersonBall::PostProcess, this, std::placeholders::_1)))
        {
            LOGE("Engine.RegisterPostProcessFunc failed");
            return false;
        }
        return true;
    }
    bool PreProcess(const std::vector<cv::Mat>& inputs_batch, std::vector<float*>& preprocessed)
    {
        const auto num_inputs = inputs_batch.size();
        if (inputs_batch.size() != num_inputs)
        {
            LOGE("Input signals size not match, expect [%d], but got [%d]", num_inputs, inputs_batch.size());
            return false;
        }

        int input_index = 0;
        for (const auto& image : inputs_batch)
        {
            auto width   = image.cols;
            auto height  = image.rows;
            auto channel = image.channels();

            if (DevicePreProcess)
            {
                CUDAKernal::ConverHWC2CHWMeanStd(image.data, height, width, channel, Mean.data(), Scale.data(),
                                                 preprocessed[input_index]);
            }
            else
            {
                ConverHWC2CHWMeanStd(image.data, height, width, channel, Mean.data(), Scale.data(),
                                     preprocessed[input_index]);
            }
        }

        return true;
    }

    std::vector<std::vector<float>> PostProcess(std::vector<std::vector<float>> model_outputs)
    {
        int offset = 0;

        float x_scale = InputWidth.value() / static_cast<float>(InferWidth.value());
        float y_scale = InputHeight.value() / static_cast<float>(InferHeight.value());

        auto               data = model_outputs[0];
        std::vector<float> temp_data(data.size());

        // std::copy((temp_data, data.data(), sizeof(float) * data.size());
        std::copy(data.begin(), data.end(), temp_data.begin());
        for (int level_i = 0; level_i < LevelNum; ++level_i)
        {
            int h      = LevelHW[level_i * 2 + 0];
            int w      = LevelHW[level_i * 2 + 1];
            int stride = LevelStrides[level_i];

            std::vector<float> xg(h * w);
            std::vector<float> yg(h * w);

            bigmeshgrid(h, w, xg.data(), yg.data());

            for (int start_i = offset; start_i < offset + h * w; ++start_i)
            {
                temp_data[start_i * 7 + 0] = (data[start_i * 7 + 0] + xg[start_i - offset]) * stride;
                temp_data[start_i * 7 + 1] = (data[start_i * 7 + 1] + yg[start_i - offset]) * stride;

                temp_data[start_i * 7 + 2] = exp(data[start_i * 7 + 2]) * stride;
                temp_data[start_i * 7 + 3] = exp(data[start_i * 7 + 3]) * stride;
            }

            offset += h * w;
        }

        std::vector<std::vector<float>> person_bboxes;
        std::vector<std::vector<float>> ball_bboxes;
        for (int i = 0; i < OutputSize; ++i)  // 8064 is output size, this value is fixed by model
        {
            float* ptr        = temp_data.data() + i * 7;
            float  cx         = ptr[0];
            float  cy         = ptr[1];
            float  w          = ptr[2];
            float  h          = ptr[3];
            float  obj_pred   = ptr[4];
            float  cls_0_pred = ptr[5];
            float  cls_1_pred = ptr[6];

            // obj_pred 0.2 best
            if (obj_pred > 0.1)
            {
                if (cls_0_pred > cls_1_pred && cls_0_pred > 0.5)
                {
                    person_bboxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls_0_pred});
                }
                else if (cls_0_pred < cls_1_pred && cls_1_pred > 0.3)
                {
                    ball_bboxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls_1_pred});
                }
            }
        }

        if (person_bboxes.size() > 0)
        {
            // preson
            std::vector<unsigned int> filter_person_index;
            filter_person_index = bignms(person_bboxes, 0.2);
            std::vector<std::vector<float>> filter_person_bboxes;
            for (int i = 0; i < filter_person_index.size(); ++i)
            {
                filter_person_bboxes.push_back(person_bboxes[filter_person_index[i]]);
            }
            person_bboxes = filter_person_bboxes;
        }
        if (ball_bboxes.size() > 0)
        {
            // ball
            std::vector<unsigned int> filter_ball_index;
            filter_ball_index = bignms(ball_bboxes, 0.01);
            std::vector<std::vector<float>> filter_ball_bboxes;
            for (int i = 0; i < filter_ball_index.size(); ++i)
            {
                filter_ball_bboxes.push_back(ball_bboxes[filter_ball_index[i]]);
            }
            ball_bboxes = filter_ball_bboxes;
        }
        // 合并结果
        int person_num          = person_bboxes.size();
        int ball_num            = ball_bboxes.size();
        int person_and_ball_num = person_num + ball_num;

        std::vector<std::vector<float>> bboxes(person_and_ball_num, std::vector<float>(5, 0.0));

        float labels[person_and_ball_num];

        for (int i = 0; i < person_and_ball_num; ++i)
        {
            if (i < person_bboxes.size())
            {
                bboxes[i][0] = person_bboxes[i][0] * x_scale;
                bboxes[i][1] = person_bboxes[i][1] * y_scale;
                bboxes[i][2] = person_bboxes[i][2] * x_scale;
                bboxes[i][3] = person_bboxes[i][3] * y_scale;
                bboxes[i][4] = person_bboxes[i][4];

                labels[i] = 0;
            }
            else
            {
                bboxes[i][0] = ball_bboxes[i - person_num][0] * x_scale;
                bboxes[i][1] = ball_bboxes[i - person_num][1] * y_scale;
                bboxes[i][2] = ball_bboxes[i - person_num][2] * x_scale;
                bboxes[i][3] = ball_bboxes[i - person_num][3] * y_scale;
                bboxes[i][4] = ball_bboxes[i - person_num][4];

                labels[i] = 1;
            }
        }
        return bboxes;
    }

    // TODO: get return type from PostProcess
    std::vector<std::vector<float>> Forwards(const std::vector<std::shared_ptr<SignalImageBGR>>& inputs)
    {
        std::vector<cv::Mat> images;
        for (const auto& input : inputs)
        {
            if (input->Val.empty())
            {
                LOGE("The input image is empty");
                return {};
            }
            if (not InputWidth.has_value())
            {
                InputWidth = input->Val.cols;
            }
            if (not InputHeight.has_value())
            {
                InputHeight = input->Val.rows;
            }
            cv::Mat resized;
            CostTimer.StartTimer();
            cv::resize(input->Val, resized, cv::Size(InferWidth.value(), InferHeight.value()));
            CostTimer.EndTimer("resize");
            images.push_back(resized);
        }
        auto ret = (this->Engine).Forwards(images);
        return ret;
    }

protected:
    Timer CostTimer{"resize"};
    bool  DevicePreProcess{true};

    std::optional<int> InputWidth;
    std::optional<int> InputHeight;
    std::optional<int> InferWidth{768};
    std::optional<int> InferHeight{512};

    std::vector<float> Mean{128, 128, 128};
    std::vector<float> Scale{128, 128, 128};

    std::vector<int> LevelHW{64, 96, 32, 48, 16, 24};
    std::vector<int> LevelStrides{8, 16, 32};
    int              LevelNum{3};

    int OutputSize{8064};
};

}  // namespace cv_infer