#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>
#include <vector>

#include "engine/preprocess_kernal.cuh"
#include "model/model_base.h"
#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
enum YoloType : int
{
    YOLOVX,
    YOLOV5,
    YOLOV7,
    YOLOV8,
    YOLOV9,
    YOLOV10,
};
template <typename EngineType, YoloType yolo_type>
class Yolo : public ModelBase<EngineType>
{
public:
    virtual bool Init(const std::string &model)
    {
        if (not ModelBase<EngineType>::Init(model, DevicePreProcess))
        {
            LOGE("ModelBase<EngineType>::Init failed");
            return false;
        }
        if (not(this->Engine)
                   .RegisterPreProcessFunc(
                       std::bind(&Yolo::PreProcess, this, std::placeholders::_1, std::placeholders::_2)))
        {
            LOGE("Engine.RegisterPreProcessFunc failed");
            return false;
        }
        if (not(this->Engine).RegisterPostProcessFunc(std::bind(&Yolo::PostProcess, this, std::placeholders::_1)))
        {
            LOGE("Engine.RegisterPostProcessFunc failed");
            return false;
        }
        return true;
    }
    bool PreProcess(const std::vector<cv::Mat> &inputs_batch, std::vector<float *> preprocessed)
    {
        const auto num_inputs = inputs_batch.size();
        if (inputs_batch.size() != num_inputs)
        {
            LOGE("Input signals size not match, expect [%d], but got [%d]", num_inputs, inputs_batch.size());
            return false;
        }

        int input_index = 0;
        for (const auto &image : inputs_batch)
        {
            auto width   = image.cols;
            auto height  = image.rows;
            auto channel = image.channels();

            if (DevicePreProcess)
            {
                CUDAKernal::ConverHWC2CHWNorm(image.data, height, width, channel, preprocessed[input_index]);
            }
            else
            {
                LOGE("ConverHWC2CHWMeanStd not implemented");
                return false;
            }
        }
        return true;
    };
    std::vector<std::vector<float>> PostProcess(const std::vector<std::vector<float>> &outputs)
    {
        std::vector<std::vector<float>> person_bboxes;

        auto confidence_threshold = 0.5f;

        auto anchor_num = 25200;
        auto class_num  = 85;

        int anchor_index = 0;

        // 1. 转换
        // outputs.shape = [1 * 214200] (214200 = 25200 * 85)
        for (int anchor_index = 0; anchor_index < anchor_num; anchor_index++)
        {
            std::vector<float> person_bbox;

            float cx     = outputs[0][anchor_index * class_num + 0];
            float cy     = outputs[0][anchor_index * class_num + 1];
            float w      = outputs[0][anchor_index * class_num + 2];
            float h      = outputs[0][anchor_index * class_num + 3];
            float left   = cx - w * 0.5f;
            float top    = cy - h * 0.5f;
            float right  = cx + w * 0.5f;
            float bottom = cy + h * 0.5f;

            if (outputs[0][anchor_index * class_num + 4] < confidence_threshold)
            {
                continue;
            }

            person_bbox.push_back(left * (InputWidth.value() / static_cast<float>(InferWidth.value())));
            person_bbox.push_back(top * (InputHeight.value() / static_cast<float>(InferHeight.value())));
            person_bbox.push_back(right * (InputWidth.value() / static_cast<float>(InferWidth.value())));
            person_bbox.push_back(bottom * (InputHeight.value() / static_cast<float>(InferHeight.value())));
            person_bbox.push_back(outputs[0][anchor_index * class_num + 4]);
            auto max_confidence = 0.0f;
            int  label          = -1;
            for (int j = 0; j < 80; j++)
            {
                if (outputs[0][anchor_index * class_num + 5 + j] > max_confidence)
                {
                    max_confidence = outputs[0][anchor_index * class_num + 5 + j];
                    label          = j;
                }
            }
            person_bbox.push_back(label);
            person_bboxes.push_back(person_bbox);
        }

        // 2. nms
        auto IoU = [](const std::vector<float> &a, const std::vector<float> &b)
        {
            float x1    = std::max(a[0], b[0]);
            float y1    = std::max(a[1], b[1]);
            float x2    = std::min(a[2], b[2]);
            float y2    = std::min(a[3], b[3]);
            float w     = std::max(0.0f, x2 - x1);
            float h     = std::max(0.0f, y2 - y1);
            float inter = w * h;
            float area1 = (a[2] - a[0]) * (a[3] - a[1]);
            float area2 = (b[2] - b[0]) * (b[3] - b[1]);
            return inter / (area1 + area2 - inter);
        };
        std::vector<std::vector<float>> filtered_person_bboxes;
        std::sort(person_bboxes.begin(), person_bboxes.end(),
                  [](const std::vector<float> &a, const std::vector<float> &b) { return a[4] > b[4]; });

        std::vector<bool> skip(person_bboxes.size(), false);
        for (int i = 0; i < person_bboxes.size(); ++i)
        {
            if (skip[i])
            {
                continue;
            }
            for (int j = i + 1; j < person_bboxes.size(); ++j)
            {
                if (skip[j])
                {
                    continue;
                }
                if (IoU(person_bboxes[i], person_bboxes[j]) > 0.5)
                {
                    skip[j] = true;
                }
            }
            filtered_person_bboxes.push_back(person_bboxes[i]);
        }

        return filtered_person_bboxes;
    }
    std::vector<std::vector<float>> Forwards(const std::vector<std::shared_ptr<SignalImageBGR>> &inputs)
    {
        std::vector<cv::Mat> images;
        for (const auto &input : inputs)
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
            cv::resize(input->Val, resized, cv::Size(InferWidth.value(), InferHeight.value()));
            images.push_back(resized);
        }
        auto ret = (this->Engine).Forwards(images);
        return ret;
    }

protected:
    bool DevicePreProcess{true};

    std::optional<int> InputWidth;
    std::optional<int> InputHeight;
    std::optional<int> InferWidth{640};
    std::optional<int> InferHeight{640};
};
}  // namespace cv_infer
