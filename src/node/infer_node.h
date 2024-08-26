#pragma once

#include <opencv2/opencv.hpp>

#include "node_base.h"
#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
template <template <typename> typename ModelType, typename EngineType>
class InferNode : public NodeBase
{
public:
    InferNode() : NodeBase(1, 1, 1){};
    virtual ~InferNode(){};

    bool Init(const std::string& model) { return Model.Init(model); }

    virtual std::vector<SignalBasePtr> Worker(std::vector<SignalBasePtr> input_signals) override
    {
        std::vector<SignalBasePtr>  output_signals;
        std::vector<SignalImageBGR> input_datas;
        for (auto& input_signal : input_signals)
        {
            if (input_signal->GetSignalType() != SignalType::SIGNAL_IMAGE_BGR)
            {
                LOGE("Input signal type not match, expect [%d], but got [%d]",
                     static_cast<int>(SignalType::SIGNAL_IMAGE_BGR), static_cast<int>(input_signal->GetSignalType()));
                return {};
            }
            auto    tmp = std::dynamic_pointer_cast<SignalImageBGR>(input_signal);
            cv::Mat resized;
            cv::resize(tmp->Val, resized, cv::Size(768, 512));
            input_datas.push_back(SignalImageBGR(resized));
        }
        auto output_data = Model.Forwards(input_datas);
        for (const auto output_date : output_data)
        {
            // output_signals.push_back(std::make_shared<Signal>(output_date));
        }
        auto image = std::dynamic_pointer_cast<SignalImageBGR>(input_signals[0])->Val;

        for (const auto& bbox : output_data)
        {
            cv::Rect rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        }

        return input_signals;
    }

private:
    ModelType<EngineType> Model;
};
}  // namespace cv_infer