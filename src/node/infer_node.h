#pragma once

#include <opencv2/core/mat.hpp>
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
    InferNode() : NodeBase(1, 1) { SetName("InferNode"); };
    virtual ~InferNode(){};

    bool Init(const std::string& model)
    {
        auto ret = Model.Init(model);
        if (not ret)
        {
            LOGE("Model.Init failed");
            return false;
        }

        // worm up
        auto signal = std::make_shared<SignalImageBGR>(cv::Mat(720, 1280, CV_8UC3, cv::Scalar(0, 0, 0)));

        Model.Forwards({signal});
        return true;
    }

    virtual bool Worker() override
    {
        SignalBasePtr signal;
        if (not InputList[0]->Pop(signal))
        {
            return false;
        }
        if (signal->GetSignalType() != SignalType::SIGNAL_IMAGE_BGR)
        {
            LOGE("Input signal type not match, expect [%d], but got [%d]",
                 static_cast<int>(SignalType::SIGNAL_IMAGE_BGR), static_cast<int>(signal->GetSignalType()));
            return {};
        }
        auto signal_bgr = std::dynamic_pointer_cast<SignalImageBGR>(signal);

        std::vector<std::shared_ptr<SignalImageBGR>> inputs{signal_bgr};

        auto output_data = Model.Forwards(inputs);  // TODO: batch
        auto image       = signal_bgr->Val;
        auto frame_index = signal_bgr->FrameIdx;
        for (const auto& bbox : output_data)
        {
            cv::Rect rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
        }
        // auto output_signal      = std::make_shared<SignalImageBGR>(image);
        // output_signal->FrameIdx = frame_index;
        OutputList[0]->Push(std::move(signal));
        return true;
    }

private:
    ModelType<EngineType> Model;
};
}  // namespace cv_infer