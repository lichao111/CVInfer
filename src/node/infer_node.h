#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include "node_base.h"
#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
template <typename ModelType>
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

        auto               output_data  = Model.Forwards(inputs);  // TODO: batch
        auto               image        = signal_bgr->Val;
        auto               frame_index  = signal_bgr->FrameIdx;
        static const char* cocolabels[] = {"person",        "bicycle",      "car",
                                           "motorcycle",    "airplane",     "bus",
                                           "train",         "truck",        "boat",
                                           "traffic light", "fire hydrant", "stop sign",
                                           "parking meter", "bench",        "bird",
                                           "cat",           "dog",          "horse",
                                           "sheep",         "cow",          "elephant",
                                           "bear",          "zebra",        "giraffe",
                                           "backpack",      "umbrella",     "handbag",
                                           "tie",           "suitcase",     "frisbee",
                                           "skis",          "snowboard",    "sports ball",
                                           "kite",          "baseball bat", "baseball glove",
                                           "skateboard",    "surfboard",    "tennis racket",
                                           "bottle",        "wine glass",   "cup",
                                           "fork",          "knife",        "spoon",
                                           "bowl",          "banana",       "apple",
                                           "sandwich",      "orange",       "broccoli",
                                           "carrot",        "hot dog",      "pizza",
                                           "donut",         "cake",         "chair",
                                           "couch",         "potted plant", "bed",
                                           "dining table",  "toilet",       "tv",
                                           "laptop",        "mouse",        "remote",
                                           "keyboard",      "cell phone",   "microwave",
                                           "oven",          "toaster",      "sink",
                                           "refrigerator",  "book",         "clock",
                                           "vase",          "scissors",     "teddy bear",
                                           "hair drier",    "toothbrush"};
        for (const auto& bbox : output_data)
        {
            cv::Rect rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
            cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
            // write label and confidence
            std::string label = cocolabels[static_cast<int>(bbox[5])];
            cv::putText(image, label, cv::Point2f(bbox[0], bbox[1]), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 255), 2);
            float confidence = bbox[4];
            label            = std::to_string(confidence);
            cv::putText(image, label, cv::Point2f(bbox[0], bbox[1] + 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 0, 255), 2);
        }
        // auto output_signal      = std::make_shared<SignalImageBGR>(image);
        // output_signal->FrameIdx = frame_index;
        OutputList[0]->Push(std::move(signal));
        return true;
    }

private:
    ModelType Model;
};
}  // namespace cv_infer