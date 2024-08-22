#include <gtest/gtest.h>

#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "engine/trt_infer.h"
#include "signal/signal.h"

using namespace cv_infer;
using namespace cv_infer::engine;

TEST(TrtInfer, ModelPersonBall)
{
    TrtEngine engine;
    EXPECT_TRUE(engine.LoadModel("/workspace/github/CVInfer/test/personball_512_768_best_1_0.onnx"));
    auto    image = cv::imread("/workspace/github/CVInfer/build/test/file_frame_1.jpg");
    cv::Mat resized_image(512, 768, CV_8UC3);
    cv::resize(image, resized_image, cv::Size(768, 512));
    auto                       input = std::make_shared<SignalImageBGR>(resized_image);
    std::vector<SignalBasePtr> output{input};

    auto bboxes = engine.Forwards({input}, output);
    for (auto& bbox : bboxes)
    {
        cv::Rect rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("tmp.png", image);
}
