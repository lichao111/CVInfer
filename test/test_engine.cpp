#include <gtest/gtest.h>

#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "engine/trt_infer.h"
#include "model/personball.h"
#include "node/infer_node.h"
#include "signal/signal.h"
#include "tools/timer.h"

using namespace cv_infer;

// TEST(TrtInfer, ModelPersonBall)
// {
//     TrtEngine engine;
//     EXPECT_TRUE(engine.LoadModel("/workspace/github/CVInfer/test/personball_512_768_best_1_0.onnx"));
//     auto    image = cv::imread("/workspace/github/CVInfer/test/vlcsnap-2024-08-22-10h45m57s292.png");
//     cv::Mat resized_image(512, 768, CV_8UC3);
//     cv::resize(image, resized_image, cv::Size(768, 512));
//     auto                       input = std::make_shared<SignalImageBGR>(resized_image);
//     std::vector<SignalBasePtr> output{input};

//     auto bboxes = engine.Forwards({input}, output);
//     for (auto& bbox : bboxes)
//     {
//         cv::Rect rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
//         cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
//     }
//     cv::imwrite("tmp.png", image);
// }

TEST(TrtInfer, Personball)
{
    Timer timer("PersonBall");
    timer.StartTimer();
    auto model = std::make_unique<PersonBall<trt::TrtEngine>>();
    EXPECT_TRUE(model->Init("/workspace/github/CVInfer/test/personball_512_768_best_1_0.onnx"));
    timer.EndTimer("model load");
    auto image         = cv::imread("/workspace/github/CVInfer/test/street.jpg");
    auto image_resized = cv::Mat(512, 768, CV_8UC3);
    cv::resize(image, image_resized, cv::Size(768, 512));
    auto input_signals = std::make_shared<SignalImageBGR>(image_resized);
    timer.EndTimer("image resize");

    // for (int i = 0; i <= 1000; i++)
    // {
    auto output_signals = model->Forwards({input_signals});
    timer.EndTimer("model forward");
    // }
    for (const auto& bbox : output_signals)
    {
        EXPECT_TRUE(bbox.size() == 5);
        cv::Rect rect(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]));
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("personball.png", image);
}

TEST(TrtInfer, infer_node) { auto infer_node = std::make_shared<InferNode<PersonBall<trt::TrtEngine>>>(); }

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Logger::SetLogLevel(LogLevel::TRACE);
    return RUN_ALL_TESTS();
}
