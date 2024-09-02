#include <gtest/gtest.h>

#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "node/decoder_node.h"
#include "node/encoder_node.h"
#include "pipeline/pipeline_base.h"
#include "signal/signal.h"

using namespace cv_infer;
using namespace std::chrono_literals;
/*
TEST(runTests, decoder_rtsp)
{
    std::string source = "rtsp://192.168.41.211:554/id=1&type=0";
    auto        node   = std::make_shared<DecoderNode>();
    EXPECT_TRUE(node->Init(source));
    EXPECT_TRUE(node->Start());
    auto signal_que_list = node->GetOutputList();
    std::this_thread::sleep_for(1s);
    for (const auto& que : signal_que_list)
    {
        int frame_index = 0;
        while (not que.get().Empty())
        {
            SignalBasePtr signal;
            que.get().Pop(signal);
            EXPECT_TRUE(signal->GetSignalType() == SignalType::SIGNAL_IMAGE_BGR);
            auto frame = std::dynamic_pointer_cast<SignalImageBGR>(signal);
            if (frame)
            {
                auto image = frame->Val;
                LOGT("frame width = %d, height = %d", image.cols, image.rows);
                std::string file_name = "rtsp_frame_" + std::to_string(frame_index) + ".jpg";
                cv::putText(image, "frame_index: " + std::to_string(frame_index++), cv::Point(40, 40),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                cv::imwrite(file_name, image);
            }
            if (frame_index >= 10)
            {
                break;
            }
        }
    }
    EXPECT_TRUE(node->Stop());
}
*/

TEST(runTests, decoder_mp4)
{
    std::string source         = "/workspace/github/CVInfer/test/2024_08_19_15_53_58.mp4";
    auto        node           = std::make_shared<DecoderNode>();
    auto        input_signals  = std::make_shared<SignalQue>();
    auto        output_signals = std::make_shared<SignalQue>();
    EXPECT_TRUE(node->Init(source));
    EXPECT_TRUE(node->AddOutputs(output_signals));
    EXPECT_TRUE(node->Start());
    std::this_thread::sleep_for(1s);
    int frame_index = 0;
    while (not output_signals->Empty())
    {
        SignalBasePtr signal;
        output_signals->Pop(signal);
        EXPECT_TRUE(signal->GetSignalType() == SignalType::SIGNAL_IMAGE_BGR);
        auto frame = std::dynamic_pointer_cast<SignalImageBGR>(signal);
        if (frame)
        {
            auto image = frame->Val;
            LOGI("frame width = %d, height = %d", image.cols, image.rows);
            std::string file_name = "file_frame_" + std::to_string(frame_index) + ".jpg";
            cv::putText(image, "frame_index: " + std::to_string(frame_index++), cv::Point(40, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::imwrite(file_name, image);
        }
        if (frame_index >= 10)
        {
            break;
        }
    }

    EXPECT_TRUE(node->Stop());
}

TEST(runTests, encoder)
{
    std::string source   = "/workspace/github/CVInfer/test/2024_08_19_15_53_58.mp4";
    std::string out_url  = "output.mp4";
    auto        decoder  = std::make_shared<DecoderNode>();
    auto        encoder  = std::make_shared<EncoderNode>();
    auto        pipeline = std::make_shared<PipelineBase>();
    EXPECT_TRUE(decoder->Init(source));
    EXPECT_TRUE(encoder->Init(out_url));
    EXPECT_TRUE(pipeline->BindAll({decoder, encoder}));
    EXPECT_TRUE(pipeline->Start());
    std::this_thread::sleep_for(2s);
    EXPECT_TRUE(pipeline->Stop());
    LOGI("encoder done");
}