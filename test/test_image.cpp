#include <gtest/gtest.h>

#include "../src/engine/trt_infer.h"
#include "../src/node/encoder_node.h"
#include "../src/node/image_loader.h"
#include "../src/node/infer_node.h"
#include "../src/pipeline/pipeline_base.h"
#include "model/yolo.h"

using namespace cv_infer;
using namespace std::chrono_literals;

TEST(Pipeline, ImageLoader)
{
    auto image_loader = std::make_shared<ImageLoader>();
    EXPECT_TRUE(image_loader->Init("/workspace/github/CVInfer/test/mi50_run"));

    auto        encoder = std::make_shared<EncoderNode>();
    YoloType    type    = YoloType::YOLOV7;
    std::string model_path{"../test/yolov7.onnx"};

    type            = YoloType::YOLOV7;
    model_path      = "/workspace/github/CVInfer/test/yolov7.onnx";
    std::string dst = "./ut_output.mp4";

    auto infer = std::make_shared<InferNode<Yolo<trt::TrtEngine, YoloType::YOLOV7>>>();

    EXPECT_TRUE(encoder->Init(dst));
    EXPECT_TRUE(infer->Init(model_path));

    auto pipeline = std::make_unique<PipelineBase>("test_pipeline");
    EXPECT_TRUE(pipeline->BindAll({image_loader, infer, encoder}));
    EXPECT_TRUE(pipeline->Start());

    std::this_thread::sleep_for(15s);
    EXPECT_TRUE(pipeline->Stop());
}