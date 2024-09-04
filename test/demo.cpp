#include <cmath>
#include <thread>
#include <vector>

#include "../src/engine/trt_infer.h"
#include "../src/model/personball_mini.h"
#include "../src/node/decoder_node.h"
#include "../src/node/encoder_node.h"
#include "../src/node/infer_node.h"
#include "../src/pipeline/pipeline_base.h"
#include "../src/tools/logger.h"
#include "../src/tools/version.h"

using namespace cv_infer;
using namespace std::chrono_literals;

bool test_personball_mini(const std::string& src, const std::string& dst)
{
    auto decoder = std::make_shared<DecoderNode>();
    auto encoder = std::make_shared<EncoderNode>();
    auto infer   = std::make_shared<InferNode<PersonBallMini, trt::TrtEngine>>();
    if (not decoder->Init(src))
    {
        LOGE("decoder init failed");
        return -1;
    }
    if (not encoder->Init(dst))
    {
        LOGE("encoder init failed");
        return -1;
    }
    if (not infer->Init("../test/personball_best_1_0.onnx"))
    {
        LOGE("infer init failed");
        return -1;
    }

    auto pipeline = std::make_unique<PipelineBase>("test_pipeline");
    if (not pipeline->BindAll({decoder, infer, encoder}))
    {
        LOGE("pipeline bind failed");
        return -1;
    }
    pipeline->Start();

    std::this_thread::sleep_for(15s);
    pipeline->Stop();
    return true;
}

bool test_personball(const std::string& src, const std::string& dst)
{
    auto decoder = std::make_shared<DecoderNode>();
    auto encoder = std::make_shared<EncoderNode>();
    auto infer   = std::make_shared<InferNode<PersonBall, trt::TrtEngine>>();
    if (not decoder->Init(src))
    {
        LOGE("decoder init failed");
        return -1;
    }
    if (not encoder->Init(dst))
    {
        LOGE("encoder init failed");
        return -1;
    }
    if (not infer->Init("../test/personball_512_768_best_1_0.onnx"))
    {
        LOGE("infer init failed");
        return -1;
    }

    auto pipeline = std::make_unique<PipelineBase>("test_pipeline");
    if (not pipeline->BindAll({decoder, infer, encoder}))
    {
        LOGE("pipeline bind failed");
        return -1;
    }
    pipeline->Start();

    std::this_thread::sleep_for(15s);
    pipeline->Stop();
    return true;
}

int main(int argc, char* argv[])
{
    Logger::SetLogLevel(LogLevel::INFO);
    LOGI("version: %s", LIB_VERSION);
    LOGI("branch: %s", BUILD_BRANCH);
    LOGI("commit: %s", BUILD_COMMIT);

    std::string test_targrt{"personball_mini"};
    if (argc >= 2)
    {
        test_targrt = argv[1];
    }

    std::string src{"../test/mi50_output_view_.mp4"};
    std::string dst{"demo_output.mp4"};

    if (test_targrt == "personball_mini")
    {
        if (not test_personball_mini(src, dst))
        {
            LOGE("test_personball_mini failed");
        };
    }
    else if (test_targrt == "personball")
    {
        if (not test_personball(src, dst))
        {
            LOGE("test_personball failed");
        }
    }
    return 0;
}