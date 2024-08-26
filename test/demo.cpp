#include <thread>
#include <vector>

#include "../src/engine/engine_base.h"
#include "../src/engine/trt_infer.h"
#include "../src/model/personball.h"
#include "../src/node/decoder_node.h"
#include "../src/node/encoder_node.h"
#include "../src/node/infer_node.h"
#include "../src/pipeline/pipeline_base.h"
#include "../src/tools/logger.h"
#include "../src/tools/threadpool.h"
#include "../src/tools/timer.h"
#include "../src/tools/version.h"

using namespace cv_infer;
using namespace std::chrono_literals;
int main()
{
    Logger::SetLogLevel(LogLevel::TRACE);
    LOGI("version: %s", LIB_VERSION);
    LOGI("branch: %s", BUILD_BRANCH);
    LOGI("commit: %s", BUILD_COMMIT);

    Timer timer{"test"};

    auto source  = "/workspace/github/CVInfer/test/2024_08_19_15_53_58.mp4";
    auto dst_url = "demo_output.mp4";
    auto decoder = std::make_shared<DecoderNode>();
    auto encoder = std::make_shared<EncoderNode>();
    auto infer   = std::make_shared<InferNode<PersonBall, trt::TrtEngine>>();
    if (not decoder->Init(source))
    {
        LOGE("decoder init failed");
        return -1;
    }
    if (not encoder->Init(dst_url))
    {
        LOGE("encoder init failed");
        return -1;
    }
    if (not infer->Init("/workspace/github/CVInfer/test/personball_512_768_best_1_0.onnx"))
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

    std::this_thread::sleep_for(10s);
    pipeline->Stop();
    return 0;
}