#pragma once

#include <NvInfer.h>
#include <NvInferRuntimeBase.h>
#include <NvOnnxParser.h>

#include <unordered_map>

#include "engine/engine_base.h"
#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer::engine
{
enum class PrecisonType
{
    INT8,
    FP32,
    FP16,
};

class NvInferLoggerC : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        switch (severity)
        {
            case Severity::kINTERNAL_ERROR:
                LOGE("TRT INTERNAL_ERROR: %s", msg);
                break;
            case Severity::kERROR:
                LOGE("TRT ERROR: %s", msg);
                break;
            case Severity::kWARNING:
                LOGW("TRTWARNING: %s", msg);
                break;
            case Severity::kINFO:
                LOGI("TRT INFO: %s", msg);
                break;
            case Severity::kVERBOSE:
                LOGT("VERBOSE: %s", msg);
                break;
            default:
                LOGI("TRT UNKNOW: %s", msg);
                break;
        }
    }
};

class TrtEngine : public EngineBase
{
public:
    bool                            LoadModel(const std::string& model);
    std::vector<std::vector<float>> Forwards(const std::vector<SignalBasePtr>& input_signals,
                                             std::vector<SignalBasePtr>&       output_signals);
    bool                            UnloadModel();

protected:
    bool        LoadEngine(const std::string& engine);
    bool        BuildEngin(const std::string& src_onnx, const std::string& dst_engine);
    std::string GetEngineName(const std::string& onnx_file, std::uint8_t max_batch_size, PrecisonType precision);
    void        CheckCudaErrorCode(cudaError_t code);
    bool        PreProcess(const std::vector<SignalBasePtr>& input_signals);
    std::vector<std::vector<float>> PostProcess();

private:
    NvInferLoggerC Logger;
    std::uint8_t   MaxBatchSize{1};
    PrecisonType   Precision{PrecisonType::FP16};

    std::unique_ptr<nvinfer1::IRuntime>          TrtRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine>       TrtEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> TrtContext;
    std::vector<void*>                           Buffers;            // hold the input and ouput buffer [gpu]
    std::vector<std::uint32_t>                   OutputsLen;         // hold the output size
    std::vector<float*>                          PreProcessBuffers;  // hold the pre-process buffer [cpu]

    std::unordered_map<std::int32_t, std::string> TensorNameMap;
    std::vector<nvinfer1::Dims>                   InputDims;
    std::vector<std::vector<float>>               Outputs;

    std::vector<std::string> InputNames;
    std::vector<std::string> OutputNames;
};
}  // namespace cv_infer::engine