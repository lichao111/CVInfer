#include "engine/trt_infer.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

#include "signal/signal.h"
#include "tools/logger.h"
#include "tools/timer.h"

namespace cv_infer::trt
{
bool TrtEngine::LoadModel(const std::string& model, bool device_preprocess)
{
    if (device_preprocess)
    {
        EnableDevicePreProcess();
    }
    // check onnx
    if (not std::filesystem::exists(model))
    {
        LOGE("Model file not exist: [%s]", model.c_str());
        return false;
    }
    auto trt_engine = GetEngineName(model, MaxBatchSize, Precision);
    // check trt
    if (not std::filesystem::exists(trt_engine))
    {
        // build trt
        if (not BuildEngin(model, trt_engine))
        {
            LOGE("Build engine failed");
            return false;
        }
    }
    // load trt
    return LoadEngine(trt_engine, device_preprocess);
}

// https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#import_onnx_c
bool TrtEngine::BuildEngin(const std::string& src_onnx, const std::string& dst_engine)
{
    Timer Timer("BuildEngine");
    Timer.StartTimer();
    LOGI("Build engine from [%s] to [%s]", src_onnx.c_str(), dst_engine.c_str());

    // 1. create builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(Logger));
    if (builder == nullptr)
    {
        LOGE("Create builder failed");
    }

    // explicit batch vs implicit batch, this is explict batch
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(
        1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (network == nullptr)
    {
        LOGE("Create network failed");
    }

    // 2. create parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, Logger));
    if (parser == nullptr)
    {
        LOGE("Create parser failed");
    }
    // parser->setFlag(nvonnxparser::OnnxParserFlag::kNATIVE_INSTANCENORM);

    // 3. parse onnx from file . another way is parse from memory, Had our onnx model file been encrypted, this approach
    // would allow us to first decrypt the buffer.
    if (not parser->parseFromFile(src_onnx.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        LOGE("Parse onnx file [%s] failed", src_onnx.c_str());
        return false;
    }

    // 4. register a single optimization profile
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (config == nullptr)
    {
        LOGE("Create builder config failed");
        return false;
    }

    auto          opt_profile = builder->createOptimizationProfile();
    const int32_t num_inputs  = network->getNbInputs();
    for (int32_t i = 0; i < num_inputs; ++i)
    {
        const auto input     = network->getInput(i);
        const auto inputName = input->getName();
        const auto inputDims = input->getDimensions();
        int32_t    inputC    = inputDims.d[1];
        int32_t    inputH    = inputDims.d[2];
        int32_t    inputW    = inputDims.d[3];

        // Specify the optimization profile
        opt_profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
                                   nvinfer1::Dims4(1, inputC, inputH, inputW));
        opt_profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
                                   nvinfer1::Dims4(1, inputC, inputH, inputW));
        opt_profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
                                   nvinfer1::Dims4(MaxBatchSize, inputC, inputH, inputW));
    }
    config->addOptimizationProfile(opt_profile);

    // default is all memory
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    if (Precision == PrecisonType::FP16)
    {
        if (not builder->platformHasFastFp16())
        {
            LOGW("Platform does not support FP16");
        }
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else if (Precision == PrecisonType::INT8)
    {
        if (not builder->platformHasFastInt8())
        {
            LOGW("Platform does not support INT8");
        }
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    // 5. set profile stream
    cudaStream_t profileStream;

    int least_priotiry  = 0;
    int greate_priority = 0;

    CheckCudaErrorCode(cudaDeviceGetStreamPriorityRange(&least_priotiry, &greate_priority));
    LOGI("Least priority = %d, Greate priority = %d", least_priotiry, greate_priority);
    CheckCudaErrorCode(cudaStreamCreateWithPriority(&profileStream, cudaStreamNonBlocking, 0));
    config->setProfileStream(profileStream);

    // 6. Build the engine
    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (plan == nullptr)
    {
        LOGE("Build engine failed");
        return false;
    }

    // 7. serialize the engine, then save to file
    std::ofstream outfile(dst_engine, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());

    LOGI("Success, saved engine to %s", dst_engine.c_str());
    CheckCudaErrorCode(cudaStreamDestroy(profileStream));
    Timer.EndTimer();
    return true;
}

bool TrtEngine::LoadEngine(const std::string& engine, bool device_preprocess)
{
    std::ifstream   file(engine, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
    {
        throw std::runtime_error("Unable to read engine file");
    }

    // doc: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#perform-inference
    TrtRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(Logger));
    if (TrtRuntime == nullptr)
    {
        LOGE("Create runtime failed");
        return false;
    }

    TrtEngine = std::unique_ptr<nvinfer1::ICudaEngine>(TrtRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (TrtEngine == nullptr)
    {
        LOGE("Deserialize engine failed");
        return false;
    }

    TrtContext = std::unique_ptr<nvinfer1::IExecutionContext>(TrtEngine->createExecutionContext());
    if (TrtContext == nullptr)
    {
        LOGE("Create context failed");
        return false;
    }

    Buffers.resize(TrtEngine->getNbIOTensors());

    // Create a cuda stream
    cudaStream_t stream;
    CheckCudaErrorCode(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));

    for (int i = 0; i < TrtEngine->getNbIOTensors(); ++i)
    {
        auto tensor_name = TrtEngine->getIOTensorName(i);
        // process input
        if (TrtEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT)
        {
            LOGD("Tensor name: [%s], type is [Input] ", tensor_name);
            auto dims = TrtEngine->getTensorShape(tensor_name);
            if (dims.d[0] == -1)
            {
                LOGD("Tensor name: [%s], is dynamic batch", tensor_name);
                DynamicBatch = true;
            }
            else
            {
                MaxBatchSize = 1;
            }
            auto input_size = MaxBatchSize * dims.d[1] * dims.d[2] * dims.d[3] * sizeof(float);

            // malloc gpu memory for input[i]
            CheckCudaErrorCode(cudaMallocAsync(&Buffers[i], input_size, stream));
            InputDims.emplace_back(dims);
            InputNames.push_back(tensor_name);
            if (device_preprocess)
            {
                PreProcessBuffers.push_back((float*)(Buffers[i]));
            }
            else
            {
                PreProcessBuffers.push_back(static_cast<float*>(malloc(input_size)));
            }
        }
        else if (TrtEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            LOGD("Tensor name: [%s], type is [Output] ", tensor_name);
            // process output
            auto dims = TrtEngine->getTensorShape(tensor_name);

            std::uint32_t output_float = 1;
            for (int j = 1; j < dims.nbDims; ++j)  // ignore batch size, use max batch size
            {
                output_float *= dims.d[j];
            }
            OutputsLen.push_back(output_float);
            CheckCudaErrorCode(cudaMallocAsync(&Buffers[i], output_float * MaxBatchSize * sizeof(float), stream));
            OutputNames.push_back(tensor_name);
        }
        else
        {
            LOGW("Unknow tensor io mode");
        }
    }

    Outputs.resize(OutputsLen.size());

    CheckCudaErrorCode(cudaStreamSynchronize(stream));
    CheckCudaErrorCode(cudaStreamDestroy(stream));

    return true;
}

std::string TrtEngine::GetEngineName(const std::string& onnx_file, std::uint8_t max_batch_size, PrecisonType precision)
{
    auto GetPrecisionStr = [](PrecisonType precision) -> std::string
    {
        switch (precision)
        {
            case PrecisonType::INT8:
                return "int8";
            case PrecisonType::FP32:
                return "fp32";
            case PrecisonType::FP16:
                return "fp16";
            default:
                LOGW("Unknow precision type");
                return "fp32";
        }
    };
    auto prefix = onnx_file.substr(0, onnx_file.find_last_of('.'));
    auto batch  = std::to_string(max_batch_size);
    auto preci  = GetPrecisionStr(precision);
    auto suffix = ".trt";
    return prefix + "." + batch + "." + preci + suffix;
}

std::vector<std::vector<float>> TrtEngine::Forwards(const std::vector<cv::Mat>& input_signals)
{
    const auto num_inputs = InputDims.size();
    if (input_signals.size() != num_inputs)
    {
        LOGE("Input signals size not match, expect [%d], but got [%d]", num_inputs, input_signals.size());
        return {};
    }

    // auto batch_size = static_cast<std::int32_t>()
    CostTimerPre.StartTimer();
    if (not PreProcessFunc(input_signals, PreProcessBuffers))
    {
        LOGE("PreProcess failed");
        return {};
    }
    CostTimerPre.EndTimer("Preprocess");

    // create cuda stream for inference
    cudaStream_t stream;
    CheckCudaErrorCode(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));

    // copy to gpu memory [preprocessbuffer -> buffer]
    if (not DevicePreProcess)
    {
        for (int i = 0; i < num_inputs; ++i)
        {
            auto batch_size = 1;  // TODO: dynamic batch size

            auto size = batch_size * InputDims[i].d[1] * InputDims[i].d[2] * InputDims[i].d[3] * sizeof(float);
            CheckCudaErrorCode(cudaMemcpyAsync(Buffers[i], PreProcessBuffers[i], size, cudaMemcpyHostToDevice, stream));
        }
    }

    for (int i = 0; i < num_inputs; ++i)
    {
        nvinfer1::Dims4 input_dims = {1, InputDims[i].d[1], InputDims[i].d[2], InputDims[i].d[3]};
        TrtContext->setInputShape(InputNames[i].c_str(), input_dims);
        auto name = InputNames[i].c_str();
        if (auto ret = TrtContext->inferShapes(1, &(name)); ret != 0)  // ??
        {
            LOGE("error inferShapes ret =[%d]", ret);
        }
    }

    // Ensure all dynamic bindings have been defined.
    if (!TrtContext->allInputDimensionsSpecified())
    {
        LOGE("fatal !! Not all input dimensions specified");
        return {};
    }

    // pass trt buffers for input and output
    for (int i = 0; i < num_inputs; ++i)
    {
        if (not TrtContext->setTensorAddress(InputNames[i].c_str(), Buffers[i]))
        {
            LOGE("Set tensor address failed, name = [%s]", InputNames[i].c_str());
            return {};
        }
    }

    for (int i = 0; i < OutputNames.size(); ++i)
    {
        if (not TrtContext->setTensorAddress(OutputNames[i].c_str(), Buffers[num_inputs + i]))
        {
            LOGE("Set tensor address failed, name = [%s]", OutputNames[i].c_str());
            return {};
        }
    }

    // Execute the inference
    if (not TrtContext->enqueueV3(stream))
    {
        LOGE("Enqueue failed");
        return {};
    }

    // copy output from gpu memory to cpu memory
    for (int i = 0; i < Outputs.size(); ++i)
    {
        auto batch_size = 1;  // TODO: dynamic batch size

        auto size = OutputsLen[i] * batch_size * sizeof(float);
        Outputs[i].resize(size);
        CheckCudaErrorCode(
            cudaMemcpyAsync(Outputs[i].data(), Buffers[num_inputs + i], size, cudaMemcpyDeviceToHost, stream));
    }

    // Synchronize the cuda stream
    CheckCudaErrorCode(cudaStreamSynchronize(stream));
    CheckCudaErrorCode(cudaStreamDestroy(stream));

    CostTimerPost.StartTimer();
    auto ret = PostProcessFunc(Outputs);
    CostTimerPost.EndTimer("Postprocess");
    return ret;
}

void TrtEngine::CheckCudaErrorCode(cudaError_t code)
{
    if (code != 0)
    {
        std::string errMsg = "CUDA operation failed with code = [" + std::to_string(code) + "], name = [" +
                             cudaGetErrorName(code) + "], with message = [" + cudaGetErrorString(code) + "]";
        LOGE("%s", errMsg.c_str());
        throw std::runtime_error(errMsg);
    }
}
}  // namespace cv_infer::trt