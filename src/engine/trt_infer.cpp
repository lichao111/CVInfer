#include "engine/trt_infer.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

#include "engine/math.h"
#include "signal/signal.h"
#include "tools/logger.h"
#include "tools/timer.h"

namespace cv_infer::engine
{
bool TrtEngine::LoadModel(const std::string& model)
{
    if (not std::filesystem::exists(model))
    {
        LOGE("Model file not exist: [%s]", model.c_str());
        return false;
    }
    auto trt_engine = GetEngineName(model, MaxBatchSize, Precision);
    if (not std::filesystem::exists(trt_engine))
    {
        if (not BuildEngin(model, trt_engine))
        {
            LOGE("Build engine failed");
            return false;
        }
    }
    return LoadEngine(trt_engine);
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
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
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

bool TrtEngine::LoadEngine(const std::string& engine)
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

    // allocate gpu memory for input and output TODO: use dynamic memory
    for (int i = 0; i < TrtEngine->getNbIOTensors(); ++i)
    {
        auto tensor_name = TrtEngine->getIOTensorName(i);
        // process input
        if (TrtEngine->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT)
        {
            LOGD("Tensor name: [%s], type is [Input] ", tensor_name);
            auto dims = TrtEngine->getTensorShape(tensor_name);
            auto size =
                MaxBatchSize * dims.d[1] * dims.d[2] * dims.d[3] * sizeof(float);  // TODO: check is dynamic batch
            CheckCudaErrorCode(cudaMallocAsync(&Buffers[i], size, stream));
            InputDims.emplace_back(dims);
            PreProcessBuffers.push_back(static_cast<float*>(malloc(size)));
            InputNames.push_back(tensor_name);
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
            TensorNameMap[i] = tensor_name;
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

// TODO: 此函数组要换为纯虚函数 不同的模型需要不同的前处理 临时实现为通道前置
bool TrtEngine::PreProcess(const std::vector<SignalBasePtr>& input_signals)
{
    const auto num_inputs = InputDims.size();
    if (input_signals.size() != num_inputs)
    {
        LOGE("Input signals size not match, expect [%d], but got [%d]", num_inputs, input_signals.size());
        return false;
    }

    int input_index = 0;
    for (const auto& signal : input_signals)
    {
        if (signal->GetSignalType() != SignalType::SIGNAL_IMAGE_BGR)
        {
            LOGE("Input signal type not match, expect [SIGNAL_IMAGE_BGR], but got [%d]", signal->GetSignalType());
            return false;
        }

        auto image = std::dynamic_pointer_cast<SignalImageBGR>(signal);

        auto width   = image->Val.cols;
        auto height  = image->Val.rows;
        auto channel = image->Val.channels();

        std::vector<float> mean{128, 128, 128};
        std::vector<float> scale{128, 128, 128};

        ConverHWC2CHWMeanStd((image->Val).data, height, width, channel, mean.data(), scale.data(),
                             PreProcessBuffers[input_index]);
    }

    return true;
}

std::vector<std::vector<float>> TrtEngine::Forwards(const std::vector<SignalBasePtr>& input_signals,
                                                    std::vector<SignalBasePtr>&       output_signals)
{
    const auto num_inputs = InputDims.size();
    if (input_signals.size() != num_inputs)
    {
        LOGE("Input signals size not match, expect [%d], but got [%d]", num_inputs, input_signals.size());
        return {};
    }

    // auto batch_size = static_cast<std::int32_t>()
    if (not PreProcess(input_signals))
    {
        LOGE("PreProcess failed");
        return {};
    }

    // create cuda stream for inference
    cudaStream_t stream;
    CheckCudaErrorCode(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0));

    // copy to gpu memory [preprocessbuffer -> buffer]
    for (int i = 0; i < num_inputs; ++i)
    {
        auto batch_size = 1;  // TODO: dynamic batch size

        auto size = batch_size * InputDims[i].d[1] * InputDims[i].d[2] * InputDims[i].d[3] * sizeof(float);
        CheckCudaErrorCode(cudaMemcpyAsync(Buffers[i], PreProcessBuffers[i], size, cudaMemcpyHostToDevice, stream));
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

    return PostProcess();
}

void bigmeshgrid(int height, int width, float* xg, float* yg)
{
    for (int y_i = 0; y_i < height; ++y_i)
    {
        for (int x_i = 0; x_i < width; ++x_i)
        {
            xg[y_i * width + x_i] = x_i;
            yg[y_i * width + x_i] = y_i;
        }
    }
}

template <class T>
struct BigSortElement
{
    BigSortElement(){};
    BigSortElement(T v, unsigned int i) : value(v), index(i){};
    T            value;
    unsigned int index;
};

template <typename T>
struct BigDescendingSort
{
    typedef T ElementType;
    bool      operator()(const BigSortElement<T>& a, const BigSortElement<T>& b) { return a.value > b.value; }
};

std::vector<unsigned int> bigsort(std::vector<std::vector<float>>& data)
{
    // num*5
    std::vector<BigSortElement<float>> temp_vector(data.size());
    unsigned int                       index = 0;
    for (unsigned int i = 0; i < data.size(); ++i)
    {
        temp_vector[i] = BigSortElement<float>(data[i][4], i);
    }

    // sort
    BigDescendingSort<float> compare_op;
    std::sort(temp_vector.begin(), temp_vector.end(), compare_op);

    std::vector<unsigned int> result_index(data.size());
    index = 0;
    typename std::vector<BigSortElement<float>>::iterator iter, iend(temp_vector.end());
    for (iter = temp_vector.begin(); iter != iend; ++iter)
    {
        result_index[index] = ((*iter).index);
        index++;
    }

    return result_index;
}

std::vector<float> bigget_ious(std::vector<std::vector<float>>& all_bbox, std::vector<float>& target_bbox,
                               std::vector<unsigned int> order, unsigned int offset)
{
    std::vector<float> iou_list;
    for (unsigned int i = offset; i < order.size(); ++i)
    {
        int   index    = order[i];
        float inter_x1 = std::max(all_bbox[index][0], target_bbox[0]);
        float inter_y1 = std::max(all_bbox[index][1], target_bbox[1]);

        float inter_x2 = std::min(all_bbox[index][2], target_bbox[2]);
        float inter_y2 = std::min(all_bbox[index][3], target_bbox[3]);

        float inter_w = std::max(inter_x2 - inter_x1, 0.0f);
        float inter_h = std::max(inter_y2 - inter_y1, 0.0f);

        float inter_area = inter_w * inter_h;
        float a_area     = (all_bbox[index][2] - all_bbox[index][0]) * (all_bbox[index][3] - all_bbox[index][1]);
        float b_area     = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]);
        float iou        = inter_area / (a_area + b_area - inter_area);
        iou_list.push_back(iou);
    }

    return iou_list;
}

std::vector<unsigned int> bignms(std::vector<std::vector<float>>& dets, float thresh)
{
    std::vector<unsigned int> order = bigsort(dets);
    std::vector<unsigned int> keep;

    while (order.size() > 0)
    {
        unsigned int index = order[0];
        keep.push_back(index);
        if (order.size() == 1)
        {
            break;
        }

        std::vector<float>        check_ious = bigget_ious(dets, dets[index], order, 1);
        std::vector<unsigned int> remained_order;
        for (int i = 0; i < check_ious.size(); ++i)
        {
            if (check_ious[i] < thresh)
            {
                remained_order.push_back(order[i + 1]);
            }
        }
        order = remained_order;
    }
    return keep;
}

std::vector<std::vector<float>> TrtEngine::PostProcess()
{
    float  level_hw[]      = {64, 96, 32, 48, 16, 24};
    float  level_strides[] = {8, 16, 32};
    int    level_num       = 3;
    int    offset          = 0;
    float  x_scale         = 1280 / 768.0f;
    float  y_scale         = 720 / 512.0f;
    auto   data            = Outputs[0];
    float* temp_data       = new float[data.size()];
    memcpy(temp_data, data.data(), sizeof(float) * data.size());
    for (int level_i = 0; level_i < level_num; ++level_i)
    {
        int h      = level_hw[level_i * 2 + 0];
        int w      = level_hw[level_i * 2 + 1];
        int stride = level_strides[level_i];

        float* xg = new float[h * w];
        float* yg = new float[h * w];
        bigmeshgrid(h, w, xg, yg);

        for (int start_i = offset; start_i < offset + h * w; ++start_i)
        {
            temp_data[start_i * 7 + 0] = (data[start_i * 7 + 0] + xg[start_i - offset]) * stride;
            temp_data[start_i * 7 + 1] = (data[start_i * 7 + 1] + yg[start_i - offset]) * stride;

            temp_data[start_i * 7 + 2] = exp(data[start_i * 7 + 2]) * stride;
            temp_data[start_i * 7 + 3] = exp(data[start_i * 7 + 3]) * stride;
        }

        delete[] xg;
        delete[] yg;
        offset += h * w;
    }

    // 3
    int                             num = 1;
    std::vector<std::vector<float>> person_bboxes;
    std::vector<std::vector<float>> ball_bboxes;
    for (int i = 0; i < 8064; ++i)
    {
        float* ptr        = temp_data + i * 7;
        float  cx         = ptr[0];
        float  cy         = ptr[1];
        float  w          = ptr[2];
        float  h          = ptr[3];
        float  obj_pred   = ptr[4];
        float  cls_0_pred = ptr[5];
        float  cls_1_pred = ptr[6];

        // obj_pred 0.2 best
        if (obj_pred > 0.1)
        {
            if (cls_0_pred > cls_1_pred && cls_0_pred > 0.5)
            {
                person_bboxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls_0_pred});
            }
            else if (cls_0_pred < cls_1_pred && cls_1_pred > 0.3)
            {
                ball_bboxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cls_1_pred});
            }
        }
    }

    if (person_bboxes.size() > 0)
    {
        // preson
        std::vector<unsigned int> filter_person_index;
        filter_person_index = bignms(person_bboxes, 0.2);
        std::vector<std::vector<float>> filter_person_bboxes;
        for (int i = 0; i < filter_person_index.size(); ++i)
        {
            filter_person_bboxes.push_back(person_bboxes[filter_person_index[i]]);
        }
        person_bboxes = filter_person_bboxes;
    }
    if (ball_bboxes.size() > 0)
    {
        // ball
        std::vector<unsigned int> filter_ball_index;
        filter_ball_index = bignms(ball_bboxes, 0.01);
        std::vector<std::vector<float>> filter_ball_bboxes;
        for (int i = 0; i < filter_ball_index.size(); ++i)
        {
            filter_ball_bboxes.push_back(ball_bboxes[filter_ball_index[i]]);
        }
        ball_bboxes = filter_ball_bboxes;
    }
    // 合并结果
    int person_num          = person_bboxes.size();
    int ball_num            = ball_bboxes.size();
    int person_and_ball_num = person_num + ball_num;

    std::vector<std::vector<float>> bboxes(person_and_ball_num, std::vector<float>(5, 0.0));

    float labels[person_and_ball_num];

    for (int i = 0; i < person_and_ball_num; ++i)
    {
        if (i < person_bboxes.size())
        {
            bboxes[i][0] = person_bboxes[i][0] * x_scale;
            bboxes[i][1] = person_bboxes[i][1] * y_scale;
            bboxes[i][2] = person_bboxes[i][2] * x_scale;
            bboxes[i][3] = person_bboxes[i][3] * y_scale;
            bboxes[i][4] = person_bboxes[i][4];

            labels[i] = 0;
        }
        else
        {
            bboxes[i][0] = ball_bboxes[i - person_num][0] * x_scale;
            bboxes[i][1] = ball_bboxes[i - person_num][1] * y_scale;
            bboxes[i][2] = ball_bboxes[i - person_num][2] * x_scale;
            bboxes[i][3] = ball_bboxes[i - person_num][3] * y_scale;
            bboxes[i][4] = ball_bboxes[i - person_num][4];

            labels[i] = 1;
        }
    }

    delete[] temp_data;
    return bboxes;
}

bool TrtEngine::UnloadModel() { return true; }

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
}  // namespace cv_infer::engine