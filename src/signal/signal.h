#pragma once

#include <tools/queue.h>

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "tools/queue.h"

namespace cv_infer
{
enum class SignalType
{
    SIGNAL_UNKNOWN = -1,
    SIGNAL_INT8T,
    SIGNAL_UINT8T,
    SIGNAL_INT16T,
    SIGNAL_UINT16T,
    SIGNAL_INT32T,
    SIGNAL_UINT32T,
    SIGNAL_INT64T,
    SIGNAL_UINT64T,
    SIGNAL_FLOAT32T,
    SIGNAL_FLOAT64T,
    SIGNAL_BOOL,
    SIGNAL_STRING,
    SIGNAL_TENSOR,
    SIGNAL_BBOX,
    SIGNAL_BBOXES,
    SIGNAL_KEYPOINTS,
    SIGNAL_IMAGE_BGR,
    SIGNAL_IMAGE_RGB,
    SIGNAL_IMAGE_GRAY,
    SIGNAL_IMAGE_RGBA,
    SIGNAL_IMAGE_BGRA,
    SIGNAL_IMAGE_YUV,
};

struct SignalBase
{
    SignalBase() : SigType{SignalType::SIGNAL_UNKNOWN} {}
    SignalBase(SignalType sig_type) : SigType{sig_type} {}
    virtual ~SignalBase() = default;
    SignalType GetSignalType() const { return SigType; }

    SignalType    SigType{SignalType::SIGNAL_UNKNOWN};
    std::uint64_t FrameIdx{0};
};

template <typename T, SignalType Type, typename = std::enable_if<std::is_arithmetic_v<T>>>
struct SigalArithMetric : public SignalBase
{
    SigalArithMetric(T value) : SignalBase(Type), Val(value) {}
    virtual ~SigalArithMetric() override = default;

    T Val{0};
};

struct SignalString : public SignalBase
{
    SignalString(const std::string &value) : SignalBase(SignalType::SIGNAL_STRING), Val(value) {}
    SignalString(std::string &&value) : SignalBase(SignalType::SIGNAL_STRING), Val(std::move(value)) {}
    virtual ~SignalString() override = default;

    std::string Val;
};

struct SigalBool : public SignalBase
{
    SigalBool(bool value) : SignalBase(SignalType::SIGNAL_BOOL), Val(value) {}
    virtual ~SigalBool() override = default;

    bool Val{false};
};

struct SignalBBox : public SignalBase
{
    SignalBBox(float x_min, float y_min, float x_max, float y_max)
        : SignalBase(SignalType::SIGNAL_BBOX), Xmin(x_min), Ymin(y_min), Xmax(x_max), Ymax(y_max)
    {
    }
    virtual ~SignalBBox() override = default;

    float Xmin{0.0f};
    float Ymin{0.0f};
    float Xmax{0.0f};
    float Ymax{0.0f};
};

struct SignalBBoxes : public SignalBase
{
    SignalBBoxes(const std::vector<std::array<float, 4>> &bboxes) : SignalBase(SignalType::SIGNAL_BBOX), Val(bboxes)
    {
        if (bboxes.empty())
        {
            throw std::invalid_argument("The input bboxes is empty");
        }
    }
    virtual ~SignalBBoxes() override = default;

    std::vector<std::array<float, 4>> Val;
};

struct SignalKeyPoints : public SignalBase
{
    SignalKeyPoints(const std::array<std::pair<float, float>, 33> &keypoints)
        : SignalBase(SignalType::SIGNAL_KEYPOINTS), Val(keypoints)
    {
        if (keypoints.size() != MaxKeyPoints)
        {
            throw std::invalid_argument("The number of keypoints, " + std::to_string(keypoints.size()) +
                                        " is not equal to " + std::to_string(MaxKeyPoints));
        }
    }
    virtual ~SignalKeyPoints() override = default;

    std::array<std::pair<float, float>, 33> Val;
    static constexpr size_t                 MaxKeyPoints = 33;
};

struct SignalImageBGR : public SignalBase
{
    SignalImageBGR(const cv::Mat &image) : SignalBase(SignalType::SIGNAL_IMAGE_BGR), Val(image)
    {
        if (image.empty())
        {
            throw std::invalid_argument("The input image is empty");
        }
    }
    virtual ~SignalImageBGR() override = default;

    cv::Mat Val;
};

struct SignalImageRGB : public SignalBase
{
    SignalImageRGB(const cv::Mat &image) : SignalBase(SignalType::SIGNAL_IMAGE_RGB), Val(image)
    {
        if (image.empty())
        {
            throw std::invalid_argument("The input image is empty");
        }
    }
    virtual ~SignalImageRGB() override = default;

    cv::Mat Val;
};

using SignalBasePtr    = std::shared_ptr<SignalBase>;
using SignalQue        = Queue<SignalBasePtr>;
using SignalQueRefList = std::vector<std::reference_wrapper<SignalQue>>;
using SignalQueList    = std::vector<SignalQue>;

SignalQueRefList GetQueRef(SignalQueList &input_signals);
bool             IsSignalQueRefListReady(const SignalQueRefList &input_signals);

// 所有的信号队列都不为空时，返回信号队列中的第一个信号组成的列表
std::vector<SignalBasePtr> GetSignaList(const SignalQueRefList &input_signals);

}  // namespace cv_infer