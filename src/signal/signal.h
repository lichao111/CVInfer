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
    SIGNAL_UNKNOWN    = -1,
    SIGNAL_INT8T      = 0,
    SIGNAL_UINT8T     = 1,
    SIGNAL_INT16T     = 2,
    SIGNAL_UINT16T    = 3,
    SIGNAL_INT32T     = 4,
    SIGNAL_UINT32T    = 5,
    SIGNAL_INT64T     = 6,
    SIGNAL_UINT64T    = 7,
    SIGNAL_FLOAT32T   = 8,
    SIGNAL_FLOAT64T   = 9,
    SIGNAL_BOOL       = 10,
    SIGNAL_STRING     = 11,
    SIGNAL_TENSOR     = 12,
    SIGNAL_BBOX       = 13,
    SIGNAL_KEYPOINTS  = 14,
    SIGNAL_IMAGE_BGR  = 15,
    SIGNAL_IMAGE_RGB  = 16,
    SIGNAL_IMAGE_GRAY = 17,
    SIGNAL_IMAGE_RGBA = 18,
    SIGNAL_IMAGE_BGRA = 19,
    SIGNAL_IMAGE_YUV  = 20,
};

struct SignalBase
{
    SignalBase() : SigType{SignalType::SIGNAL_UNKNOWN} {}
    SignalBase(SignalType sig_type) : SigType{sig_type} {}
    virtual ~SignalBase() = default;
    SignalType GetSignalType() const { return SigType; }

    SignalType SigType{SignalType::SIGNAL_UNKNOWN};
};

template <typename T, SignalType Type,
          typename = std::enable_if<std::is_arithmetic_v<T>>>
struct SigalArithMetric : public SignalBase
{
    SigalArithMetric(T value) : SignalBase(Type), Val(value) {}
    virtual ~SigalArithMetric() override = default;

    T Val{0};
};

struct SignalString : public SignalBase
{
    SignalString(const std::string &value)
        : SignalBase(SignalType::SIGNAL_STRING), Val(value)
    {
    }
    SignalString(std::string &&value)
        : SignalBase(SignalType::SIGNAL_STRING), Val(std::move(value))
    {
    }
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
        : SignalBase(SignalType::SIGNAL_BBOX),
          Xmin(x_min),
          Ymin(y_min),
          Xmax(x_max),
          Ymax(y_max)
    {
    }
    virtual ~SignalBBox() override = default;

    float Xmin{0.0f};
    float Ymin{0.0f};
    float Xmax{0.0f};
    float Ymax{0.0f};
};

struct SignalKeyPoints : public SignalBase
{
    SignalKeyPoints(const std::array<std::pair<float, float>, 33> &keypoints)
        : SignalBase(SignalType::SIGNAL_KEYPOINTS), Val(keypoints)
    {
        if (keypoints.size() != MaxKeyPoints)
        {
            throw std::invalid_argument(
                "The number of keypoints, " + std::to_string(keypoints.size()) +
                " is not equal to " + std::to_string(MaxKeyPoints));
        }
    }
    virtual ~SignalKeyPoints() override = default;

    std::array<std::pair<float, float>, 33> Val;
    static constexpr size_t                 MaxKeyPoints = 33;
};

struct SignalImageBGR : public SignalBase
{
    SignalImageBGR(const cv::Mat &image)
        : SignalBase(SignalType::SIGNAL_IMAGE_BGR), Val(image)
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
    SignalImageRGB(const cv::Mat &image)
        : SignalBase(SignalType::SIGNAL_IMAGE_RGB), Val(image)
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
using SignalQue        = Queue<std::shared_ptr<SignalBase>>;
using SignalQueRefList = std::vector<std::reference_wrapper<SignalQue>>;
using SignalQueList    = std::vector<SignalQue>;

bool IsSignalQueRefListReady(const SignalQueRefList &input_signals);
std::vector<SignalBasePtr> GetSignaList(const SignalQueRefList &input_signals);

}  // namespace cv_infer