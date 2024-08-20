#pragma once

#include <functional>
#include <unordered_map>

namespace cv_infer
{
enum class EventId
{
    FirstFrameDone = 0,
    OneFrameDone,
    AllFrameDone,
};
using EventCallbackFunc = std::function<void(const char*)>;
using EventCallbackMap  = std::unordered_map<EventId, EventCallbackFunc>;

}  // namespace cv_infer