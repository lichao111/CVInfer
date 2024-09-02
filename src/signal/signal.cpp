#include "signal.h"

#include <algorithm>

namespace cv_infer
{

SignalQueRefList GetQueRef(SignalQueList &input_signals)
{
    SignalQueRefList que_ref_list;
    for (auto &que : input_signals)
    {
        que_ref_list.push_back(std::ref(que));
    }
    return que_ref_list;
}

bool IsSignalQueListReady(const SignalQueRefList &input_signals)
{
    return not std::any_of(input_signals.begin(), input_signals.end(),
                           [](const auto &que) { return que.get().Empty(); });
}

bool IsSignalQueListReady(const SignalQuePtrList &input_signals)
{
    return not std::any_of(input_signals.begin(), input_signals.end(), [](const auto &que) { return que->Empty(); });
}

std::vector<SignalBasePtr> GetSignalList(const SignalQueRefList &input_signals)
{
    if (not IsSignalQueListReady(input_signals))
    {
        return {};
    }
    std::vector<SignalBasePtr> signals;
    for (const auto &que : input_signals)
    {
        SignalBasePtr sig;
        que.get().Pop(sig);
        signals.push_back(sig);
    }
    return signals;
}

std::vector<SignalBasePtr> GetSignalList(const SignalQuePtrList &input_signals)
{
    if (not IsSignalQueListReady(input_signals))
    {
        return {};
    }
    std::vector<SignalBasePtr> signals;
    for (const auto &que : input_signals)
    {
        SignalBasePtr sig;
        que->Pop(sig);
        signals.push_back(sig);
    }
    return signals;
}

}  // namespace cv_infer
