#include "signal.h"

#include <algorithm>

namespace cv_infer
{

bool IsSignalQueRefListReady(const SignalQueRefList &input_signals)
{
    return not std::any_of(input_signals.begin(), input_signals.end(),
                           [](const auto &que) { return que.get().Empty(); });
}

std::vector<SignalBasePtr> GetSignaList(const SignalQueRefList &input_signals)
{
    if (not IsSignalQueRefListReady(input_signals))
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

}  // namespace cv_infer
