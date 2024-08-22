#pragma once

#include "node_base.h"
#include "signal/signal.h"

namespace cv_infer
{
template <typename T>
class InferNode : public NodeBase
{
public:
    InferNode();
    virtual ~InferNode();

    bool Init(const std::string& model) { return ModelEngine.LoadModel(model); }

    virtual std::vector<SignalBasePtr> Worker(std::vector<SignalBasePtr> input_signals) override
    {
        std::vector<SignalBasePtr> output_signals;
        for (auto& input_signal : input_signals)
        {
            auto output_data = ModelEngine.Infer(input_signal);
            output_signals.push_back(output_data);
        }
        return output_signals;
    }

private:
    T ModelEngine;
};
}  // namespace cv_infer