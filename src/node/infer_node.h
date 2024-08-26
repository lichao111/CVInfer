#pragma once

#include "node_base.h"
#include "signal/signal.h"
#include "tools/logger.h"

namespace cv_infer
{
template <typename ModelType, typename EngineType>
class InferNode : public NodeBase
{
public:
    InferNode();
    virtual ~InferNode();

    bool Init(const std::string& model) { return Model.LoadModel(model); }

    virtual std::vector<SignalBasePtr> Worker(std::vector<SignalBasePtr> input_signals) override
    {
        std::vector<SignalBasePtr> output_signals;
        for (auto& input_signal : input_signals)
        {
            if (input_signal->GetSignalType() != SignalType::SIGNAL_IMAGE_BGR)
            {
                LOGE("Input signal type not match, expect [%d], but got [%d]",
                     static_cast<int>(SignalType::SIGNAL_IMAGE_BGR), static_cast<int>(input_signal->GetSignalType()));
                return {};
            }
            auto output_data = Model.Infer(input_signal);
            for (const auto output_date : output_data)
            {
                output_signals.push_back(std::make_shared<Signal>(output_date));
            }
        }
        return output_signals;
    }

private:
    ModelType<EngineType> Model;
};
}  // namespace cv_infer