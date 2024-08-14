#include "node_base.h"

namespace cv_infer
{
bool NodeBase::bind(std::vector<std::reference_wrapper<SignalQue>> Inputs,
                    std::vector<std::reference_wrapper<SignalQue>> Outputs)
{
    InputCount  = Inputs.size();
    OutputCount = Outputs.size();
    InputList   = Inputs;
    OutputList  = Outputs;
    return true;
}
}  // namespace cv_infer