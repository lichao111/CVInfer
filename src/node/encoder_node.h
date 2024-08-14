#pragma once

#include "node/node_base.h"

namespace cv_infer
{
class EncoderNode : public NodeBase
{
public:
    EncoderNode()          = default;
    virtual ~EncoderNode() = default;
};
}  // namespace cv_infer