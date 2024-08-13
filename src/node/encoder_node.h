#pragma once

#include "node/node_base.h"

namespace cvinfer
{
class EncoderNode : public NodeBase
{
public:
    EncoderNode() = default;
    virtual ~EncoderNode() = default;
};
}  // namespace cvinfer