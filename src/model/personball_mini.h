#pragma once

#include "personball.h"

namespace cv_infer
{
template <typename EngineType>
class PersonBallMini : public PersonBall<EngineType>
{
public:
    PersonBallMini()
    {
        this->InferWidth   = 384;
        this->InferHeight  = 256;
        this->LevelHW      = {32, 48, 16, 24, 8, 12};
        this->LevelStrides = {8, 16, 32};
        this->LevelNum     = 3;
        this->OutputSize   = 2016;
    }
};

}  // namespace cv_infer