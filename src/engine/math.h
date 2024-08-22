#pragma once

namespace cv_infer::engine
{
void ConverHWC2CHWMeanStd(const unsigned char* src, int h, int w, int c, const float* mean, const float* scale,
                          float* dst);
}  // namespace cv_infer::engine