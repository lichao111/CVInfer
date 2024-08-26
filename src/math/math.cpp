#include "math.h"

namespace cv_infer
{
void ConverHWC2CHWMeanStd(const unsigned char* src, int h, int w, int c, const float* mean, const float* scale,
                          float* dst)
{
    for (int i = 0; i < c; i++)
    {
        for (int j = 0; j < h; j++)
        {
            for (int k = 0; k < w; k++)
            {
                dst[i * h * w + j * w + k] = ((float)(src[j * w * c + k * c + i] - mean[i])) / scale[i];
            }
        }
    }
}
}  // namespace cv_infer