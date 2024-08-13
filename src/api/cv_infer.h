#ifndef _CV_INFER_H_
#define _CV_INFER_H_

#ifdef _WIN32
#define CV_INFER_API __declspec(dllexport) extern "C"
#else
#define CV_INFER_API extern "C" __attribute__((visibility("default")))
#endif

typedef int     ErrorCode;
const ErrorCode EC_SUCCESS = 0;
const ErrorCode EC_FAILURE = -1;

#endif  // _CV_INFER_H_