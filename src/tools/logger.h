#pragma once

#include <string>
namespace cv_infer
{
#ifdef __ADNROID__
#define LOG_TAG "CVInfer"
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#elif defined(__linux__)

enum class LogLevel
{
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class Logger
{
public:
    static void        __log_print__(LogLevel level, const char* file, int line,
                                     const char* fmt, ...);
    static const char* GetLevelString(LogLevel level);
    static void        SetLogLevel(LogLevel level) { Level = level; }
    static std::string GetCurrentTime();
    static std::string GetFileName(const char* file);

private:
    Logger()                         = default;
    ~Logger()                        = default;
    Logger(const Logger&)            = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&)                 = delete;
    Logger& operator=(Logger&&)      = delete;

    static LogLevel Level;
};

#define LOGT(...) \
    Logger::__log_print__(LogLevel::TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define LOGD(...) \
    Logger::__log_print__(LogLevel::DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOGI(...) \
    Logger::__log_print__(LogLevel::INFO, __FILE__, __LINE__, __VA_ARGS__)
#define LOGW(...) \
    Logger::__log_print__(LogLevel::WARN, __FILE__, __LINE__, __VA_ARGS__)
#define LOGE(...) \
    Logger::__log_print__(LogLevel::ERROR, __FILE__, __LINE__, __VA_ARGS__)

#endif

}  // namespace cv_infer