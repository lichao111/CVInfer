#include "logger.h"

#include <bit>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace cv_infer
{
LogLevel Logger::Level = LogLevel::INFO;

void Logger::__log_print__(LogLevel level, const char* file, int line,
                           const char* fmt, ...)
{
    if (level < Level) return;
    va_list vl;
    va_start(vl, fmt);
    auto now = GetCurrentTime();
    auto fn  = GetFileName(file);
    char buffer[2048];

    int n = snprintf(buffer, sizeof(buffer), "[%s]", now.c_str());

    if (level == LogLevel::ERROR)
    {
        n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[31m%s\033[0m]",
                      GetLevelString(level));
    }
    else if (level == LogLevel::WARN)
    {
        n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[33m%s\033[0m]",
                      GetLevelString(level));
    }
    else if (level == LogLevel::INFO)
    {
        n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[35m%s\033[0m]",
                      GetLevelString(level));
    }
    else if (level == LogLevel::TRACE)
    {
        n += snprintf(buffer + n, sizeof(buffer) - n, "[\033[34m%s\033[0m]",
                      GetLevelString(level));
    }
    else
    {
        n += snprintf(buffer + n, sizeof(buffer) - n, "[%s]",
                      GetLevelString(level));
    }

    n += snprintf(buffer + n, sizeof(buffer) - n, "[%s:%d]", fn.c_str(), line);
    n += vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
    va_end(vl);
    fprintf(stdout, "%s\n", buffer);
    fflush(stdout);
}

const char* Logger::GetLevelString(LogLevel level)
{
    switch (level)
    {
        case LogLevel::TRACE:
            return "TRACE";
        case LogLevel::DEBUG:
            return "DEBUG";
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARN:
            return "WARN";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::string Logger::GetCurrentTime()
{
    auto now          = std::chrono::system_clock::now();
    auto in_time_t    = std::chrono::system_clock::to_time_t(now);
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()) %
                        1000;
    std::stringstream ss;  // Create a stringstream
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << milliseconds.count();
    return ss.str();
}

std::string Logger::GetFileName(const char* file)
{
    std::string fn(file);
    auto        pos = fn.find_last_of('/');
    if (pos != std::string::npos)
    {
        fn = fn.substr(pos + 1);
    }
    return fn;
}
}  // namespace cv_infer