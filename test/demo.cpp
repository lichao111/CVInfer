#include <thread>
#include <vector>

#include "../src/tools/logger.h"
#include "../src/tools/threadpool.h"
#include "../src/tools/timer.h"
#include "../src/tools/version.h"

using namespace cv_infer;
using namespace std::chrono_literals;
int main()
{
    Logger::SetLogLevel(LogLevel::TRACE);
    LOGI("version: %s", LIB_VERSION);
    LOGI("branch: %s", BUILD_BRANCH);
    LOGI("commit: %s", BUILD_COMMIT);

    Timer timer;
    for (int i = 0; i < 2; i++)
    {
        timer.StartTimer();
        std::this_thread::sleep_for(1s * i);
        timer.EndTimer();
    }
    timer.Reset();

    return 0;
}