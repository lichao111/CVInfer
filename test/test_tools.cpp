#include <gtest/gtest.h>

#include <atomic>
#include <future>

#include "tools/logger.h"
#include "tools/queue.h"
#include "tools/threadpool.h"

using namespace cv_infer;

TEST(runTests, ThreadPool)
{
    ThreadPool pool(4);
    pool.Start();
    std::vector<std::future<void>> futures;
    std::atomic_int                start_num = 0;
    for (int i = 0; i < 1000; i++)
    {
        auto func = [&i, &start_num]() -> void { start_num++; };
        auto ret  = pool.Commit(func);
        futures.push_back(std::move(ret));
    }
    for (auto& future : futures)
    {
        future.get();
    }
    pool.Stop();
    EXPECT_EQ(start_num, 1000);
}

TEST(runTests, Que)
{
    Queue<int> que;
    for (int i = 0; i < 100; i++)
    {
        que.Push(i);
    }
    for (int i = 0; i < 100; i++)
    {
        int val;
        que.Pop(val);
        EXPECT_EQ(val, i);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}