#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>

#include "node/node_base.h"
#include "signal/signal.h"
#include "tools/logger.h"
#include "tools/queue.h"
#include "tools/threadpool.h"

using namespace cv_infer;
using namespace std::chrono_literals;
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

class NodeImplTest : public NodeBase
{
public:
    NodeImplTest(std::size_t inputs, std::size_t outputs, std::size_t thds = 1)
        : NodeBase(inputs, outputs, thds)
    {
    }

    virtual std::vector<SignalBasePtr> Worker(
        std::vector<SignalBasePtr> input_signals) override
    {
        for (auto& sig : input_signals)
        {
            LOGT("NodeImpl::Worker() get signal in thread [%ld]",
                 std::this_thread::get_id());
        }
        return {};
    }
};

TEST(runTests, NodeBase)
{
    NodeImplTest  node(4, 5, 2);
    SignalQueList input_signals;
    for (int i = 0; i < 4; i++)
    {
        input_signals.push_back(SignalQue());
    }
    SignalQueRefList input_signals_ref;
    for (auto& que : input_signals)
    {
        input_signals_ref.push_back(std::ref(que));
    }
    node.SetInputs(input_signals_ref);
    node.Start();
    node.Run();
    std::this_thread::sleep_for(std::chrono::seconds(5s));
    node.Stop();
}
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}