#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <vector>

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

class NodeImplTestBase : public NodeBase
{
public:
    NodeImplTestBase(std::size_t inputs, std::size_t outputs,
                     std::size_t thds = 1)
        : NodeBase(inputs, outputs, thds)
    {
    }

    virtual std::vector<SignalBasePtr> Worker(
        std::vector<SignalBasePtr> input_signals) override
    {
        std::vector<SignalBasePtr> output_signals;
        for (auto& sig : input_signals)
        {
            LOGI("NodeImpl::Worker() get signal in thread [%ld]",
                 std::this_thread::get_id());
            output_signals.push_back(std::move(sig));
        }

        return output_signals;
    }
};

class NodeImplTestArthMetric : public NodeBase
{
public:
    NodeImplTestArthMetric(std::size_t inputs, std::size_t outputs,
                           std::size_t thds = 1)
        : NodeBase(inputs, outputs, thds)
    {
    }
    virtual std::vector<SignalBasePtr> Worker(
        std::vector<SignalBasePtr> input_signals) override
    {
        std::vector<SignalBasePtr> output_signals;
        for (auto& sig : input_signals)
        {
            // check if sig type is SignalArithMetric
            if (sig->GetSignalType() != SignalType::SIGNAL_UINT8T)
            {
                LOGE("NodeImpl::Worker() get signal type error, excepted");
                continue;
            }

            LOGI("NodeImpl::Worker() get signal in thread [%ld]",
                 std::this_thread::get_id());
            output_signals.push_back(std::move(sig));
        }

        return output_signals;
    }
};

TEST(runTests, NodeBase)
{
    auto             input_count  = 4;
    auto             output_count = 4;
    auto             thread_count = 2;
    NodeImplTestBase node(input_count, output_count, thread_count);
    SignalQueList    input_signals;
    for (int i = 0; i < 4; i++)
    {
        input_signals.push_back(SignalQue());
    }
    SignalQueRefList input_signals_ref;
    for (auto& que : input_signals)
    {
        que.Push(std::make_shared<SignalBase>());
        input_signals_ref.push_back(std::ref(que));
    }
    node.SetInputs(input_signals_ref);
    node.Start();
    std::this_thread::sleep_for(10ms);
    node.Stop();
}

TEST(runTests, NodeBaseArthMetric)
{
    auto                   input_count  = 4;
    auto                   output_count = 4;
    auto                   thread_count = 2;
    NodeImplTestArthMetric node(input_count, output_count, thread_count);
    SignalQueList          input_signals;
    for (int i = 0; i < 4; i++)
    {
        input_signals.push_back(SignalQue());
    }
    SignalQueRefList input_signals_ref;
    for (auto& que : input_signals)
    {
        que.Push(std::make_shared<
                 SigalArithMetric<uint8_t, SignalType::SIGNAL_UINT8T>>(3));
        input_signals_ref.push_back(std::ref(que));
    }
    node.SetInputs(input_signals_ref);
    node.Start();
    std::this_thread::sleep_for(10ms);
    node.Stop();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Logger::SetLogLevel(LogLevel::TRACE);
    return RUN_ALL_TESTS();
}