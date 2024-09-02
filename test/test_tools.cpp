#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "node/node_base.h"
#include "pipeline/pipeline_base.h"
#include "signal/signal.h"
#include "tools/logger.h"
#include "tools/queue.h"
#include "tools/threadpool.h"
#include "tools/uuid.h"

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
    NodeImplTestBase(std::size_t inputs, std::size_t outputs) : NodeBase(inputs, outputs) {}

    virtual bool Worker() override
    {
        for (auto& sig_queue : InputList)
        {
            if (sig_queue->Empty())
            {
                return false;
            }
            SignalBasePtr sig;
            sig_queue->Pop(sig);
            OutputList[0]->Push(std::move(sig));
        }
        return true;
    }
};

TEST(runTests, NodeBase)
{
    auto             input_count  = 4;
    auto             output_count = 1;
    NodeImplTestBase node(input_count, output_count);
    SignalQuePtrList input_signals;
    SignalQuePtr     output_signals = std::make_shared<SignalQue>();

    for (int i = 0; i < input_count; i++)
    {
        input_signals.push_back(std::make_shared<SignalQue>());
    }

    for (auto& que : input_signals)
    {
        que->Push(std::make_shared<SignalBase>());
        node.AddInputs(que);
    }
    node.AddOutputs(output_signals);
    node.Start();
    std::this_thread::sleep_for(10ms);
    node.Stop();
}

TEST(runTests, UUID)
{
    for (auto i = 0; i < 10; ++i)
    {
        LOGI("uuid: [%s]", GenerateUUID().c_str());
    }
    ASSERT_TRUE(true);
}

class NodeImplIcr : public NodeBase
{
public:
    NodeImplIcr() : NodeBase(1, 1) {}
    virtual bool Worker() override
    {
        SignalBasePtr input_signals;
        if (not InputList[0]->Pop(input_signals))
        {
            return false;
        }

        // check if sig type is SignalArithMetric
        if (input_signals->GetSignalType() != SignalType::SIGNAL_UINT8T)
        {
            LOGE("NodeImpl::Worker() get signal type error, excepted");
            return true;
        }

        auto value =
            std::dynamic_pointer_cast<SigalArithMetric<uint8_t, SignalType::SIGNAL_UINT8T>>(input_signals)->Val;
        auto sigal_out = std::make_shared<SigalArithMetric<uint8_t, SignalType::SIGNAL_UINT8T>>(++value);
        OutputList[0]->Push(sigal_out);

        return true;
    }
};

TEST(runTests, NodeBaseIcr)
{
    std::shared_ptr<NodeBase> node1 = std::make_shared<NodeImplIcr>();
    std::shared_ptr<NodeBase> node2 = std::make_shared<NodeImplIcr>();
    std::shared_ptr<NodeBase> node3 = std::make_shared<NodeImplIcr>();
    std::shared_ptr<NodeBase> node4 = std::make_shared<NodeImplIcr>();

    SignalQuePtr input_signal  = std::make_shared<SignalQue>();
    SignalQuePtr output_signal = std::make_shared<SignalQue>();

    input_signal->Push(std::make_shared<SigalArithMetric<uint8_t, SignalType::SIGNAL_UINT8T>>(0));
    std::unique_ptr<PipelineBase> pipeline = std::make_unique<PipelineBase>("pipeline_icr");
    node1->AddInputs(input_signal);
    ASSERT_TRUE(pipeline->GetName() == "pipeline_icr");
    auto ret = pipeline->BindAll({node1, node2, node3, node4});
    node4->AddOutputs(output_signal);
    ASSERT_TRUE(ret);

    pipeline->Start();
    std::this_thread::sleep_for(40ms);
    while (not output_signal->Empty())
    {
        SignalBasePtr signal;
        output_signal->Pop(signal);
        auto value = std::dynamic_pointer_cast<SigalArithMetric<uint8_t, SignalType::SIGNAL_UINT8T>>(signal)->Val;
        ASSERT_EQ(value, 4);
    }
    pipeline->Stop();
}
