#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

namespace cv_infer
{
// 1. 构造函数指定线程数量
// 2. 析构函数join所有线程
// 3. 提交任务，返回future
// 4. 线程池中的线程不断从任务队列中取任务执行
class ThreadPool
{
public:
    ThreadPool(std::uint8_t thread_count = 1) : ThreadCount(thread_count) {}
    virtual ~ThreadPool()
    {
        Running = false;
        Condition.notify_all();  // notify all threads to stop, nor all thread
                                 // will wait task forever
        for (auto& worker : Workers)
        {
            worker.join();
        }
    };

    bool SetThreadCount(std::uint8_t thread_count)
    {
        if (Running)
        {
            return false;
        }
        ThreadCount = thread_count;
        return true;
    }

    void Start()
    {
        Running = true;
        for (size_t i = 0; i < ThreadCount; ++i)
            Workers.emplace_back(
                [this]
                {
                    for (;;)
                    {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(this->Mutex);
                            this->Condition.wait(lock,
                                                 [this] {
                                                     return (not Running) or
                                                            (not Tasks.empty());
                                                 });
                            if ((not Running) and Tasks.empty())
                            {
                                return;
                            }
                            task = std::move(Tasks.front());

                            Tasks.pop();
                        }

                        task();
                    }
                });
    }
    void Stop() { Running = false; }

    template <typename Func, typename... Args>
    auto Commit(Func&& f, Args&&... args)
        -> std::future<typename std::result_of<Func(Args...)>::type>
    {
        using return_type = typename std::result_of<Func(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()> >(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();
        {
            std::lock_guard<std::mutex> lock{Mutex};
            if (not Running)
            {
                throw std::runtime_error("ThreadPool is stopped.");
            }
            Tasks.emplace([task]() { (*task)(); });
        }
        Condition.notify_one();
        return result;
    }

private:
    std::uint8_t            ThreadCount{1};
    std::mutex              Mutex;
    std::condition_variable Condition;
    std::atomic_bool        Running{false};

    std::queue<std::function<void()> > Tasks;
    std::vector<std::thread>           Workers;
};

}  // namespace cv_infer