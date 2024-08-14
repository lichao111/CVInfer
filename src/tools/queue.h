#pragma once

#include <mutex>
#include <queue>

namespace cv_infer
{
template <typename T>
class Queue
{
public:
    Queue()  = default;
    ~Queue() = default;

    bool Push(const T& item)
    {
        std::lock_guard<std::mutex> lock(Mutex);
        Que.push(item);
        if (Que.size() > MaxSize)
        {
            Que.pop();
        }
        return true;
    }

    bool Push(T&& item)
    {
        std::lock_guard<std::mutex> lock(Mutex);
        Que.push(std::move(item));
        if (Que.size() > MaxSize)
        {
            Que.pop();
        }
        return true;
    }

    bool Pop(T& item)
    {
        std::lock_guard<std::mutex> lock(Mutex);
        if (Que.empty())
        {
            return false;
        }
        item = Que.front();
        Que.pop();
        return true;
    }

    bool Empty()
    {
        std::lock_guard<std::mutex> lock(Mutex);
        return Que.empty();
    }

    size_t Size()
    {
        std::lock_guard<std::mutex> lock(Mutex);
        return Que.size();
    }

private:
    std::queue<T> Que;
    std::mutex    Mutex;
    std::uint64_t MaxSize{1000};
};
}  // namespace cv_infer