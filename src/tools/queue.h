#pragma once

#include <mutex>
#include <queue>

namespace cv_infer
{
template <typename T>
class Queue
{
public:
    Queue() = default;  // 默认构造函数

    // 复制构造函数
    Queue(const Queue& other)
    {
        std::lock_guard<std::mutex> lock(Mutex);
        Que     = other.Que;
        MaxSize = other.MaxSize;
    }

    // 赋值运算符
    Queue& operator=(const Queue& other)
    {
        if (this != &other)
        {
            std::lock_guard<std::mutex> lock1(Mutex, std::adopt_lock);
            // std::lock_guard<std::mutex> lock2(other.Mutex, std::adopt_lock);
            Que     = other.Que;
            MaxSize = other.MaxSize;
        }
        return *this;
    }

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
    std::uint64_t MaxSize{100000};
};
}  // namespace cv_infer