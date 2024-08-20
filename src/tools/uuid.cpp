#include "uuid.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace cv_infer
{

std::string GenerateUUID()
{
    std::random_device rd;
    std::mt19937_64 eng(rd());  // 使用随机设备和Mersenne Twister算法引擎
    std::uniform_int_distribution<unsigned long long> distr;

    // 生成8个随机的unsigned long long值
    unsigned long long random_bits[8];
    for (auto& bits : random_bits)
    {
        bits = distr(eng);
    }

    // 将这些值格式化为UUID的形式
    std::stringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << random_bits[0]
       << '-' << std::setw(4) << std::setfill('0') << (random_bits[0] >> 32)
       << '-' << std::setw(4) << std::setfill('0') << (random_bits[1] >> 48)
       << '-' << std::setw(4) << std::setfill('0') << (random_bits[1] >> 16)
       << '-' << std::setw(12) << std::setfill('0') << random_bits[1];

    return ss.str();
}
}  // namespace cv_infer