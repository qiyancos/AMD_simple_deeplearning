#ifndef TEST_TIME_LOGGER_HPP
#define TEST_TIME_LOGGER_HPP

#include <vector>
#include <chrono>

using time_point = std::chrono::high_resolution_clock::time_point;
using system_clock = std::chrono::system_clock;
using std::chrono::duration_cast;
using microseconds = std::chrono::microseconds;

class TimeLogger{

private:
    time_point timeOld, timeNow;
public:
    TimeLogger(){
        timeOld = system_clock::now();
        timeNow = system_clock::now();
    }
    void record(){
        timeOld = timeNow;
        timeNow = system_clock::now();
    }
    uint32_t getGap(){
        return duration_cast<microseconds>(timeNow - timeOld).count();
    }
    uint32_t getGapNow(){
        record();
        return getGap();
    }
};

#endif
