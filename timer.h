
//Simple timer
//Auher£ºRanto
//Timer£º2017/5

#pragma once

#include <iostream>
#include <chrono>
#include <exception>


//Simple Timer class(Milliseconds)
class Timer
{
	using clock = std::chrono::steady_clock;
	using time_point = std::chrono::time_point<std::chrono::steady_clock>;
	using duration = std::chrono::milliseconds;

	clock clk;
	time_point beginTimePoint, endTimePoint;
	duration countedTime;
	double t;
	bool started;
	
public:
	Timer()
	{
		t = 0;
		started = false;
	}
	Timer(const Timer &timer) = delete;
	~Timer() = default;

	//Begin timing. Time is accumulating.
	void begin()
	{
		beginTimePoint = clk.now();
		started = true;
	}

	//End timing.
	void end()
	{
		if (started)
		{
			started = false;
			endTimePoint = clk.now();
			countedTime = std::chrono::duration_cast<duration>(endTimePoint - beginTimePoint);
			t += countedTime.count() / 1000.0;
		}
	}

	//Return counted time.
	double time() const noexcept
	{
		return t;
	}
	//Reset time to zero and stop timing.
	void reset() noexcept
	{
		this->end();
		t = 0;
	}
};

