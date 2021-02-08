#pragma once

#include <map>
#include <string>
#include <iostream>
#include <chrono>

#include <assert.h>

#include "singleton.h"
#include "error_format.h"
#include <stdexcept>

namespace timer = std::chrono;

#ifdef defined(PROFILE)
const bool profile_flag = true;
#else
const bool profile_flag = true;
#endif

class Profiler final : public Singleton<Profiler>{
	friend class Singleton<Profiler>;

private:
	std::map< std::string, timer::high_resolution_clock::time_point > start_points;
	std::map< std::string, timer::duration<double> > duration_list;
	Profiler(){};

public:
	~Profiler() {};

	void push(std::string name) {
		if (profile_flag) {
			if (this->start_points.count(name)) {
				//Warning: Resetting currently running timer
			}
			this->start_points[name] = timer::high_resolution_clock::now();
		}
	}

	void pop(std::string name) {
		if (profile_flag) {
			if (this->start_points.count(name) == 0) {
				throw std::runtime_error(Formatter() << "Profiler: Attempting to pop non-existent time region, " << name);
			}

			timer::high_resolution_clock::time_point current_time = timer::high_resolution_clock::now();
			timer::high_resolution_clock::time_point start_time = this->start_points[name];
			timer::duration<double> time_block = timer::duration_cast<timer::duration<double>>(current_time - start_time);

			if (this->duration_list.count(name) == 0) {
				this->duration_list[name] = time_block;
			}
			else {
				this->duration_list[name] += time_block;
			}

			this->start_points.erase(name);
		}
	}

	void reset(std::string name = "NULL") {
		if (profile_flag) {
			if (name == "NULL") {
				this->duration_list.clear();
				this->start_points.clear();
			}
			else {
				if (this->start_points.count(name)) {
					this->start_points.erase(name);
				}
				if (this->duration_list.count(name)) {
					this->duration_list.erase(name);
				}
			}
		}
	}

	void print() {
		if (profile_flag) {
			for (auto x : duration_list) {
				std::cout << x.first << " : " << x.second.count() << " (s) " << std::endl;
			}
		}
	}

};

