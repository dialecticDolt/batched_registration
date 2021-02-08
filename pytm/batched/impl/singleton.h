#pragma once

template<typename T>
class Singleton {
	friend class fftHandlePC;
	friend class fftHandleNGF;
	friend class Profiler;
public:
	static T& instance() {
		static const std::unique_ptr<T> instance{ new T() };
		return *instance;
	}
private:
	Singleton() {};
	~Singleton() {};
	Singleton(const Singleton&) = delete;
	void operator = (const Singleton&) = delete;
};

