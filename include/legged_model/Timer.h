#pragma once
#include <functional>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <thread>
#include <atomic>

using namespace std::chrono;

namespace LeggedAI
{

class Timer {
public:
    using Callback = std::function<void(int64_t)>;

    enum class Mode {
        Realtime,  // ✅ wall-clock 定时触发（真实时间）
        SlowSim    // ✅ 外部 step 推进（慢速仿真）
    };

    Timer(int period_ms, Callback cb, Mode mode = Mode::Realtime)
        : period_ms_(period_ms), cb_(std::move(cb)), mode_(mode),
          last_tick_(-1), start_tick_(-1), acc_ms_(0),
          executed_count_(0), expected_count_(0),
          wall_running_(false) {}

    ~Timer() {
        stop_wall_timer();
    }

    // ========== 外部驱动：SlowSim 才使用 ==========
    void step(int64_t tick_ms) {
        // ✅ Realtime 完全不用 step，避免双触发
        if (mode_ == Mode::Realtime) {
            std::cerr << "[Timer] Warning: step() called in Realtime mode, ignored." << std::endl;
            return;
        }

        // -------- SlowSim Mode --------
        if (last_tick_ < 0 || tick_ms < last_tick_) {
            last_tick_      = tick_ms;
            start_tick_     = tick_ms;
            acc_ms_         = 0;
            executed_count_ = 0;
            expected_count_ = 0;
            start_wall_     = std::chrono::steady_clock::now();
            wall_running_.store(true);
            return;
        }

        int64_t dt = tick_ms - last_tick_;
        last_tick_ = tick_ms;

        if (dt <= 0) return;
        acc_ms_ += static_cast<int>(dt);

        bool crossed = false;
        while (acc_ms_ >= period_ms_) {
            acc_ms_ -= period_ms_;
            ++expected_count_;
            crossed = true;
        }

        if (crossed) {
            cb_(tick_ms);   // Drop policy: trigger once only
            ++executed_count_;
        }
    }

    // ========== wall-clock 驱动：Realtime 使用 ==========
    void start_wall_timer() {
        if (wall_running_.load()) return;

        start_wall_ = std::chrono::steady_clock::now();

        wall_running_.store(true);
        wall_thread_ = std::thread([this]() { this->wall_loop(); });
    }

    void stop_wall_timer() {
        if (!wall_running_.load()) return;
        wall_running_.store(false);
        if (wall_thread_.joinable()) wall_thread_.join();
    }

    bool wall_timer_running() const {
        return wall_running_.load();
    }

    double exec_freq() {
        if (start_wall_.time_since_epoch().count() == 0) {
            std::cout << "[Timer] Timmer has not started.\n";
            return -1.0;
        }
        double real_elapsed = duration<double>(steady_clock::now() - start_wall_).count();
        return executed_count_ / real_elapsed;
    }

    void report() const {
        if (start_wall_.time_since_epoch().count() == 0) {
            std::cout << "[Timer] Timmer has not started.\n";
            return;
        }

        double real_elapsed = duration<double>(steady_clock::now() - start_wall_).count();

        // SlowSim 才有 sim_elapsed 概念
        double sim_elapsed  = (mode_ == Mode::SlowSim && last_tick_ >= 0 && start_tick_ >= 0)
                                ? (last_tick_ - start_tick_) / 1000.0
                                : 0.0;

        double exec_freq     = executed_count_ / real_elapsed;
        double expected_freq = expected_count_ / real_elapsed;
        double time_ratio    = (sim_elapsed > 0.0) ? (sim_elapsed / real_elapsed) : 0.0;
        double hit_rate      = (expected_count_ > 0) ? double(executed_count_) / expected_count_ : 1.0;

        std::cout << "[Timer]["
                  << (mode_ == Mode::Realtime ? "Realtime(wall)" : "SlowSim(step)") << "]"
                  << " exec="     << exec_freq     << " Hz"
                  << " expected=" << expected_freq << " Hz"
                  << " time_ratio=" << time_ratio
                  << " hit_rate="   << hit_rate
                  << " (executed="  << executed_count_
                  << ", expected="  << expected_count_ << ")"
                  << std::endl;
    }

private:
    void wall_loop() {
        using clock = std::chrono::steady_clock;
        auto next = clock::now();

        // ✅ Realtime：每个周期都“期望一次并执行一次”
        while (wall_running_.load()) {
            next += std::chrono::milliseconds(period_ms_);

            int64_t wall_tick = duration_cast<milliseconds>(clock::now() - start_wall_).count();

            ++expected_count_;
            cb_(wall_tick);
            ++executed_count_;

            std::this_thread::sleep_until(next);
        }
    }

private:
    int period_ms_;
    Callback cb_;
    Mode mode_;

    // SlowSim(step) 相关
    int64_t last_tick_;
    int64_t start_tick_;
    int acc_ms_;

    // 统计（两种模式通用）
    int64_t executed_count_;
    int64_t expected_count_;
    std::chrono::steady_clock::time_point start_wall_{};

    // Realtime(wall) 相关
    std::atomic<bool> wall_running_;
    std::thread wall_thread_;

};

} // namespace LeggedAI
