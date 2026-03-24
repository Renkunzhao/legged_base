#include "legged_base/CsvLogger.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

CsvLogger::CsvLogger() = default;

CsvLogger& CsvLogger::getInstance() {
    static CsvLogger instance;
    return instance;
}

void CsvLogger::setCsvPath(const std::string& path) {
    csv_path_ = path;
}

static inline int64_t currentTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void CsvLogger::init() {
    start_time_ms_ = currentTimeMs();
}

void CsvLogger::update(int64_t time_ms, const std::string& name, double value) {
    timestamps_set_.insert(time_ms);
    data_[name][time_ms] = value;
}

void CsvLogger::update(int64_t time_ms, const std::string& name, int value) {
    update(time_ms, name, static_cast<double>(value));
}

void CsvLogger::update(int64_t time_ms, const std::string& name, const Eigen::Vector3d& vec) {
    update(time_ms, name + "_x", vec.x());
    update(time_ms, name + "_y", vec.y());
    update(time_ms, name + "_z", vec.z());
}

void CsvLogger::update(int64_t time_ms, const std::string& name, const Eigen::VectorXd& vec) {
    for (int i = 0; i < vec.size(); ++i) {
        update(time_ms, name + "_" + std::to_string(i), vec(i));
    }
}

void CsvLogger::update(int64_t time_ms, const std::string& name, const Eigen::MatrixXd& mat) {
    for (int r = 0; r < mat.rows(); ++r) {
        for (int c = 0; c < mat.cols(); ++c) {
            update(time_ms, name + "_" + std::to_string(r) + std::to_string(c), mat(r, c));
        }
    }
}

void CsvLogger::update(const std::string& name, double value) { update(currentTimeMs(), name, value); }
void CsvLogger::update(const std::string& name, int value) { update(currentTimeMs(), name, value); }
void CsvLogger::update(const std::string& name, const Eigen::Vector3d& vec) { update(currentTimeMs(), name, vec); }
void CsvLogger::update(const std::string& name, const Eigen::VectorXd& vec) { update(currentTimeMs(), name, vec); }
void CsvLogger::update(const std::string& name, const Eigen::MatrixXd& mat) { update(currentTimeMs(), name, mat); }

void CsvLogger::update(double time_sec, const std::string& name, double value) { update(timeToInt(time_sec), name, value); }
void CsvLogger::update(double time_sec, const std::string& name, int value) { update(timeToInt(time_sec), name, static_cast<double>(value)); }

void CsvLogger::update(double time_sec, const std::string& name, const Eigen::Vector3d& vec) {
    const int64_t t = timeToInt(time_sec);
    update(t, name + "_x", vec.x());
    update(t, name + "_y", vec.y());
    update(t, name + "_z", vec.z());
}

void CsvLogger::update(double time_sec, const std::string& name, const Eigen::VectorXd& vec) {
    const int64_t t = timeToInt(time_sec);
    for (int i = 0; i < vec.size(); ++i) {
        update(t, name + "_" + std::to_string(i), vec(i));
    }
}

void CsvLogger::update(double time_sec, const std::string& name, const Eigen::MatrixXd& mat) {
    const int64_t t = timeToInt(time_sec);
    update(t, name, mat);
}

void CsvLogger::save() {
    if (csv_path_.empty()) {
        std::cerr << "错误: CSV 文件路径未设置，请调用 setCsvPath()" << std::endl;
        return;
    }

    namespace fs = std::filesystem;

    try {
        const fs::path path(csv_path_);
        const fs::path dir = path.parent_path();

        if (!dir.empty() && !fs::exists(dir)) {
            std::cout << "目录不存在，自动创建: " << dir << std::endl;
            fs::create_directories(dir);
        }

        std::ofstream file(csv_path_);
        if (!file.is_open()) {
            std::cerr << "错误: 无法打开文件: " << csv_path_ << std::endl;
            return;
        }

        std::vector<std::string> headers{"time_s"};
        for (const auto& pair : data_) {
            headers.push_back(pair.first);
        }

        for (size_t i = 0; i < headers.size(); ++i) {
            file << headers[i] << (i == headers.size() - 1 ? "" : ",");
        }
        file << "\n";

        for (const int64_t t : timestamps_set_) {
            const double relative_time_s = (t - start_time_ms_) / 1000.0;
            file << std::fixed << std::setprecision(6) << relative_time_s;

            for (size_t i = 1; i < headers.size(); ++i) {
                const std::string& header = headers[i];
                const auto& var_data = data_.at(header);
                const auto it = var_data.find(t);
                file << ",";
                if (it != var_data.end()) {
                    file << std::fixed << std::setprecision(6) << it->second;
                }
            }
            file << "\n";
        }

        file.close();
        std::cout << "数据成功保存到: " << csv_path_ << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "保存 CSV 时发生错误: " << e.what() << std::endl;
    }
}

void CsvLogger::clear() {
    timestamps_set_.clear();
    data_.clear();
    start_time_ms_ = 0;
}
