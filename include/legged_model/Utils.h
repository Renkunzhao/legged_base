#pragma once
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

using namespace Eigen;

namespace legged {

inline double smoothstep(double t, size_t order = 5) {
    t = std::clamp(t, 0.0, 1.0);

    switch (order)
    {
        case 1: // Linear
            return t;

        case 3: // Cubic smoothstep (C¹ continuous)
            return t * t * (3.0 - 2.0 * t);  // 3t² - 2t³

        case 5: // Quintic smootherstep (C² continuous)
        {
            double t2 = t * t;
            double t3 = t2 * t;
            return t3 * (10.0 + t * (6.0 * t - 15.0));  // 6t⁵ -15t⁴ +10t³
        }

        default:
            throw std::invalid_argument("smoothstep: order must be 1, 3, or 5");
    }
}

inline double lerp(double t, const double& x0, const double& x1, size_t order = 5) {
    double s = smoothstep(t, order);
    return x0 + (x1 - x0) * s;
}

// Linear interpolate: t in [0,1]
template<class DerivedA, class DerivedB>
inline typename DerivedA::PlainObject
lerp(double t,
     const Eigen::MatrixBase<DerivedA>& x0,
     const Eigen::MatrixBase<DerivedB>& x1,
     size_t order = 5) {
    return (x0 + (x1 - x0) * smoothstep(t, order)).eval();  // return a plain (materialized) vector
}

// [xyzw]
inline VectorXd baseSlerp(double t,
                        const Eigen::VectorXd& qBaseDes,
                        const Eigen::VectorXd& qBaseInit, 
                        size_t order = 5) {
    // 1. Position interpolation
    Eigen::Vector3d base_pos = lerp(t, qBaseInit.head<3>(), qBaseDes.head<3>(), order).eval();

    // 2. Orientation interpolation
    Eigen::Quaterniond quat_init(qBaseInit[6], qBaseInit[3], qBaseInit[4], qBaseInit[5]); // w,x,y,z
    Eigen::Quaterniond quat_des(qBaseDes[6], qBaseDes[3], qBaseDes[4], qBaseDes[5]);     // w,x,y,z
    quat_init.normalize();
    quat_des.normalize();

    if (quat_init.dot(quat_des) < 0.0)
        quat_des.coeffs() *= -1.0;

    Eigen::Quaterniond quat_interp = quat_init.slerp(smoothstep(t, order), quat_des);
    quat_interp.normalize();

    // 3. Combine
    Eigen::VectorXd qBase(7);
    qBase << base_pos, quat_interp.vec(), quat_interp.w();
    return qBase;
}

// 将 YAML list 转换为 Eigen::VectorXd
inline Eigen::VectorXd yamlToEigenVector(const YAML::Node& node) {
    if (!node || !node.IsSequence()) {
        throw std::runtime_error("YAML node is not a valid sequence.");
    }
    std::vector<double> vec = node.as<std::vector<double>>();
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
}

// 将 std::vector<Eigen::VectorX> 拼成一个大的 VectorXd
template <typename VecT>
inline Eigen::VectorXd concatVectors(const std::vector<VecT>& vecs)
{
    static_assert(
        Eigen::internal::traits<VecT>::ColsAtCompileTime == 1,
        "VecT must be a column vector (Eigen::Matrix<..., 1>)"
    );

    // total size = sum of all vector sizes
    size_t total_size = 0;
    for (const auto& v : vecs) total_size += v.size();

    Eigen::VectorXd out(total_size);

    // copy data
    size_t offset = 0;
    for (const auto& v : vecs) {
        out.segment(offset, v.size()) = v;
        offset += v.size();
    }

    return out;
}

//
// 1) 主函数：按 sizes_vector 切
//
template <typename VecT>
inline std::vector<VecT> splitVectors(
    const Eigen::VectorXd& big,
    const std::vector<int>& sizes)
{
    static_assert(
        Eigen::internal::traits<VecT>::ColsAtCompileTime == 1,
        "VecT must be a column vector"
    );

    std::vector<VecT> out;
    out.reserve(sizes.size());

    size_t offset = 0;
    for (int len : sizes) {

        // 如果 VecT 是固定长度，检查一致性
        if constexpr (VecT::SizeAtCompileTime != Eigen::Dynamic) {
            assert(VecT::SizeAtCompileTime == len &&
                   "Fixed-size VecT does not match segment size");
        }

        VecT v(len);
        v = big.segment(offset, len);
        out.push_back(v);

        offset += len;
    }

    return out;
}

//
// 2) 重载：输入单个长度 n → 自动切分为 size = big.size() / n
//
template <typename VecT>
inline std::vector<VecT> splitVectors(
    const Eigen::VectorXd& big,
    int len_per_segment)
{
    assert(len_per_segment > 0);

    // 自动推断段数
    assert(big.size() % len_per_segment == 0 &&
           "big.size() must be divisible by len_per_segment");

    int n_segments = big.size() / len_per_segment;

    std::vector<int> sizes(n_segments, len_per_segment);

    return splitVectors<VecT>(big, sizes);
}

inline Eigen::VectorXd computeNominalEE3DofForces(const std::vector<bool>& contactFlag, double mass) {
    const double g = 9.81;
    const size_t n_legs = contactFlag.size();
    Eigen::VectorXd f_des = Eigen::VectorXd::Zero(3 * n_legs);

    // 支撑腿数量
    int n_contacts = 0;
    for (bool c : contactFlag)
        if (c) n_contacts++;

    // 空中情况（无支撑腿）
    if (n_contacts == 0)
        return f_des;

    // 均分竖向力
    double fz = mass * g / n_contacts;

    for (size_t i = 0; i < n_legs; ++i) {
        if (contactFlag[i]) {
            f_des[3*i + 2] = fz;  // 只分配竖向力
        }
    }

    return f_des;
}

// 输入：
//   vec_in     - 输入向量 (dim * n)
//   names_in   - 输入的名称 (size = n)
//   names_out  - 目标名称顺序 (size = n)
// 输出：
//   vec_out    - 输出向量 (dim * n)
inline void reorder(
            const std::vector<std::string> &names_in,
            const Eigen::VectorXd &vec_in,
            const std::vector<std::string> &names_out,
            Eigen::VectorXd &vec_out
        ) {
    int n_in = names_in.size();
    int n_out = names_out.size();
    if (n_in != n_out)
        throw std::invalid_argument("names_in and names_out must have same length");

    if (vec_in.size() % n_in != 0)
        throw std::invalid_argument("vec_in size must be divisible by names_in.size()");

    int dim = vec_in.size() / n_in;
    vec_out.resize(vec_in.size());

    // 建立名字 -> 索引映射
    std::unordered_map<std::string, int> name_to_idx;
    for (int i = 0; i < n_in; ++i) {
        name_to_idx[names_in[i]] = i;
    }

    // 重排
    for (int i = 0; i < n_out; ++i) {
        auto it = name_to_idx.find(names_out[i]);
        if (it == name_to_idx.end())
            throw std::invalid_argument("Name " + names_out[i] + " not found in names_in");

        int src_idx = it->second;
        vec_out.segment(i * dim, dim) = vec_in.segment(src_idx * dim, dim);
    }
}

// 规则：矩阵按“块”重排：
// - 行重排：mat_in 的行被分成 n 个块，每块 row_dim 行
// - 列重排：mat_in 的列被分成 n 个块，每块 col_dim 列

inline void reorder_rows(
    const std::vector<std::string>& names_in,
    const Eigen::Ref<const Eigen::MatrixXd>& mat_in,
    const std::vector<std::string>& names_out,
    Eigen::Ref<Eigen::MatrixXd> mat_out
) {
    const int n_in  = static_cast<int>(names_in.size());
    const int n_out = static_cast<int>(names_out.size());
    if (n_in != n_out)
        throw std::invalid_argument("reorder_rows: names_in and names_out must have same length");
    if (n_in == 0)
        throw std::invalid_argument("reorder_rows: names_in is empty");
    if (mat_in.rows() % n_in != 0)
        throw std::invalid_argument("reorder_rows: mat_in.rows() must be divisible by names_in.size()");

    const int row_dim = mat_in.rows() / n_in;

    // 输出尺寸检查
    if (mat_out.rows() != mat_in.rows() || mat_out.cols() != mat_in.cols())
        throw std::invalid_argument("reorder_rows: mat_out must have same shape as mat_in");

    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(static_cast<size_t>(n_in));
    for (int i = 0; i < n_in; ++i) name_to_idx[names_in[i]] = i;

    for (int i = 0; i < n_out; ++i) {
        auto it = name_to_idx.find(names_out[i]);
        if (it == name_to_idx.end())
            throw std::invalid_argument("reorder_rows: Name " + names_out[i] + " not found in names_in");

        const int src = it->second;
        mat_out.middleRows(i * row_dim, row_dim) =
            mat_in.middleRows(src * row_dim, row_dim);
    }
}

inline void reorder_cols(
    const std::vector<std::string>& names_in,
    const Eigen::Ref<const Eigen::MatrixXd>& mat_in,
    const std::vector<std::string>& names_out,
    Eigen::Ref<Eigen::MatrixXd> mat_out
) {
    const int n_in  = static_cast<int>(names_in.size());
    const int n_out = static_cast<int>(names_out.size());
    if (n_in != n_out)
        throw std::invalid_argument("reorder_cols: names_in and names_out must have same length");
    if (n_in == 0)
        throw std::invalid_argument("reorder_cols: names_in is empty");
    if (mat_in.cols() % n_in != 0)
        throw std::invalid_argument("reorder_cols: mat_in.cols() must be divisible by names_in.size()");

    const int col_dim = mat_in.cols() / n_in;

    // 输出尺寸检查
    if (mat_out.rows() != mat_in.rows() || mat_out.cols() != mat_in.cols())
        throw std::invalid_argument("reorder_cols: mat_out must have same shape as mat_in");

    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(static_cast<size_t>(n_in));
    for (int i = 0; i < n_in; ++i) name_to_idx[names_in[i]] = i;

    for (int i = 0; i < n_out; ++i) {
        auto it = name_to_idx.find(names_out[i]);
        if (it == name_to_idx.end())
            throw std::invalid_argument("reorder_cols: Name " + names_out[i] + " not found in names_in");

        const int src = it->second;
        mat_out.middleCols(i * col_dim, col_dim) =
            mat_in.middleCols(src * col_dim, col_dim);
    }
}

// 重载版本：用于 vector<bool> 的重排
inline void reorder(
    const std::vector<std::string> &names_in,
    const std::vector<bool> &vec_in,
    const std::vector<std::string> &names_out,
    std::vector<bool> &vec_out
) {
    int n_in = names_in.size();
    int n_out = names_out.size();

    if (n_in != n_out)
        throw std::invalid_argument("names_in and names_out must have same length");

    if ((int)vec_in.size() != n_in)
        throw std::invalid_argument("vec_in size must equal names_in size");

    vec_out.resize(n_out);

    // 建立名字 -> 索引
    std::unordered_map<std::string, int> name_to_idx;
    for (int i = 0; i < n_in; ++i)
        name_to_idx[names_in[i]] = i;

    // 重排
    for (int i = 0; i < n_out; ++i) {
        auto it = name_to_idx.find(names_out[i]);
        if (it == name_to_idx.end())
            throw std::invalid_argument("Name " + names_out[i] + " not found in names_in");

        int src_idx = it->second;
        vec_out[i] = vec_in[src_idx];  // vector<bool> OK
    }
}

} // namespace legged
